"""Streaming evaluation metrics for binary anomaly segmentation, computed in LOGIT space.

Scores are binned by logit, not probability. A probability histogram lumps every pixel with
p < 1/4096 into one bin, so scenarios where the model is confidently "no" everywhere collapse to
AUROC == 0.5 whatever the true ranking. Logit bins keep resolution in that region.

Reported at three levels: pooled over all pixels, per scenario (within the scenario, so its
anomaly-free frames supply the false-positive pressure) and per frame. ``empty`` pools the pixels
of frames without any anomaly to report the false-positive rate there at a fixed threshold.
All histograms are exact counts, so results are deterministic and mergeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

LOGIT_LO, LOGIT_HI, NUM_BINS = -24.0, 24.0, 12288  # bin width ~0.0039 logits


def _trapezoid(y, x):
    fn = getattr(np, "trapezoid", None) or np.trapz
    return float(fn(y, x))


class LogitCurve:
    """Histogram of logits for positive and negative pixels."""

    def __init__(self, lo: float = LOGIT_LO, hi: float = LOGIT_HI, num_bins: int = NUM_BINS):
        self.lo, self.hi, self.nb = lo, hi, num_bins
        self.pos = np.zeros(num_bins, dtype=np.int64)
        self.neg = np.zeros(num_bins, dtype=np.int64)

    # -- accumulation ---------------------------------------------------------
    def _bins(self, logits: torch.Tensor) -> torch.Tensor:
        x = logits.float().clamp(self.lo, self.hi - 1e-4)
        return ((x - self.lo) / (self.hi - self.lo) * self.nb).long()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """logits/targets: any matching shape. Targets are thresholded at 0.5."""
        logits = torch.as_tensor(logits).reshape(-1)
        t = torch.as_tensor(targets, device=logits.device).reshape(-1) > 0.5
        b = self._bins(logits)
        self.pos += torch.bincount(b[t], minlength=self.nb).cpu().numpy().astype(np.int64)
        self.neg += torch.bincount(b[~t], minlength=self.nb).cpu().numpy().astype(np.int64)

    def merged(self, other: "LogitCurve") -> "LogitCurve":
        out = LogitCurve(self.lo, self.hi, self.nb)
        out.pos, out.neg = self.pos + other.pos, self.neg + other.neg
        return out

    # -- derived quantities ---------------------------------------------------
    def _sweep(self):
        tp = np.cumsum(self.pos[::-1]).astype(np.float64)   # threshold high -> low
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)
        P, N = float(self.pos.sum()), float(self.neg.sum())
        return tp, fp, P, N

    def threshold_of(self, k: int) -> float:
        """Logit threshold represented by sweep index k (lower edge of that bin)."""
        width = (self.hi - self.lo) / self.nb
        return self.hi - (k + 1) * width

    def quantile(self, which: str, q: float) -> float:
        """Logit quantile of the positive ('pos') or negative ('neg') pixels."""
        h = self.pos if which == "pos" else self.neg
        total = h.sum()
        if total == 0:
            return float("nan")
        k = int(np.searchsorted(np.cumsum(h), q * total))
        return self.lo + (min(k, self.nb - 1) + 0.5) * (self.hi - self.lo) / self.nb

    def auroc(self) -> float:
        tp, fp, P, N = self._sweep()
        if P == 0 or N == 0:
            return float("nan")
        return _trapezoid(np.r_[0.0, tp / P, 1.0], np.r_[0.0, fp / N, 1.0])

    def ap(self) -> float:
        tp, fp, P, N = self._sweep()
        if P == 0:
            return float("nan")
        recall = tp / P
        precision = tp / np.maximum(tp + fp, 1)
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    def best_threshold(self, criterion: str = "dice") -> tuple[float, dict]:
        """Threshold (logit) maximising Dice or IoU on this curve, plus its metrics."""
        tp, fp, P, N = self._sweep()
        if P == 0:
            return float("nan"), {}
        fn = P - tp
        score = (2 * tp / np.maximum(2 * tp + fp + fn, 1) if criterion == "dice"
                 else tp / np.maximum(tp + fp + fn, 1))
        k = int(np.argmax(score))
        return self.threshold_of(k), self.at_index(k)

    def at_index(self, k: int) -> dict:
        tp, fp, P, N = self._sweep()
        tpk, fpk = tp[k], fp[k]
        fnk = P - tpk
        return {
            "threshold_logit": self.threshold_of(k),
            "dice": float(2 * tpk / max(2 * tpk + fpk + fnk, 1)),
            "iou": float(tpk / max(tpk + fpk + fnk, 1)),
            "precision": float(tpk / max(tpk + fpk, 1)),
            "recall": float(tpk / max(P, 1)),
            "fpr": float(fpk / max(N, 1)),
            "fp_pixels": int(fpk), "tp_pixels": int(tpk),
        }

    def at_threshold(self, logit_threshold: float) -> dict:
        """Exact-to-bin metrics when predicting positive for logit >= threshold."""
        width = (self.hi - self.lo) / self.nb
        k = int(np.clip(round((self.hi - logit_threshold) / width) - 1, 0, self.nb - 1))
        return self.at_index(k)

    def summary(self) -> dict:
        thr, best = self.best_threshold("dice")
        return {"ap": self.ap(), "auroc": self.auroc(),
                "n_pos": int(self.pos.sum()), "n_neg": int(self.neg.sum()),
                "best_dice": best.get("dice", float("nan")),
                "best_thr_logit": thr}


@dataclass
class EvalAccumulator:
    """Pooled + per-scenario + frame-level accumulators for one split.

    Every scenario contains frames with and without the anomaly, so per-scenario AP is
    computed within the scenario (its empty frames supply the false-positive pressure).
    ``empty`` pools the pixels of frames that contain no anomaly at all, to report the
    false-positive rate on anomaly-free frames at a fixed threshold.
    """
    pooled: LogitCurve = field(default_factory=LogitCurve)
    empty: LogitCurve = field(default_factory=LogitCurve)
    scen: dict = field(default_factory=dict)
    frame_scores: list = field(default_factory=list)
    frame_labels: list = field(default_factory=list)
    frame_scen: list = field(default_factory=list)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, masks: torch.Tensor, scenario_ids) -> None:
        """logits/masks: [B,1,H,W] at the SAME resolution."""
        for i, sid in enumerate(scenario_ids):
            lg, mk = logits[i], masks[i]
            has_anomaly = bool((mk > 0.5).any().item())
            self.pooled.update(lg, mk)
            self.scen.setdefault(sid, LogitCurve()).update(lg, mk)
            if not has_anomaly:
                self.empty.update(lg, mk)
            self.frame_scores.append(float(lg.max().item()))
            self.frame_labels.append(has_anomaly)
            self.frame_scen.append(sid)

    # -- reports --------------------------------------------------------------
    def scenario_table(self, threshold_logit: float | None = None):
        """Per-scenario metrics (AP/AUROC within the scenario)."""
        import pandas as pd
        rows = []
        for sid, c in sorted(self.scen.items()):
            row = {"scenario_id": sid, "gt_pixels": int(c.pos.sum())}
            if c.pos.sum() > 0:
                row.update(ap=c.ap(), auroc=c.auroc())
                thr, best = c.best_threshold("dice")
                row.update(best_dice=best["dice"], best_thr_logit=thr,
                           pos_logit_median=c.quantile("pos", 0.5),
                           pos_logit_p90=c.quantile("pos", 0.9),
                           neg_logit_median=c.quantile("neg", 0.5),
                           separation=c.quantile("pos", 0.5) - c.quantile("neg", 0.5))
                if threshold_logit is not None:
                    m = c.at_threshold(threshold_logit)  # within-scenario at a fixed thr
                    row.update(dice_at_thr=m["dice"], recall_at_thr=m["recall"],
                               precision_at_thr=m["precision"])
            rows.append(row)
        return pd.DataFrame(rows)

    def frame_metrics(self) -> dict:
        s, y = np.array(self.frame_scores), np.array(self.frame_labels)
        return {"frame_ap": average_precision(s, y), "frame_auroc": roc_auc(s, y),
                "frames": int(len(y)), "positive_frames": int(y.sum())}

    def macro_scenario_ap(self) -> float:
        t = self.scenario_table()
        if t.empty or "ap" not in t:
            return float("nan")
        return float(t["ap"].dropna().mean())

    def empty_frame_report(self, threshold_logit: float) -> dict:
        """False-positive pixel rate on frames with no anomaly, at a fixed threshold."""
        if self.empty.neg.sum() == 0:
            return {"empty_fpr_pixels": float("nan"), "empty_fp_pixels": 0, "empty_pixels": 0}
        m = self.empty.at_threshold(threshold_logit)
        return {"empty_fpr_pixels": m["fpr"], "empty_fp_pixels": m["fp_pixels"],
                "empty_pixels": int(self.empty.neg.sum())}


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average precision with tied scores grouped at one threshold (matches scikit-learn).

    Equal scores are accepted or rejected together, so the result does not depend on the order
    in which tied items happen to arrive (bf16 inference produces many exact ties).
    """
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    positives = labels.sum()
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    last_of_group = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    tp = np.cumsum(labels)[last_of_group]
    precision = tp / (last_of_group + 1)
    recall = tp / positives
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    from scipy.stats import rankdata
    scores, labels = np.asarray(scores, float), np.asarray(labels).astype(bool)
    P, N = labels.sum(), (~labels).sum()
    if P == 0 or N == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - P * (P + 1) / 2) / (P * N))


def summarize(acc: EvalAccumulator, threshold_logit: float | None = None) -> dict:
    """Flat metric dictionary for one split; threshold-dependent entries need ``threshold_logit``."""
    s = acc.pooled.summary()
    out = {"pixel_ap": s["ap"], "pixel_auroc": s["auroc"], "pixel_best_dice": s["best_dice"],
           "pixel_best_thr_logit": s["best_thr_logit"], "macro_scenario_ap": acc.macro_scenario_ap()}
    out.update(acc.frame_metrics())
    if threshold_logit is not None:
        out["threshold_logit"] = threshold_logit
        out.update({f"at_thr_{k}": v for k, v in acc.pooled.at_threshold(threshold_logit).items()
                    if k != "threshold_logit"})
        out.update(acc.empty_frame_report(threshold_logit))
    return out
