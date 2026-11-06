"""Streaming, dataset-level pixel metrics for binary anomaly segmentation.

AUROC and average precision (AUPRC) are approximated using a fixed-width
probability histogram to avoid storing every pixel of a large LOFO fold.
Thresholded confusion counts (and derived metrics) are exact.
"""
from __future__ import annotations

import numpy as np
import torch


class SegmentationMetrics:
    """Accumulate pixel metrics over batches, including frames without any anomaly.

    Args:
        threshold: Pixels with sigmoid(logit) >= threshold are predicted positive.
        num_bins: Number of equally spaced probability bins for AUROC/AUPRC.
        from_logits: If true, inputs are raw model logits; otherwise probabilities.

    ``compute`` returns NaN for AUROC/AUPRC when ground truth has one class.
    Dice and IoU use zero_division=1 when both masks and predictions are empty;
    precision and recall use zero_division=0 when their denominators are zero.
    """

    def __init__(self, threshold: float = 0.5, num_bins: int = 4096,
                 from_logits: bool = True) -> None:
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1")
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins < 2:
            raise ValueError("num_bins must be an integer >= 2")
        self.threshold = float(threshold)
        self.num_bins = num_bins
        self.from_logits = bool(from_logits)
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.tn = 0
        self.positive_hist = np.zeros(self.num_bins, dtype=np.int64)
        self.negative_hist = np.zeros(self.num_bins, dtype=np.int64)

    @torch.no_grad()
    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if predictions.shape != targets.shape or predictions.ndim != 4 or predictions.shape[1] != 1:
            raise ValueError("predictions and targets must have matching [B, 1, H, W] shapes")
        if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
            raise ValueError("predictions and targets must be finite")
        if not torch.all((targets == 0) | (targets == 1)):
            raise ValueError("targets must contain only 0 and 1")
        if not self.from_logits and not torch.all((predictions >= 0) & (predictions <= 1)):
            raise ValueError("probabilities must be in [0, 1]")

        probabilities = predictions.sigmoid() if self.from_logits else predictions
        probs = probabilities.detach().reshape(-1).to("cpu", dtype=torch.float64).numpy()
        truth = targets.detach().reshape(-1).to("cpu", dtype=torch.uint8).numpy().astype(bool)
        positive = probs >= self.threshold
        self.tp += int(np.count_nonzero(positive & truth))
        self.fp += int(np.count_nonzero(positive & ~truth))
        self.fn += int(np.count_nonzero(~positive & truth))
        self.tn += int(np.count_nonzero(~positive & ~truth))

        # The last bin includes p=1.0; other bins are [left, right).
        indices = np.minimum((probs * self.num_bins).astype(np.int64), self.num_bins - 1)
        self.positive_hist += np.bincount(indices[truth], minlength=self.num_bins)
        self.negative_hist += np.bincount(indices[~truth], minlength=self.num_bins)

    def compute(self) -> dict[str, float | int]:
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        total = tp + fp + fn + tn
        if total == 0:
            raise ValueError("No pixels accumulated; call update first")
        dice_denom = 2 * tp + fp + fn
        iou_denom = tp + fp + fn
        dice = (2 * tp / dice_denom) if dice_denom else 1.0
        iou = (tp / iou_denom) if iou_denom else 1.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0

        n_pos = int(self.positive_hist.sum())
        n_neg = int(self.negative_hist.sum())
        auroc = float("nan")
        auprc = float("nan")
        if n_pos and n_neg:
            # Sweep bins from high to low predicted probability.
            cum_tp = np.cumsum(self.positive_hist[::-1], dtype=np.int64)
            cum_fp = np.cumsum(self.negative_hist[::-1], dtype=np.int64)
            tpr = np.r_[0.0, cum_tp / n_pos]
            fpr = np.r_[0.0, cum_fp / n_neg]
            auroc = float(np.trapezoid(tpr, fpr))
            # Step-wise precision-recall area = average precision.
            pr_precision = np.divide(cum_tp, cum_tp + cum_fp,
                                     out=np.zeros_like(cum_tp, dtype=float),
                                     where=(cum_tp + cum_fp) != 0)
            recall_steps = np.diff(tpr)
            auprc = float(np.sum(recall_steps * pr_precision))

        return {
            "pixel_auroc": auroc, "pixel_auprc": auprc,
            "dice": float(dice), "iou": float(iou),
            "precision": float(precision), "recall": float(recall),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive_pixels": n_pos, "negative_pixels": n_neg,
            "threshold": self.threshold, "num_bins": self.num_bins,
        }
