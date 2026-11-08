"""Read finished runs back and aggregate them across seeds and folds."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

METRICS = ["macro_ap", "pixel_ap", "pixel_auroc", "frame_auroc", "dice", "precision", "recall", "empty_fpr"]
_TEST_KEYS = {"macro_ap": "macro_scenario_ap", "pixel_ap": "pixel_ap", "pixel_auroc": "pixel_auroc",
              "frame_auroc": "frame_auroc", "dice": "at_thr_dice", "precision": "at_thr_precision",
              "recall": "at_thr_recall", "empty_fpr": "empty_fpr_pixels"}


def _run_dir(root, name, family, seed) -> Path:
    return Path(root) / name / family / f"seed_{seed}"


def load_reports(output_root, names, families, seeds, roles=("baseline",)) -> pd.DataFrame:
    """One row per (run, role, family, seed, which); which = 'selected' (smoothed-val pick) or 'last'."""
    families = [families] if isinstance(families, str) else list(families)
    rows = []
    for name in names:
        for family in families:
            for seed in seeds:
                path = _run_dir(output_root, name, family, seed) / "report.json"
                if not path.exists():
                    continue
                report = json.loads(path.read_text())
                for role in roles:
                    for which, result in report.get(role, {}).items():
                        row = {"run": name, "role": role, "family": family, "seed": seed,
                               "which": which, "epoch": result["epoch"]}
                        row.update({m: result["test"][k] for m, k in _TEST_KEYS.items()})
                        rows.append(row)
    return pd.DataFrame(rows)


def _mean_std(group, count_name: str) -> pd.DataFrame:
    out = group[METRICS].mean().add_suffix("_mean").join(group[METRICS].std(ddof=1).add_suffix("_std"))
    out = out.join(group.size().rename(count_name))
    return out[sorted(out.columns, key=lambda c: (c.rsplit("_", 1)[0] if c != count_name else "~", c))]


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and std across seeds for each (run, role, family, which)."""
    if df.empty:
        return df
    return _mean_std(df.groupby(["run", "role", "family", "which"]), "n_seeds")


def fold_average(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: average seeds within each fold first, then mean/std ACROSS FOLDS per (run, role, which)."""
    if df.empty:
        return df
    per_fold = df.groupby(["run", "role", "which", "family"])[METRICS].mean().reset_index()
    return _mean_std(per_fold.groupby(["run", "role", "which"]), "n_folds")


def per_scenario_ap(output_root, names, family, seeds, role="baseline", which="last") -> pd.DataFrame:
    """Scenario x run table of test AP averaged over seeds (one fold, one role)."""
    suffix = "_last" if which == "last" else ""
    columns = {}
    for name in names:
        frames = []
        for seed in seeds:
            path = _run_dir(output_root, name, family, seed) / role / f"test_scenarios{suffix}.csv"
            if path.exists():
                frames.append(pd.read_csv(path, dtype={"scenario_id": str}).set_index("scenario_id")["ap"])
        if frames:
            columns[name] = pd.concat(frames, axis=1).mean(axis=1)
    out = pd.DataFrame(columns)
    out.index = [str(i)[:8] for i in out.index]
    return out
