import math

import numpy as np
import pandas as pd


def create_lofo_splits(
    manifest: pd.DataFrame,
    val_frac: float = 0.2,
    min_family_scenarios: int = 8,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Leave-one-anomaly-family-out splits with object-disjoint validation.

    One fold per family with at least ``min_family_scenarios`` scenarios:
      * test  : every scenario of the held-out family (an unseen family),
      * val   : whole anomaly objects of the other families, about ``val_frac`` of each family's
                scenarios, chosen once per family so the same objects validate in every fold,
      * train : everything else. Families smaller than ``min_family_scenarios`` are too small to
                test on and are always train.
    An anomaly object is never in train and val together, and a scenario is never split.
    """
    if not 0 < val_frac < 1:
        raise ValueError("val_frac must be between 0 and 1")

    if manifest.empty:
        raise ValueError("Manifest is empty")

    required = {"scenario_id", "anomaly_family", "anomaly_object"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Manifest missing columns: {sorted(required - set(manifest.columns))}")

    scenarios = manifest[["scenario_id", "anomaly_family", "anomaly_object"]].drop_duplicates()

    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("A scenario belongs to multiple anomaly families or objects")

    counts = scenarios.groupby("anomaly_family")["scenario_id"].nunique()
    fold_families = sorted(counts[counts >= min_family_scenarios].index)

    if len(fold_families) < 2:
        raise ValueError(
            f"LOFO requires at least two anomaly families with >= {min_family_scenarios} "
            f"scenarios, got {counts.to_dict()}"
        )

    val_objects = {}

    for family in fold_families:
        part = scenarios[scenarios["anomaly_family"] == family]
        scenarios_per_object = part.groupby("anomaly_object")["scenario_id"].nunique()
        objects = sorted(scenarios_per_object.index)

        # A per-family generator keeps a family's validation objects stable when other families change.
        rng = np.random.default_rng([seed, *map(ord, family)])
        rng.shuffle(objects)

        target = math.ceil(val_frac * len(part))
        chosen, covered = set(), 0

        for obj in objects:
            # Always leave at least one object for training.
            if covered >= target or len(chosen) >= len(objects) - 1:
                break
            chosen.add(obj)
            covered += int(scenarios_per_object[obj])

        val_objects[family] = chosen

    folds = {}

    for held_out_family in fold_families:
        scenario_splits = scenarios.copy()
        scenario_splits["split"] = "train"

        for family in fold_families:
            if family == held_out_family:
                continue
            is_val = (
                (scenario_splits["anomaly_family"] == family)
                & scenario_splits["anomaly_object"].isin(val_objects[family])
            )
            scenario_splits.loc[is_val, "split"] = "val"

        scenario_splits.loc[
            scenario_splits["anomaly_family"] == held_out_family, "split"
        ] = "test"

        folds[held_out_family] = manifest.merge(
            scenario_splits[["scenario_id", "split"]],
            on="scenario_id",
            how="left",
            validate="many_to_one",
        )

    return folds


def summarize_folds(folds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per fold: families, scenarios and frames in each split."""
    rows = []

    for held_out_family, fold in folds.items():
        row = {"test_family": held_out_family}

        for split in ("train", "val", "test"):
            part = fold[fold["split"] == split]
            row[f"{split}_families"] = ",".join(sorted(part["anomaly_family"].unique()))
            row[f"{split}_scenarios"] = part["scenario_id"].nunique()
            row[f"{split}_frames"] = len(part)

        rows.append(row)

    return pd.DataFrame(rows)
