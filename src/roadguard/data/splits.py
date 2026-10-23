
import numpy as np
import pandas as pd


def create_standard_split(
    manifest: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign scenario-disjoint train, val, and test splits."""

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1")

    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must be positive")

    if manifest.empty:
        raise ValueError("Manifest is empty")

    scenarios = manifest[
        ["source", "scenario_id"]
    ].drop_duplicates()

    n = len(scenarios)

    if n < 3:
        raise ValueError("At least 3 scenarios are required")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    # Ensure each split has at least one scenario.
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    n_test = n - n_train - n_val

    if n_test < 1:
        raise ValueError("Not enough scenarios for the requested ratios")

    split_labels = np.empty(n, dtype=object)

    split_labels[indices[:n_train]] = "train"
    split_labels[indices[n_train:n_train + n_val]] = "val"
    split_labels[indices[n_train + n_val:]] = "test"

    scenarios = scenarios.copy()
    scenarios["split"] = split_labels

    result = manifest.merge(
        scenarios,
        on=["source", "scenario_id"],
        how="left",
        validate="many_to_one",
    )

    return result



def create_lofo_splits(
    manifest: pd.DataFrame,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Create leave-one-anomaly-family-out scenario-disjoint splits."""

    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")

    if manifest.empty:
        raise ValueError("Manifest is empty")

    if not (manifest["source"] == "static").all():
        raise ValueError("LOFO requires static anomaly scenarios only")

    scenarios = manifest[
        ["source", "scenario_id", "anomaly_family"]
    ].drop_duplicates()

    if scenarios["scenario_id"].duplicated().any():
        raise ValueError("A scenario belongs to multiple anomaly families")

    families = sorted(scenarios["anomaly_family"].unique())

    if len(families) < 2:
        raise ValueError("LOFO requires at least two anomaly families")

    folds = {}

    for held_out_family in families:
        scenario_splits = scenarios.copy()
        scenario_splits["split"] = "train"

        test_mask = (
            scenario_splits["anomaly_family"] == held_out_family
        )
        scenario_splits.loc[test_mask, "split"] = "test"

        rng = np.random.default_rng(seed)

        for family in families:
            if family == held_out_family:
                continue

            family_indices = scenario_splits.index[
                scenario_splits["anomaly_family"] == family
            ].to_numpy()

            n = len(family_indices)

            if n < 2:
                continue

            n_val = max(1, int(n * val_ratio))
            n_val = min(n_val, n - 1)

            selected = rng.choice(
                family_indices,
                size=n_val,
                replace=False,
            )

            scenario_splits.loc[selected, "split"] = "val"

        fold = manifest.merge(
            scenario_splits[
                ["source", "scenario_id", "split"]
            ],
            on=["source", "scenario_id"],
            how="left",
            validate="many_to_one",
        )

        folds[held_out_family] = fold

    return folds
