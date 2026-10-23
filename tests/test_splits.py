
import pandas as pd
import pytest

from roadguard.data.splits import create_standard_split, create_lofo_splits


@pytest.fixture
def normality_manifest():
    records = []

    for scenario_id in range(20):
        for frame_id in range(5):
            records.append({
                "source": "normality",
                "scenario_id": str(scenario_id),
                "frame_id": str(frame_id),
                "anomaly_family": "normality",
            })

    return pd.DataFrame(records)


def test_create_standard_split(normality_manifest):
    result = create_standard_split(normality_manifest)

    assert len(result) == 100
    assert set(result["split"]) == {"train", "val", "test"}

    # Each scenario belongs to exactly one split.
    assert (
        result.groupby(["source", "scenario_id"])["split"]
        .nunique()
        .eq(1)
        .all()
    )

    scenario_counts = (
        result[["scenario_id", "split"]]
        .drop_duplicates()["split"]
        .value_counts()
    )

    assert scenario_counts["train"] == 14
    assert scenario_counts["val"] == 3
    assert scenario_counts["test"] == 3


def test_create_standard_split_reproducible(normality_manifest):
    first = create_standard_split(normality_manifest, seed=42)
    second = create_standard_split(normality_manifest, seed=42)

    pd.testing.assert_frame_equal(first, second)


def test_create_standard_split_invalid_ratios(normality_manifest):
    with pytest.raises(ValueError, match="sum to 1"):
        create_standard_split(
            normality_manifest,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.2,
        )


def test_create_standard_split_too_few_scenarios():
    manifest = pd.DataFrame([
        {"source": "normality", "scenario_id": "1"},
        {"source": "normality", "scenario_id": "2"},
    ])

    with pytest.raises(ValueError, match="At least 3 scenarios"):
        create_standard_split(manifest)



@pytest.fixture
def static_manifest():
    records = []

    for family in ("animal", "vehicle", "object"):
        for scenario_id in range(4):
            for frame_id in range(5):
                records.append({
                    "source": "static",
                    "scenario_id": f"{family}_{scenario_id}",
                    "frame_id": str(frame_id),
                    "anomaly_family": family,
                })

    return pd.DataFrame(records)


def test_create_lofo_splits(static_manifest):
    folds = create_lofo_splits(static_manifest)

    assert set(folds) == {"animal", "vehicle", "object"}

    for held_out_family, fold in folds.items():
        assert len(fold) == len(static_manifest)

        # All held-out family scenarios must be in test.
        assert (
            fold.loc[
                fold["anomaly_family"] == held_out_family,
                "split",
            ] == "test"
        ).all()

        # No other family should appear in test.
        assert (
            fold.loc[
                fold["split"] == "test",
                "anomaly_family",
            ] == held_out_family
        ).all()

        # No scenario may cross partitions.
        assert (
            fold.groupby("scenario_id")["split"]
            .nunique()
            .eq(1)
            .all()
        )

        assert set(fold["split"]) == {"train", "val", "test"}


def test_create_lofo_splits_reproducible(static_manifest):
    first = create_lofo_splits(static_manifest, seed=42)
    second = create_lofo_splits(static_manifest, seed=42)

    for family in first:
        pd.testing.assert_frame_equal(first[family], second[family])


def test_create_lofo_splits_rejects_normality(static_manifest):
    invalid = static_manifest.copy()
    invalid.loc[0, "source"] = "normality"

    with pytest.raises(ValueError, match="static anomaly scenarios only"):
        create_lofo_splits(invalid)


def test_create_lofo_splits_rejects_single_family(static_manifest):
    single_family = static_manifest[
        static_manifest["anomaly_family"] == "animal"
    ]

    with pytest.raises(ValueError, match="at least two anomaly families"):
        create_lofo_splits(single_family)
