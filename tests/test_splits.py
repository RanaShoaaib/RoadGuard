import pandas as pd
import pytest

from roadguard.data.splits import create_lofo_splits, summarize_folds


@pytest.fixture
def static_manifest():
    """animal 4 objects x 3 scenarios, home 5 x 2, special 6 x 2, nature 1 x 1 (too small to test on)."""
    records = []

    for family, n_objects, per_object in [("animal", 4, 3), ("home", 5, 2), ("special", 6, 2), ("nature", 1, 1)]:
        for obj in range(n_objects):
            for scenario in range(per_object):
                for frame_id in range(3):
                    records.append({
                        "scenario_id": f"{family}{obj}_{scenario}",
                        "frame_id": str(frame_id),
                        "anomaly_family": family,
                        "anomaly_object": f"{family}{obj}",
                    })

    return pd.DataFrame(records)


def test_create_lofo_splits(static_manifest):
    folds = create_lofo_splits(static_manifest, val_frac=0.25, min_family_scenarios=6)

    # nature has one scenario: never a test fold
    assert set(folds) == {"animal", "home", "special"}

    for held_out_family, fold in folds.items():
        assert len(fold) == len(static_manifest)

        # All held-out family scenarios are in test, and only those.
        assert (fold.loc[fold["anomaly_family"] == held_out_family, "split"] == "test").all()
        assert (fold.loc[fold["split"] == "test", "anomaly_family"] == held_out_family).all()

        # No scenario crosses partitions.
        assert fold.groupby("scenario_id")["split"].nunique().eq(1).all()
        assert set(fold["split"]) == {"train", "val", "test"}

        # Validation: whole objects, never in train, never from the held-out family.
        objects = fold.groupby("split")["anomaly_object"].agg(set)
        assert not (objects["train"] & objects["val"])
        assert held_out_family not in set(fold.loc[fold["split"] == "val", "anomaly_family"])

        # The family below the size limit is always training data.
        assert set(fold.loc[fold["anomaly_family"] == "nature", "split"]) == {"train"}


def test_validation_objects_are_stable_across_folds(static_manifest):
    folds = create_lofo_splits(static_manifest, val_frac=0.25, min_family_scenarios=6)

    val_in_animal_fold = set(folds["home"].query("split == 'val' and anomaly_family == 'animal'")["anomaly_object"])
    val_in_special_fold = set(folds["special"].query("split == 'val' and anomaly_family == 'animal'")["anomaly_object"])

    assert val_in_animal_fold == val_in_special_fold


def test_create_lofo_splits_reproducible(static_manifest):
    first = create_lofo_splits(static_manifest, min_family_scenarios=6, seed=42)
    second = create_lofo_splits(static_manifest, min_family_scenarios=6, seed=42)

    for family in first:
        pd.testing.assert_frame_equal(first[family], second[family])


def test_create_lofo_splits_seed_changes_validation(static_manifest):
    first = create_lofo_splits(static_manifest, min_family_scenarios=6, seed=1)["special"]
    others = [create_lofo_splits(static_manifest, min_family_scenarios=6, seed=s)["special"] for s in range(2, 8)]

    assert any(not first["split"].equals(other["split"]) for other in others)


def test_create_lofo_splits_leaves_an_object_for_training():
    manifest = pd.DataFrame([
        {"scenario_id": f"{family}{i}", "frame_id": "0", "anomaly_family": family, "anomaly_object": family}
        for family in ("a", "b") for i in range(5)      # one object per family
    ])

    folds = create_lofo_splits(manifest, val_frac=0.5, min_family_scenarios=2)

    assert "val" not in set(folds["a"]["split"])       # a single object cannot be both train and val


def test_create_lofo_splits_requires_two_large_families(static_manifest):
    with pytest.raises(ValueError, match="at least two anomaly families"):
        create_lofo_splits(static_manifest, min_family_scenarios=50)


def test_create_lofo_splits_rejects_invalid_input(static_manifest):
    with pytest.raises(ValueError, match="val_frac"):
        create_lofo_splits(static_manifest, val_frac=1.5)

    with pytest.raises(ValueError, match="Manifest is empty"):
        create_lofo_splits(static_manifest.iloc[0:0])

    with pytest.raises(ValueError, match="missing columns"):
        create_lofo_splits(static_manifest.drop(columns="anomaly_object"))

    broken = static_manifest.copy()
    broken.loc[0, "anomaly_family"] = "home"
    with pytest.raises(ValueError, match="multiple"):
        create_lofo_splits(broken, min_family_scenarios=6)


def test_summarize_folds(static_manifest):
    folds = create_lofo_splits(static_manifest, min_family_scenarios=6)
    summary = summarize_folds(folds)

    assert sorted(summary["test_family"]) == ["animal", "home", "special"]
    row = summary.set_index("test_family").loc["special"]
    assert row["test_scenarios"] == 12
    assert row["test_frames"] == 36
    assert row["train_scenarios"] + row["val_scenarios"] + row["test_scenarios"] == static_manifest["scenario_id"].nunique()
