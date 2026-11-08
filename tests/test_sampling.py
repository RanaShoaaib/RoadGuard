"""Tests for the scenario-balanced sampler."""
import pandas as pd
import pytest

from roadguard.training.sampling import make_scenario_balanced_sampler, scenario_weights


def _frame_df():
    rows = []
    for s, px in [("g1", 50000), ("g2", 40000), ("g3", 30000), ("t1", 10), ("t2", 12)]:
        for i in range(100):
            rows.append((s, px if i < 60 else 0))
    return pd.DataFrame(rows, columns=["scenario_id", "anomaly_pixels"])

def test_sampler_balances_scenarios_and_enriches_positives():
    df = _frame_df()
    w = scenario_weights(df, pos_share_within=0.7)
    assert w.sum() == pytest.approx(1.0)
    by = pd.Series(w).groupby(df.scenario_id).sum()
    assert by.round(6).nunique() == 1                                        # equal per scenario
    pos_share = pd.Series(w)[df.anomaly_pixels > 0].groupby(df.scenario_id[df.anomaly_pixels > 0]).sum()
    assert (pos_share / by[pos_share.index]).round(3).eq(0.7).all()
    idx = list(make_scenario_balanced_sampler(df, seed=0, num_samples=40000))
    drawn = pd.Series(df.scenario_id.to_numpy()[idx]).value_counts(normalize=True)
    assert abs(drawn["g1"] - drawn["t1"]) < 0.02                             # giants no longer dominate

def test_all_empty_or_all_positive_scenarios_keep_their_share():
    df = _frame_df()
    extra = pd.concat([pd.DataFrame({"scenario_id": "empty", "anomaly_pixels": [0] * 50}),
                       pd.DataFrame({"scenario_id": "full", "anomaly_pixels": [900] * 50})])
    df2 = pd.concat([df, extra]).reset_index(drop=True)
    by = pd.Series(scenario_weights(df2)).groupby(df2.scenario_id).sum()
    assert by["empty"] > 0 and by["full"] > 0 and by.round(6).nunique() == 1


def test_invalid_inputs():
    with pytest.raises(ValueError, match="pos_share_within"):
        scenario_weights(_frame_df(), pos_share_within=1.0)
    with pytest.raises(ValueError, match="empty"):
        scenario_weights(_frame_df().iloc[:0])
