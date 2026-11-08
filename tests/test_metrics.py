"""Tests for logit-space evaluation metrics."""
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from roadguard.evaluation.metrics import EvalAccumulator, LogitCurve, summarize


def test_logit_curve_matches_sklearn():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.02)
    z = rng.normal(size=y.size) * 2 - 3 + y * 2.0
    c = LogitCurve(); c.update(torch.tensor(z), torch.tensor(y.astype(float)))
    assert abs(c.ap() - average_precision_score(y, z)) < 5e-3
    assert abs(c.auroc() - roc_auc_score(y, z)) < 5e-3

def test_threshold_metrics_consistent():
    z = torch.cat([torch.full((10,), 3.0), torch.full((990,), -3.0)])
    y = torch.cat([torch.ones(10), torch.zeros(990)])
    c = LogitCurve(); c.update(z, y)
    m = c.at_threshold(0.0)
    assert m["dice"] == pytest.approx(1.0) and m["fp_pixels"] == 0

def test_scenario_table_is_within_scenario_and_empty_frames_are_pooled():
    acc = EvalAccumulator()
    good = torch.full((1, 1, 8, 8), -5.0); good[..., :2, :2] = 5.0
    gt = torch.zeros(1, 1, 8, 8); gt[..., :2, :2] = 1
    acc.update(good, gt, ["a"])
    acc.update(torch.full((1, 1, 8, 8), -5.0), torch.zeros(1, 1, 8, 8), ["a"])      # empty frame of a
    acc.update(torch.full((1, 1, 8, 8), 4.0), torch.zeros(1, 1, 8, 8), ["b"])       # empty frame, false alarm
    t = acc.scenario_table().set_index("scenario_id")
    assert t.loc["a", "ap"] > 0.99                       # within-scenario: the empty frame costs nothing here
    assert t.loc["b", "gt_pixels"] == 0 and pd.isna(t.loc["b", "ap"])   # no positives: no AP
    rep = acc.empty_frame_report(0.0)
    assert rep["empty_pixels"] == 128 and rep["empty_fp_pixels"] == 64 and rep["empty_fpr_pixels"] == pytest.approx(0.5)

def test_quantile_and_scenario_table_columns():
    acc = EvalAccumulator()
    z = torch.full((1, 1, 8, 8), -9.0); z[..., :2, :2] = -6.0
    y = torch.zeros(1, 1, 8, 8); y[..., :2, :2] = 1
    acc.update(z, y, ["a"])
    t = acc.scenario_table()
    assert t.loc[0, "separation"] == pytest.approx(3.0, abs=0.05)        # signal visible even at p~1e-3
    assert t.loc[0, "ap"] > 0.99



def test_logit_bins_do_not_collapse_for_tiny_probabilities():
    """Scores far below p=1e-4 (a probability histogram would put them all in one bin) still rank."""
    rng = np.random.default_rng(1)
    y = (rng.random(100_000) < 0.05)
    z = np.where(y, -11.0, -13.0) + rng.normal(size=y.size) * 0.3
    t = torch.tensor(y.astype(np.float32))[None, None, :, None]
    curve = LogitCurve(); curve.update(torch.tensor(z.astype(np.float32))[None, None, :, None], t)
    assert curve.auroc() > 0.99


def test_summarize_reports_threshold_free_and_thresholded_keys():
    acc = EvalAccumulator()
    z = torch.full((1, 1, 8, 8), -5.0); z[..., :2, :2] = 5.0
    y = torch.zeros(1, 1, 8, 8); y[..., :2, :2] = 1
    acc.update(z, y, ["a"])
    out = summarize(acc, threshold_logit=0.0)
    assert out["macro_scenario_ap"] > 0.99 and out["pixel_ap"] > 0.99
    assert out["at_thr_dice"] == pytest.approx(1.0) and "empty_fpr_pixels" in out and "frame_auroc" in out
    assert "at_thr_dice" not in summarize(acc)


def test_average_precision_groups_tied_scores():
    from roadguard.evaluation.metrics import average_precision
    tied = np.zeros(2)
    assert average_precision(tied, np.array([1, 0])) == pytest.approx(0.5)
    assert average_precision(tied, np.array([0, 1])) == pytest.approx(0.5)   # order must not matter
    rng = np.random.default_rng(3)
    y = rng.random(5000) < 0.1
    z = np.round(rng.normal(size=5000) + 1.5 * y, 1)                          # many exact ties
    assert average_precision(z, y) == pytest.approx(average_precision_score(y, z), abs=1e-9)
    assert np.isnan(average_precision(z, np.zeros(5000)))
