"""Tests for streaming RoadGuard binary segmentation metrics."""
import math

import pytest
import torch

from roadguard.evaluation.metrics import SegmentationMetrics


def tensors(scores, labels):
    p = torch.tensor(scores, dtype=torch.float32).reshape(1, 1, 1, -1)
    y = torch.tensor(labels, dtype=torch.float32).reshape(1, 1, 1, -1)
    return p, y


def test_perfect_predictions():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.1, 0.9, 0.2, 0.8], [0, 1, 0, 1]))
    result = metric.compute()
    for key in ("dice", "iou", "precision", "recall", "pixel_auroc", "pixel_auprc"):
        assert result[key] == pytest.approx(1.0)


def test_confusion_counts_and_overlap():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.9, 0.8, 0.7, 0.1], [1, 0, 1, 0]))
    result = metric.compute()
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 0, 1)
    assert result["dice"] == pytest.approx(0.8)
    assert result["iou"] == pytest.approx(2 / 3)
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == pytest.approx(1)


def test_empty_frame_false_positives_contribute_to_global_metrics():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.9, 0.1], [1, 0]))
    metric.update(*tensors([0.8, 0.2], [0, 0]))
    result = metric.compute()
    assert result["fp"] == 1
    assert result["dice"] == pytest.approx(2 / 3)
    assert result["positive_pixels"] == 1
    assert result["negative_pixels"] == 3


def test_all_background_with_no_predictions():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.1, 0.2], [0, 0]))
    result = metric.compute()
    assert result["dice"] == result["iou"] == 1.0
    assert result["precision"] == result["recall"] == 0.0
    assert math.isnan(result["pixel_auroc"])
    assert math.isnan(result["pixel_auprc"])


def test_all_positive_ground_truth_undefined_curves():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.9, 0.2], [1, 1]))
    assert math.isnan(metric.compute()["pixel_auroc"])
    assert math.isnan(metric.compute()["pixel_auprc"])


def test_threshold_is_inclusive():
    metric = SegmentationMetrics(from_logits=False, threshold=0.5)
    metric.update(*tensors([0.5, 0.49], [1, 0]))
    assert metric.compute()["tp"] == 1
    assert metric.compute()["tn"] == 1


def test_logits_are_sigmoided():
    metric = SegmentationMetrics(from_logits=True)
    metric.update(*tensors([-3, 3], [0, 1]))
    assert metric.compute()["dice"] == 1.0


def test_updates_equivalent_to_combined_batch():
    separate = SegmentationMetrics(from_logits=False)
    combined = SegmentationMetrics(from_logits=False)
    a = tensors([0.1, 0.9], [0, 1])
    b = tensors([0.7, 0.4], [0, 1])
    separate.update(*a)
    separate.update(*b)
    combined.update(torch.cat([a[0], b[0]]), torch.cat([a[1], b[1]]))
    assert separate.compute() == combined.compute()


def test_reset():
    metric = SegmentationMetrics(from_logits=False)
    metric.update(*tensors([0.1, 0.9], [0, 1]))
    metric.reset()
    with pytest.raises(ValueError, match="No pixels"):
        metric.compute()


@pytest.mark.parametrize("scores,labels", [([float("nan")], [0]), ([0.2], [0.5]), ([1.2], [1])])
def test_reject_invalid_data(scores, labels):
    metric = SegmentationMetrics(from_logits=False)
    with pytest.raises(ValueError):
        metric.update(*tensors(scores, labels))


def test_reject_invalid_shapes():
    metric = SegmentationMetrics()
    with pytest.raises(ValueError, match="matching"):
        metric.update(torch.zeros(1, 1, 2, 2), torch.zeros(1, 1, 2, 3))


def test_reject_invalid_configuration():
    with pytest.raises(ValueError):
        SegmentationMetrics(threshold=1.0)
    with pytest.raises(ValueError):
        SegmentationMetrics(num_bins=1)


def test_histogram_approximation_is_reasonable():
    # All probabilities are far enough apart to occupy separate bins.
    metric = SegmentationMetrics(from_logits=False, num_bins=4096)
    metric.update(*tensors([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]))
    result = metric.compute()
    assert result["pixel_auroc"] == pytest.approx(0.75)
    assert result["pixel_auprc"] == pytest.approx((1.0 + 2 / 3) / 2)
