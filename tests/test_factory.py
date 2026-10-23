"""Unit tests for the RoadGuard model factory and U-Net."""

import pytest
import torch

from roadguard.models.factory import UNet, build_model


@pytest.mark.parametrize("channels", [3, 4])
def test_unet_output_shape_and_finite_logits(channels):
    model = build_model("unet", in_channels=channels, base_channels=4)
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, channels, 64, 80))
    assert logits.shape == (2, 1, 64, 80)
    assert torch.isfinite(logits).all()


def test_odd_image_dimensions_preserved():
    model = build_model("UNET", base_channels=4).eval()
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 65, 79))
    assert logits.shape == (1, 1, 65, 79)


def test_backward_pass():
    model = build_model("unet", in_channels=4, base_channels=4)
    logits = model(torch.randn(2, 4, 32, 32))
    logits.square().mean().backward()
    assert model.enc1[0].weight.grad is not None
    assert torch.isfinite(model.enc1[0].weight.grad).all()


def test_raw_logits_not_sigmoid_probabilities():
    model = build_model("unet", base_channels=4)
    assert isinstance(model, UNet)
    assert isinstance(model.head, torch.nn.Conv2d)
    assert not any(isinstance(module, torch.nn.Sigmoid) for module in model.modules())


@pytest.mark.parametrize("kwargs,match", [
    ({"name": "unknown"}, "Unknown model"),
    ({"name": "unet", "in_channels": 1}, "in_channels"),
    ({"name": "unet", "num_classes": 2}, "num_classes"),
    ({"name": "unet", "base_channels": 0}, "base_channels"),
])
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        build_model(**kwargs)


def test_too_small_image_rejected():
    model = build_model("unet", base_channels=4)
    with pytest.raises(ValueError, match="at least 16"):
        model(torch.randn(1, 3, 15, 32))


@pytest.mark.parametrize("channels", [3, 4])
def test_segformer_shape_and_backward(channels):
    pytest.importorskip("transformers")
    model = build_model("segformer", in_channels=channels, pretrained=False)
    model.eval()
    x = torch.randn(1, channels, 64, 80)
    logits = model(x)
    assert logits.shape == (1, 1, 64, 80)
    assert torch.isfinite(logits).all()
    logits.mean().backward()
    proj = model.network.segformer.stages[0].patch_embeddings.proj
    assert proj.weight.grad is not None
    assert torch.isfinite(proj.weight.grad).all()


def test_segformer_four_channel_projection():
    pytest.importorskip("transformers")
    model = build_model("segformer", in_channels=4, pretrained=False)
    projection = model.network.segformer.stages[0].patch_embeddings.proj
    assert projection.in_channels == 4
    assert torch.allclose(projection.weight[:, 3], projection.weight[:, :3].mean(dim=1))


@pytest.mark.parametrize("kwargs", [
    {"in_channels": 1}, {"num_classes": 2},
])
def test_segformer_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        build_model("segformer", pretrained=False, **kwargs)
