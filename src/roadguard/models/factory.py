"""Model factory for RoadGuard binary anomaly segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Sequential):
    """Two convolution–normalization–activation blocks."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """Four-level U-Net returning unactivated per-pixel logits."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 base_channels: int = 32) -> None:
        super().__init__()
        if in_channels not in (3, 4):
            raise ValueError("in_channels must be 3 (RGB) or 4 (RGB + depth)")
        if num_classes != 1:
            raise ValueError("RoadGuard currently supports binary segmentation (num_classes=1)")
        if not isinstance(base_channels, int) or isinstance(base_channels, bool) or base_channels < 1:
            raise ValueError("base_channels must be a positive integer")

        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.bottleneck = DoubleConv(c * 8, c * 16)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
        self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
        self.dec1 = DoubleConv(c * 2 + c, c)
        self.head = nn.Conv2d(c, num_classes, kernel_size=1)

    @staticmethod
    def _upsample_and_concat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((skip, x), dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor of shape [B, C, H, W]")
        if min(x.shape[-2:]) < 16:
            raise ValueError("U-Net requires image height and width of at least 16")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        z = self.bottleneck(self.pool(e4))
        z = self.dec4(self._upsample_and_concat(z, e4))
        z = self.dec3(self._upsample_and_concat(z, e3))
        z = self.dec2(self._upsample_and_concat(z, e2))
        z = self.dec1(self._upsample_and_concat(z, e1))
        return self.head(z)


class SegFormer(nn.Module):
    """SegFormer-B0 with logits upsampled to the input resolution."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1,
                 pretrained: bool = True) -> None:
        super().__init__()
        if in_channels not in (3, 4):
            raise ValueError("in_channels must be 3 (RGB) or 4 (RGB + depth)")
        if num_classes != 1:
            raise ValueError("RoadGuard currently supports binary segmentation (num_classes=1)")
        try:
            from transformers import SegformerConfig, SegformerForSemanticSegmentation
        except ImportError as exc:
            raise ImportError("SegFormer requires transformers: pip install transformers") from exc

        if pretrained:
            # The segmentation classifier is newly initialized for one output channel.
            self.network = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0", num_labels=1, ignore_mismatched_sizes=True
            )
        else:
            self.network = SegformerForSemanticSegmentation(
                SegformerConfig(num_labels=1)
            )

        if in_channels == 4:
            # Preserve pretrained RGB filters; initialize depth from their mean.
            projection = self.network.segformer.stages[0].patch_embeddings.proj
            expanded = nn.Conv2d(
                4, projection.out_channels, kernel_size=projection.kernel_size,
                stride=projection.stride, padding=projection.padding,
                dilation=projection.dilation, bias=projection.bias is not None,
            )
            with torch.no_grad():
                expanded.weight[:, :3].copy_(projection.weight)
                expanded.weight[:, 3:4].copy_(projection.weight.mean(dim=1, keepdim=True))
                if projection.bias is not None:
                    expanded.bias.copy_(projection.bias)
            self.network.segformer.stages[0].patch_embeddings.proj = expanded
            self.network.config.num_channels = 4
            self.network.segformer.config.num_channels = 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected input tensor of shape [B, C, H, W]")
        logits = self.network(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)


def build_model(name: str, in_channels: int = 3, num_classes: int = 1,
                base_channels: int = 32, pretrained: bool = True) -> nn.Module:
    """Build U-Net or SegFormer-B0; base_channels applies only to U-Net."""
    if not isinstance(name, str):
        raise ValueError("Model name must be a string")
    if name.lower() == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes,
                    base_channels=base_channels)
    if name.lower() == "segformer":
        return SegFormer(in_channels=in_channels, num_classes=num_classes,
                         pretrained=pretrained)
    raise ValueError(f"Unknown model '{name}'. Available models: unet, segformer")
