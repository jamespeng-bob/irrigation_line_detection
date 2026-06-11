"""Multi-class U-Net for irrigation_line_detection.

Thin wrapper around ``segmentation_models_pytorch.Unet`` configured for
``K``-channel multi-label output (one sigmoid per class, NOT softmax). The
loss / metric stack downstream consumes raw logits and is responsible for
applying ``sigmoid`` per channel.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class SMPUnet(nn.Module):
    """U-Net with a configurable encoder + ``K``-channel sigmoid head."""

    def __init__(
        self,
        num_classes: int,
        encoder_name: str = "mit_b2",
        encoder_weights: str | None = "imagenet",
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}")
        self.num_classes = int(num_classes)
        self.encoder_name = str(encoder_name)
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self.num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(B, K, H, W)``."""
        return self.model(x)


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Construct a model from the ``model:`` config block.

    Parameters
    ----------
    cfg
        Sub-dict ``cfg["model"]``: keys ``name``, ``encoder``, ``encoder_weights``.
    num_classes
        ``K``, taken from the dataset / class-remap rather than the model
        config so the two cannot get out of sync.
    """
    name = cfg.get("name", "smp_unet")
    encoder = cfg.get("encoder", "mit_b2")
    weights = cfg.get("encoder_weights", "imagenet")
    if weights in ("null", "none", None, ""):
        weights = None
    if name == "smp_unet":
        return SMPUnet(
            num_classes=num_classes,
            encoder_name=encoder,
            encoder_weights=weights,
        )
    raise ValueError(f"Unknown model name: {name!r}")
