import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights

from cbam import CBAM


class ChestXRayModel(nn.Module):
    """DenseNet121-based multi-label chest X-ray classifier.

    Architecture:
        DenseNet121 feature extractor (pretrained ImageNet)
        → optional CBAM attention (Experiment 3+)
        → Global Average Pooling
        → Linear classifier (14 outputs, raw logits)

    No sigmoid is applied inside forward() — sigmoid is applied at evaluation
    time so that BCEWithLogitsLoss and AsymmetricLoss (both expecting raw
    logits) work without modification.

    The ReLU after the feature extractor uses inplace=False because Grad-CAM
    registers backward hooks on denseblock4; inplace operations on a hooked
    output raise RuntimeError during backward.

    Args:
        num_classes: Number of output classes (14 for NIH ChestX-ray14).
        use_cbam:    Insert CBAM between features and classifier.
        pretrained:  Load ImageNet weights for the DenseNet121 backbone.
    """

    def __init__(
        self,
        num_classes: int = 14,
        use_cbam: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep only the feature extractor; drop the original classifier head.
        self.features   = backbone.features        # outputs (B, 1024, 7, 7) for 224×224 input
        self.cbam       = CBAM(1024) if use_cbam else None
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats  = self.features(x)                           # (B, 1024, 7, 7)
        feats  = F.relu(feats, inplace=False)               # inplace=False — required for Grad-CAM
        if self.cbam is not None:
            feats = self.cbam(feats)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1))       # (B, 1024, 1, 1)
        flat   = torch.flatten(pooled, 1)                   # (B, 1024)
        return self.classifier(flat)                        # (B, 14)  raw logits


def build_model(cfg) -> ChestXRayModel:
    """Build a ChestXRayModel from config module settings.

    Args:
        cfg: The config module (import config; build_model(config)).

    Returns:
        ChestXRayModel instance (not yet moved to device).
    """
    return ChestXRayModel(
        num_classes=cfg.NUM_CLASSES,
        use_cbam=cfg.USE_CBAM,
        pretrained=True,
    )
