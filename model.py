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
        → optional CBAM attention at one or more dense blocks
        → Global Average Pooling
        → Linear classifier (14 outputs, raw logits)

    CBAM placement is controlled by cbam_placement:
        "block4"    — after denseblock4/norm5/ReLU only (default)
        "block34"   — after denseblock3 (before transition3) + block4
        "block1234" — after each of the four dense blocks

    No sigmoid is applied inside forward() — sigmoid is applied at evaluation
    time so that BCEWithLogitsLoss and AsymmetricLoss (both expecting raw
    logits) work without modification.

    The ReLU after norm5 uses inplace=False because Grad-CAM registers
    backward hooks on denseblock4; inplace operations on a hooked output
    raise RuntimeError during backward.

    Args:
        num_classes:    Number of output classes (14 for NIH ChestX-ray14).
        use_cbam:       Insert CBAM modules (placement controlled by cbam_placement).
        cbam_placement: One of "block4", "block34", "block1234". Ignored when use_cbam=False.
        pretrained:     Load ImageNet weights for the DenseNet121 backbone.
    """

    def __init__(
        self,
        num_classes: int = 14,
        use_cbam: bool = False,
        cbam_placement: str = "block4",
        pretrained: bool = True,
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)

        # Keep the full feature extractor as one attribute so that
        # model.features.denseblock4 remains accessible for Grad-CAM hooks
        # (gradcam.py registers hooks on this attribute — do not remove it).
        self.features = backbone.features  # outputs (B, 1024, 7, 7) for 224×224 input

        # CBAM modules — separate instances per block (channel sizes differ).
        # block1=256ch, block2=512ch, block3=1024ch, block4/norm5=1024ch.
        # None when not active; skipped cleanly in forward().
        b1 = use_cbam and cbam_placement == "block1234"
        b2 = use_cbam and cbam_placement == "block1234"
        b3 = use_cbam and cbam_placement in ("block34", "block1234")
        b4 = use_cbam   # always active when use_cbam=True

        self.cbam_block1 = CBAM(256)  if b1 else None
        self.cbam_block2 = CBAM(512)  if b2 else None
        self.cbam_block3 = CBAM(1024) if b3 else None
        self.cbam_block4 = CBAM(1024) if b4 else None

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features

        # Initial stem: conv0 → norm0 → relu0 → pool0
        x = f.conv0(x)
        x = f.norm0(x)
        x = f.relu0(x)
        x = f.pool0(x)

        # Dense block 1 → optional CBAM → transition 1
        x = f.denseblock1(x)                        # (B, 256, 56, 56)
        if self.cbam_block1 is not None:
            x = self.cbam_block1(x)
        x = f.transition1(x)                        # (B, 128, 28, 28)

        # Dense block 2 → optional CBAM → transition 2
        x = f.denseblock2(x)                        # (B, 512, 28, 28)
        if self.cbam_block2 is not None:
            x = self.cbam_block2(x)
        x = f.transition2(x)                        # (B, 256, 14, 14)

        # Dense block 3 → optional CBAM → transition 3
        x = f.denseblock3(x)                        # (B, 1024, 14, 14)
        if self.cbam_block3 is not None:
            x = self.cbam_block3(x)
        x = f.transition3(x)                        # (B, 512, 7, 7)

        # Dense block 4 → norm5 → ReLU → optional CBAM → GAP
        x = f.denseblock4(x)                        # (B, 1024, 7, 7)
        x = f.norm5(x)
        x = F.relu(x, inplace=False)               # inplace=False — required for Grad-CAM
        if self.cbam_block4 is not None:
            x = self.cbam_block4(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))       # (B, 1024, 1, 1)
        x = torch.flatten(x, 1)                    # (B, 1024)
        return self.classifier(x)                  # (B, 14)  raw logits


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
        cbam_placement=cfg.CBAM_PLACEMENT,
        pretrained=True,
    )
