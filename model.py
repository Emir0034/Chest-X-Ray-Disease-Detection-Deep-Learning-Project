import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import DenseNet121_Weights

from cbam import CBAM
from faar import FAAR


# DenseNet121-based multi-label chest X-ray classifier
class ChestXRayModel(nn.Module):

    def __init__(
        self,
        num_classes: int = 14,
        use_cbam: bool = False,
        cbam_placement: str = "block4",
        pretrained: bool = True,
        use_faar: bool = False,
        faar_freq_weights=None,
        faar_alpha_init: float = 0.0,
    ):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
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

        self.faar = (
            FAAR(
                num_classes=num_classes,
                num_channels=1024,
                freq_weights=faar_freq_weights,
                alpha_init=faar_alpha_init,
            )
            if use_faar
            else None
        )

        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.features

        # Initial stem: conv0 -> norm0 -> relu0 -> pool0
        x = f.conv0(x)
        x = f.norm0(x)
        x = f.relu0(x)
        x = f.pool0(x)

        # Dense block 1 -> optional CBAM -> transition 1
        x = f.denseblock1(x)                        # (B, 256, 56, 56)
        if self.cbam_block1 is not None:
            x = self.cbam_block1(x)
        x = f.transition1(x)                        # (B, 128, 28, 28)

        # Dense block 2 -> optional CBAM -> transition 2
        x = f.denseblock2(x)                        # (B, 512, 28, 28)
        if self.cbam_block2 is not None:
            x = self.cbam_block2(x)
        x = f.transition2(x)                        # (B, 256, 14, 14)

        # Dense block 3 -> optional CBAM -> transition 3
        x = f.denseblock3(x)                        # (B, 1024, 14, 14)
        if self.cbam_block3 is not None:
            x = self.cbam_block3(x)
        x = f.transition3(x)                        # (B, 512, 7, 7)

        # Dense block 4 -> norm5 -> ReLU -> optional CBAM -> GAP
        x = f.denseblock4(x)                        # (B, 1024, 7, 7)
        x = f.norm5(x)
        x = F.relu(x, inplace=False)               # inplace=False — required for Grad-CAM
        if self.cbam_block4 is not None:
            x = self.cbam_block4(x)

        if self.faar is not None:
            x = self.faar(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))       # (B, 1024, 1, 1)
        x = torch.flatten(x, 1)                    # (B, 1024)
        return self.classifier(x)                  # (B, 14)  raw logits


def build_model(cfg, faar_freq_weights=None) -> ChestXRayModel:

    return ChestXRayModel(
        num_classes=cfg.NUM_CLASSES,
        use_cbam=cfg.USE_CBAM,
        cbam_placement=cfg.CBAM_PLACEMENT,
        pretrained=True,
        use_faar=getattr(cfg, "USE_FAAR", False),
        faar_freq_weights=faar_freq_weights,
        faar_alpha_init=getattr(cfg, "FAAR_ALPHA_INIT", 0.0),
    )
