import torch
import torch.nn as nn


class FAAR(nn.Module):
    """Frequency-Aware Attention Refinement.

    Modulates the 1024-channel feature map produced by DenseNet121 using
    inverse class-frequency weights. A linear projection maps the frequency
    vector (one value per disease class) to per-channel scale factors, then
    applies a residual scaling controlled by a learnable scalar alpha.

    When alpha=0.0 (the default initialisation), the module is an exact
    identity — the first training step starts from the same point as a model
    without FAAR, and alpha is learned from there.

    Args:
        num_classes:   Number of disease classes (must match len(freq_weights)).
        num_channels:  Number of feature-map channels to modulate (1024 for DenseNet121).
        freq_weights:  1-D tensor of length num_classes containing pre-computed
                       inverse-frequency weights for each class. Registered as a
                       non-trainable buffer.
        alpha_init:    Initial value for the learnable residual scale (default 0.0).

    Raises:
        ValueError: If freq_weights is None or its length != num_classes.
    """

    def __init__(
        self,
        num_classes: int = 14,
        num_channels: int = 1024,
        freq_weights: torch.Tensor = None,
        alpha_init: float = 0.0,
    ):
        super().__init__()

        if freq_weights is None:
            raise ValueError(
                "FAAR requires freq_weights (inverse-frequency tensor of length "
                f"{num_classes}). Got None."
            )
        if len(freq_weights) != num_classes:
            raise ValueError(
                f"FAAR freq_weights must have length {num_classes}, "
                f"got {len(freq_weights)}."
            )

        self.register_buffer("freq_weights", freq_weights.float())
        self.fc    = nn.Linear(num_classes, num_channels)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply frequency-aware residual scaling.

        Args:
            x: Feature map, shape [B, C, H, W].

        Returns:
            Scaled feature map of the same shape.
        """
        scale = torch.sigmoid(self.fc(self.freq_weights))   # [num_channels]
        scale = scale.view(1, -1, 1, 1)                     # [1, C, 1, 1]
        return x * (1.0 + self.alpha * scale)
