import torch
import torch.nn as nn


# Frequency-Aware Attention Refinement
class FAAR(nn.Module):

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

        scale = torch.sigmoid(self.fc(self.freq_weights))   # [num_channels]
        scale = scale.view(1, -1, 1, 1)                     # [1, C, 1, 1]
        return x * (1.0 + self.alpha * scale)
