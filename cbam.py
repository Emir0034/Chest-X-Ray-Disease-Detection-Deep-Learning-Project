import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel Attention Module from CBAM.

    Applies a shared MLP to both average-pooled and max-pooled feature maps,
    then combines the outputs with sigmoid to produce a per-channel weight.

    Args:
        in_channels: Number of input feature channels (1024 for DenseNet121 output).
        reduction:   Bottleneck reduction ratio for the MLP (default 16).
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        bottleneck = max(in_channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        avg = self.avg_pool(x).view(B, C)       # (B, C)
        mx  = self.max_pool(x).view(B, C)       # (B, C)
        att = self.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return att.view(B, C, 1, 1)             # broadcast multiplier


class SpatialAttention(nn.Module):
    """Spatial Attention Module from CBAM.

    Concatenates channel-wise average and max projections, then applies a
    convolution + sigmoid to produce a spatial weight map.

    Args:
        kernel_size: Convolution kernel size (7 recommended by the CBAM paper).
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)            # (B, 1, H, W)
        max_out = x.max(dim=1, keepdim=True).values      # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(combined))          # (B, 1, H, W)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).

    Sequentially applies Channel Attention then Spatial Attention to
    recalibrate feature maps along both channel and spatial dimensions.

    Reference: Woo et al., "CBAM: Convolutional Block Attention Module", ECCV 2018.

    Args:
        in_channels:    Number of feature channels (1024 for DenseNet121).
        reduction:      Channel attention MLP reduction ratio (default 16).
        spatial_kernel: Spatial attention conv kernel size (default 7).
    """

    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x)   # channel recalibration
        x = x * self.spatial_att(x)   # spatial recalibration
        return x
