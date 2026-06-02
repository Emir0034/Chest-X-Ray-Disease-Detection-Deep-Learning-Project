import torch
import torch.nn as nn

# Channel Attention Module from CBAM
# Convolutional Block Attention Module (CBAM)
class ChannelAttention(nn.Module):

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
        avg = self.avg_pool(x).view(B, C)
        mx  = self.max_pool(x).view(B, C)
        att = self.sigmoid(self.shared_mlp(avg) + self.shared_mlp(mx))
        return att.view(B, C, 1, 1)

# Spatial Attention Module from CBAM
# Spatial attention: highlight important regions
class SpatialAttention(nn.Module):

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = kernel_size // 2
        self.conv    = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)            # (B, 1, H, W)
        max_out = x.max(dim=1, keepdim=True).values      # (B, 1, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        return self.sigmoid(self.conv(combined))         # (B, 1, H, W)


class CBAM(nn.Module):

    def __init__(self, in_channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_att(x) #  ne: hangi özellikler önemli
        x = x * self.spatial_att(x) # nerede: hangi bölgeler önemli
        return x
