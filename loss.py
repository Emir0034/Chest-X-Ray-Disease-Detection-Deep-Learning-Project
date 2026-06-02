import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip      = clip
        self.eps       = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        logits  = logits.float()
        targets = targets.float()

        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos

        # Shift and clamp negative probabilities
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(min=self.eps, max=1.0 - self.eps)
        else:
            xs_neg = xs_neg.clamp(min=self.eps, max=1.0 - self.eps)

        # Clamp positive probabilities
        xs_pos = xs_pos.clamp(min=self.eps, max=1.0 - self.eps)

        los_pos = targets       * torch.log(xs_pos)
        los_neg = (1 - targets) * torch.log(xs_neg)
        loss    = los_pos + los_neg

        # Asymmetric focal weighting
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt      = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma   = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weights = torch.pow(1.0 - pt, gamma)
            loss    = loss * weights

        return -loss.sum(dim=1).mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # None or scalar float in (0, 1)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        pt   = torch.exp(-bce_loss)
        loss = (1.0 - pt) ** self.gamma * bce_loss
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss
        return loss.mean()


def get_loss_fn(
    use_asymmetric: bool,
    gamma_neg: float = 4.0,
    gamma_pos: float = 1.0,
    clip: float = 0.05,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha=None,
) -> nn.Module:
    if use_asymmetric:
        return AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
    if use_focal:
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
    return nn.BCEWithLogitsLoss()
