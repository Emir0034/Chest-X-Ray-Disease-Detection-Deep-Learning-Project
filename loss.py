import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """Asymmetric Loss for multi-label classification.

    Addresses class imbalance by treating positive and negative samples
    asymmetrically:
      - Positive samples use a soft focal weight with gamma_pos (default 1).
      - Negative samples use a stronger focal weight with gamma_neg (default 4),
        and their probability is shifted upward by `clip` before the log,
        which suppresses easy negatives (abundant "No Finding" samples).

    Reference: Ben-Baruch et al., "Asymmetric Loss For Multi-Label Classification",
               ICCV 2021.

    Args:
        gamma_neg: Focusing parameter for negative samples (default 4).
        gamma_pos: Focusing parameter for positive samples (default 1).
        clip:      Probability margin added to negative probabilities before log,
                   to shift and ignore easy negatives (default 0.05).
        eps:       Small constant for numerical stability in log (default 1e-8).
    """

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
        """Compute asymmetric loss.

        Args:
            logits:  Raw model outputs, shape (B, C). No sigmoid applied yet.
            targets: Float binary labels, shape (B, C), values in {0.0, 1.0}.

        Returns:
            Scalar loss value (mean over batch and classes).
        """
        xs_pos = torch.sigmoid(logits)           # p  — probability of positive
        xs_neg = 1.0 - xs_pos                    # 1-p — probability of negative

        # Shift negative probability up by `clip` and clamp to [0, 1].
        # This raises the effective threshold for counting a negative as "hard",
        # ignoring very easy negatives and reducing their gradient contribution.
        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1.0)

        # Standard binary log-likelihood terms
        los_pos = targets       * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss    = los_pos + los_neg

        # Asymmetric focal weighting: reduce contribution of easy samples.
        # pt is the model's probability for the correct class.
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            pt      = xs_pos * targets + xs_neg * (1.0 - targets)
            gamma   = self.gamma_pos * targets + self.gamma_neg * (1.0 - targets)
            weights = torch.pow(1.0 - pt, gamma)
            loss    = loss * weights

        return -loss.sum(dim=1).mean()


def get_loss_fn(use_asymmetric: bool) -> nn.Module:
    """Return the loss function for the current experiment.

    Both functions accept raw logits and float targets so the calling code
    in train.py does not need to change between experiments.

    Args:
        use_asymmetric: If True, return AsymmetricLoss; otherwise BCEWithLogitsLoss.
    """
    if use_asymmetric:
        return AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    return nn.BCEWithLogitsLoss()
