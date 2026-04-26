import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.
    Expects logits of shape (B, 1, H, W) and targets (B, H, W) or (B, 1, H, W) in {0,1}.
    """
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        dims = (0, 2, 3)
        inter = (probs * targets).sum(dims)
        denom = probs.sum(dims) + targets.sum(dims)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class DiceBCELoss(nn.Module):
    """Composite loss: alpha * Dice + (1 - alpha) * BCEWithLogits."""
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0,
                 pos_weight: float = None):
        super().__init__()
        self.alpha = alpha
        self.dice = DiceLoss(smooth=smooth)
        pw = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if targets.dim() == 3:
            t_bce = targets.unsqueeze(1).float()
        else:
            t_bce = targets.float()
        return self.alpha * self.dice(logits, targets) + (1 - self.alpha) * self.bce(logits, t_bce)
