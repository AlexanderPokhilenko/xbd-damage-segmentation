import torch


@torch.no_grad()
def binary_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5,
                   eps: float = 1e-7) -> dict:
    """Per-batch binary segmentation metrics.
    Returns dict with dice, iou, pixel_acc, precision, recall.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (0, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1 - targets)).sum(dims)
    fn = ((1 - preds) * targets).sum(dims)
    tn = ((1 - preds) * (1 - targets)).sum(dims)

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": dice.mean().item(),
        "iou": iou.mean().item(),
        "pixel_acc": pixel_acc.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
    }


class MetricAccumulator:
    """Accumulate batch metrics over an epoch (weighted by batch size)."""
    def __init__(self):
        self.totals = {}
        self.n = 0

    def update(self, metrics: dict, batch_size: int):
        for k, v in metrics.items():
            self.totals[k] = self.totals.get(k, 0.0) + v * batch_size
        self.n += batch_size

    def compute(self) -> dict:
        if self.n == 0:
            return {}
        return {k: v / self.n for k, v in self.totals.items()}
