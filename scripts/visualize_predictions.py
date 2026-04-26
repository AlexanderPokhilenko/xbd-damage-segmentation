"""Visualize predictions: image | ground truth | prediction grid.

Usage:
    python scripts/visualize_predictions.py --config configs/bafunet.yaml \
        --output results/bafunet_predictions.png --n-samples 12
"""
import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.data.dataset import XBDDataset
from src.models import custom_seg, unet_smp  # noqa: F401
from src.models.registry import build_model
from src.training.augmentations import val_transform, IMAGENET_MEAN, IMAGENET_STD
from src.training.checkpoint import load_checkpoint
from src.training.train import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def denormalize(img_tensor):
    """Reverse ImageNet normalization to get displayable RGB."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    img = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img


def make_overlay(image_np: np.ndarray, mask_np: np.ndarray, color=(220, 60, 60), alpha=0.5):
    overlay = image_np.copy()
    sel = mask_np > 0
    if sel.any():
        for c, val in enumerate(color):
            overlay[sel, c] = (overlay[sel, c] * (1 - alpha) + val * alpha).astype(np.uint8)
    return overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="mixed",
                        choices=["mixed", "best", "worst", "random"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    run_name = cfg.get("run_name") or args.config.stem
    run_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / run_name
    ckpt_path = run_dir / args.checkpoint

    data_root = Path(cfg["data"]["root"])
    tile_size = cfg["data"].get("tile_size", 256)
    ds = XBDDataset(data_root, args.split, transform=val_transform(tile_size))

    model = build_model(cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)
    load_checkpoint(ckpt_path, model=model, device=device)
    model.eval()

    # Compute per-tile Dice, then pick samples by strategy
    per_tile = []
    with torch.no_grad():
        for idx in range(len(ds)):
            image, mask = ds[idx]
            x = image.unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= 0.5).astype(np.uint8)
            gt = mask.numpy().astype(np.uint8) if torch.is_tensor(mask) else mask.astype(np.uint8)
            inter = ((pred == 1) & (gt == 1)).sum()
            denom = (pred == 1).sum() + (gt == 1).sum()
            dice = (2 * inter + 1) / (denom + 1)
            per_tile.append({"idx": idx, "dice": float(dice)})

    rng = np.random.default_rng(args.seed)
    if args.strategy == "best":
        chosen = sorted(per_tile, key=lambda r: -r["dice"])[:args.n_samples]
    elif args.strategy == "worst":
        # exclude tiles with no GT damage to avoid trivial 0/0 cases
        chosen = sorted(per_tile, key=lambda r: r["dice"])[:args.n_samples]
    elif args.strategy == "random":
        idxs = rng.choice(len(per_tile), size=args.n_samples, replace=False)
        chosen = [per_tile[i] for i in idxs]
    else:  # mixed
        n_each = args.n_samples // 3
        best = sorted(per_tile, key=lambda r: -r["dice"])[:n_each]
        worst = sorted(per_tile, key=lambda r: r["dice"])[:n_each]
        random_pick = list(rng.choice(len(per_tile), size=args.n_samples - 2 * n_each, replace=False))
        chosen = best + worst + [per_tile[i] for i in random_pick]

    # Render
    n_cols = 3
    n_rows = len(chosen)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    if n_rows == 1:
        axes = axes[None, :]

    for row, info in enumerate(chosen):
        idx = info["idx"]
        meta = ds.get_meta(idx)
        image_t, mask_t = ds[idx]
        with torch.no_grad():
            logits = model(image_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= 0.5).astype(np.uint8)
        gt = mask_t.numpy() if torch.is_tensor(mask_t) else mask_t
        img_rgb = denormalize(image_t)

        axes[row, 0].imshow(img_rgb)
        axes[row, 0].set_title(f"image\n{meta['disaster']}", fontsize=9)
        axes[row, 1].imshow(make_overlay(img_rgb, gt, color=(60, 200, 60)))
        axes[row, 1].set_title("ground truth (green)", fontsize=9)
        axes[row, 2].imshow(make_overlay(img_rgb, pred, color=(220, 60, 60)))
        axes[row, 2].set_title(f"prediction (red)\nDice={info['dice']:.3f}", fontsize=9)
        for c in range(n_cols):
            axes[row, c].axis("off")

    fig.suptitle(f"{run_name} on {args.split} ({args.strategy})", y=1.001)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
