"""Per-tile Dice analysis: distribution, breakdown by disaster, worst cases.

Usage:
    python scripts/error_analysis.py --config configs/bafunet.yaml \
        --output-dir results/bafunet_errors
"""
import sys
import argparse
import json
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.dataset import XBDDataset
from src.models import custom_seg, unet_smp  # noqa: F401
from src.models.registry import build_model
from src.training.augmentations import val_transform
from src.training.checkpoint import load_checkpoint
from src.training.train import load_config
from src.utils.device import get_device
from src.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
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

    rows = []
    with torch.no_grad():
        for idx in range(len(ds)):
            image, mask = ds[idx]
            meta = ds.get_meta(idx)
            x = image.unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits[0, 0]).cpu().numpy() >= 0.5).astype(np.uint8)
            gt = mask.numpy().astype(np.uint8) if torch.is_tensor(mask) else mask.astype(np.uint8)
            tp = ((pred == 1) & (gt == 1)).sum()
            fp = ((pred == 1) & (gt == 0)).sum()
            fn = ((pred == 0) & (gt == 1)).sum()
            dice = (2 * tp + 1) / (2 * tp + fp + fn + 1)
            iou = (tp + 1) / (tp + fp + fn + 1)
            rows.append({
                "tile_id": meta["tile_id"],
                "disaster": meta["disaster"],
                "damage_ratio": meta["damage_ratio"],
                "dice": float(dice),
                "iou": float(iou),
                "tp": int(tp), "fp": int(fp), "fn": int(fn),
            })

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "per_tile.csv", index=False)

    # Per-disaster summary
    by_dis = df.groupby("disaster").agg(
        n=("dice", "size"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        mean_iou=("iou", "mean"),
    ).round(4).reset_index()
    by_dis.to_csv(args.output_dir / "by_disaster.csv", index=False)
    print("\n=== By disaster ===")
    print(by_dis.to_string(index=False))

    # Histogram of Dice
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["dice"], bins=30, edgecolor="black", alpha=0.75)
    ax.axvline(df["dice"].mean(), color="red", linestyle="--",
               label=f"mean={df['dice'].mean():.3f}")
    ax.set_xlabel("per-tile Dice")
    ax.set_ylabel("count")
    ax.set_title(f"{run_name} -- Dice distribution on {args.split}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dice_distribution.png", dpi=120)
    plt.close(fig)

    # Dice vs damage_ratio scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(df["damage_ratio"], df["dice"], alpha=0.4, s=12)
    ax.set_xlabel("damage_ratio (fraction of damaged pixels in tile)")
    ax.set_ylabel("per-tile Dice")
    ax.set_title("Dice vs ground-truth damage ratio")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dice_vs_damage_ratio.png", dpi=120)
    plt.close(fig)

    # Summary
    summary = {
        "split": args.split,
        "n_tiles": len(df),
        "mean_dice": float(df["dice"].mean()),
        "median_dice": float(df["dice"].median()),
        "mean_iou": float(df["iou"].mean()),
        "by_disaster": by_dis.to_dict(orient="records"),
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
