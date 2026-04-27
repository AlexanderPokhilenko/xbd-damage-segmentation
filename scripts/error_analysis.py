"""Per-tile Dice analysis: distribution, breakdown by disaster, worst cases.

Excludes tiles with no GT damage from Dice statistics (Dice undefined for empty GT).
Reports false positive rate separately for negative tiles.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

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
            tp = int(((pred == 1) & (gt == 1)).sum())
            fp = int(((pred == 1) & (gt == 0)).sum())
            fn = int(((pred == 0) & (gt == 1)).sum())
            tn = int(((pred == 0) & (gt == 0)).sum())
            gt_pos = int((gt == 1).sum())
            pred_pos = int((pred == 1).sum())
            n_pixels = gt.size

            has_gt = gt_pos > 0
            # Dice/IoU only defined when GT has positive pixels OR pred has positive pixels.
            # Strictly: skip tiles with gt_pos == 0 from positive-tile statistics.
            if has_gt:
                dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            else:
                dice = np.nan
                iou = np.nan

            # FPR for empty-GT tiles: fraction of background pixels predicted as damage
            fpr = fp / n_pixels if not has_gt else np.nan

            rows.append({
                "tile_id": meta["tile_id"],
                "disaster": meta["disaster"],
                "damage_ratio": meta["damage_ratio"],
                "has_gt_damage": has_gt,
                "gt_pixels": gt_pos,
                "pred_pixels": pred_pos,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "dice": dice,
                "iou": iou,
                "fpr_on_empty": fpr,
            })

    df = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_dir / "per_tile.csv", index=False)

    pos = df[df["has_gt_damage"]].copy()
    neg = df[~df["has_gt_damage"]].copy()

    # Per-disaster summary on POSITIVE tiles
    by_dis = pos.groupby("disaster").agg(
        n_positive=("dice", "size"),
        mean_dice=("dice", "mean"),
        median_dice=("dice", "median"),
        mean_iou=("iou", "mean"),
    ).round(4).reset_index()

    # Add empty-tile stats per disaster
    neg_by_dis = neg.groupby("disaster").agg(
        n_empty=("fpr_on_empty", "size"),
        mean_fpr=("fpr_on_empty", "mean"),
    ).round(6).reset_index()

    summary_table = by_dis.merge(neg_by_dis, on="disaster", how="left")

    summary_table.to_csv(args.output_dir / "by_disaster.csv", index=False)
    print("\n=== By disaster (positive tiles only for Dice/IoU) ===")
    print(summary_table.to_string(index=False))

    # Histogram of Dice (positive tiles only)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pos["dice"], bins=30, edgecolor="black", alpha=0.75)
    if len(pos):
        ax.axvline(pos["dice"].mean(), color="red", linestyle="--",
                   label=f"mean={pos['dice'].mean():.3f}")
        ax.axvline(pos["dice"].median(), color="orange", linestyle=":",
                   label=f"median={pos['dice'].median():.3f}")
    ax.set_xlabel("per-tile Dice (positive tiles only)")
    ax.set_ylabel("count")
    ax.set_title(f"{run_name} -- Dice distribution on {args.split} ({len(pos)} positive tiles)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dice_distribution.png", dpi=120)
    plt.close(fig)

    # Dice vs damage_ratio scatter (positive tiles only)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(pos["damage_ratio"], pos["dice"], alpha=0.4, s=12)
    ax.set_xlabel("damage_ratio (fraction of damaged pixels in tile)")
    ax.set_ylabel("per-tile Dice")
    ax.set_title("Dice vs ground-truth damage ratio (positive tiles)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output_dir / "dice_vs_damage_ratio.png", dpi=120)
    plt.close(fig)

    # FPR distribution on empty tiles (how badly the model false-positives on clean ground)
    if len(neg):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(neg["fpr_on_empty"], bins=30, edgecolor="black", alpha=0.75, color="darkred")
        ax.axvline(neg["fpr_on_empty"].mean(), color="black", linestyle="--",
                   label=f"mean FPR={neg['fpr_on_empty'].mean():.4f}")
        ax.set_xlabel("false-positive pixel rate (empty-GT tiles only)")
        ax.set_ylabel("count")
        ax.set_title(f"{run_name} -- FPR on {len(neg)} empty tiles")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.output_dir / "fpr_on_empty.png", dpi=120)
        plt.close(fig)

    summary = {
        "split": args.split,
        "n_total": int(len(df)),
        "n_positive": int(len(pos)),
        "n_empty": int(len(neg)),
        "positive_tile_metrics": {
            "mean_dice": float(pos["dice"].mean()) if len(pos) else None,
            "median_dice": float(pos["dice"].median()) if len(pos) else None,
            "mean_iou": float(pos["iou"].mean()) if len(pos) else None,
        },
        "empty_tile_metrics": {
            "mean_fpr": float(neg["fpr_on_empty"].mean()) if len(neg) else None,
        },
        "by_disaster": summary_table.to_dict(orient="records"),
    }
    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal: {len(df)} tiles ({len(pos)} positive + {len(neg)} empty)")
    print(f"Saved analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
