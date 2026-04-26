"""Evaluate a trained model on the test split.

Usage:
    python scripts/evaluate.py --config configs/unet.yaml
    python scripts/evaluate.py --config configs/unet.yaml --checkpoint best.pt --split test
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import XBDDataset
from src.models import custom_seg, unet_smp  # noqa: F401
from src.models.registry import build_model
from src.training.augmentations import val_transform
from src.training.checkpoint import load_checkpoint
from src.training.losses import DiceBCELoss
from src.training.metrics import MetricAccumulator, binary_metrics
from src.training.train import load_config
from src.utils.device import device_info, get_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("evaluate")


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    accum = MetricAccumulator()
    loss_sum = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="eval"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        logits = model(images)
        loss = loss_fn(logits, masks)
        bs = images.size(0)
        loss_sum += loss.item() * bs
        n += bs
        accum.update(binary_metrics(logits.float(), masks), bs)
    metrics = accum.compute()
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str, default="best.pt",
                        help="Checkpoint name within run dir")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))
    device = get_device()
    log.info(f"Device: {device_info(device)}")

    run_name = cfg.get("run_name") or args.config.stem
    run_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / run_name
    ckpt_path = run_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    data_root = Path(cfg["data"]["root"])
    tile_size = cfg["data"].get("tile_size", 256)
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 2)

    ds = XBDDataset(data_root, args.split, transform=val_transform(tile_size))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    log.info(f"{args.split} tiles: {len(ds)}")

    model = build_model(cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)
    log.info(f"Loading {ckpt_path}")
    state = load_checkpoint(ckpt_path, model=model, device=device)
    log.info(f"  ckpt epoch={state.get('epoch')}  best_val_dice={state.get('best_val_dice'):.4f}")

    loss_fn = DiceBCELoss(alpha=cfg["training"].get("loss", {}).get("dice_weight", 0.5))
    metrics = evaluate(model, loader, loss_fn, device)

    print(f"\n=== {run_name} on {args.split} ===")
    for k, v in metrics.items():
        print(f"  {k:14s} {v:.4f}")

    out_path = run_dir / f"{args.split}_metrics.json"
    with open(out_path, "w") as f:
        json.dump({"split": args.split, "checkpoint": args.checkpoint,
                   "metrics": metrics}, f, indent=2)
    log.info(f"Saved {out_path}")


if __name__ == "__main__":
    main()
