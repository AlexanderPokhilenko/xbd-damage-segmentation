"""Universal training entry point.

Usage:
    python -m src.training.train --config configs/unet.yaml
    python -m src.training.train --config configs/unet.yaml --resume auto
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import XBDDataset
from src.models import unet_smp  # noqa: F401  (register models)
from src.models.registry import build_model
from src.training.augmentations import train_transform, val_transform
from src.training.checkpoint import (TrainHistory, find_resume_checkpoint,
                                     load_checkpoint, save_checkpoint)
from src.training.losses import DiceBCELoss
from src.training.metrics import MetricAccumulator, binary_metrics
from src.utils.device import device_info, get_device
from src.utils.logging import get_logger
from src.utils.seed import set_seed

log = get_logger("train")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def make_dataloaders(cfg: dict):
    data_root = Path(cfg["data"]["root"])
    tile_size = cfg["data"].get("tile_size", 256)
    bs = cfg["training"]["batch_size"]
    nw = cfg["training"].get("num_workers", 2)

    train_ds = XBDDataset(data_root, "train", transform=train_transform(tile_size))
    val_ds = XBDDataset(data_root, "val", transform=val_transform(tile_size))

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, pin_memory=True, drop_last=True,
                              persistent_workers=(nw > 0))
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, pin_memory=True,
                            persistent_workers=(nw > 0))
    return train_loader, val_loader


def make_optimizer(model, cfg: dict):
    opt_cfg = cfg["training"]["optimizer"]
    return torch.optim.AdamW(model.parameters(),
                             lr=opt_cfg["lr"],
                             weight_decay=opt_cfg.get("weight_decay", 1e-4))


def make_scheduler(optimizer, cfg: dict, epochs: int):
    sch_cfg = cfg["training"].get("scheduler", {})
    name = sch_cfg.get("name", "cosine")
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                          factor=0.5, patience=3)
    return None


def train_one_epoch(model, loader, optimizer, loss_fn, device, use_amp: bool):
    model.train()
    accum = MetricAccumulator()
    loss_sum = 0.0
    n = 0
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    pbar = tqdm(loader, desc="train", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).long()
        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(images)
                loss = loss_fn(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
        bs = images.size(0)
        loss_sum += loss.item() * bs
        n += bs
        m = binary_metrics(logits.detach().float(), masks)
        accum.update(m, bs)
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{m['dice']:.3f}")
    metrics = accum.compute()
    metrics["loss"] = loss_sum / max(n, 1)
    return metrics


@torch.no_grad()
def validate(model, loader, loss_fn, device):
    model.eval()
    accum = MetricAccumulator()
    loss_sum = 0.0
    n = 0
    for images, masks in tqdm(loader, desc="val", leave=False):
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
    parser.add_argument("--resume", default="none", choices=["none", "auto"],
                        help="auto = resume from last.pt if exists")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name
    run_name = cfg.get("run_name") or args.config.stem
    seed = cfg.get("seed", 42)
    set_seed(seed)

    device = get_device()
    log.info(f"Device: {device_info(device)}")

    run_dir = Path(cfg.get("checkpoint_dir", "./checkpoints")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run dir: {run_dir}")

    train_loader, val_loader = make_dataloaders(cfg)
    log.info(f"Train tiles: {len(train_loader.dataset)}  Val tiles: {len(val_loader.dataset)}")

    model = build_model(cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {cfg['model']['name']}  trainable params: {n_params:,}")

    epochs = cfg["training"]["epochs"]
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg, epochs)

    loss_cfg = cfg["training"].get("loss", {})
    loss_fn = DiceBCELoss(alpha=loss_cfg.get("dice_weight", 0.5),
                          pos_weight=loss_cfg.get("pos_weight", None))

    use_amp = (device.type == "cuda") and cfg["training"].get("amp", True)
    log.info(f"AMP: {use_amp}")

    history = TrainHistory()
    start_epoch = 0
    best_val_dice = -1.0
    best_epoch = -1
    early_stop_patience = cfg["training"].get("early_stop_patience", 7)
    no_improve = 0

    if args.resume == "auto":
        ckpt_path = find_resume_checkpoint(run_dir)
        if ckpt_path is not None:
            log.info(f"Resuming from {ckpt_path}")
            state = load_checkpoint(ckpt_path, model=model, optimizer=optimizer,
                                    scheduler=scheduler, device=device)
            start_epoch = state["epoch"] + 1
            best_val_dice = state.get("best_val_dice", -1.0)
            best_epoch = state.get("best_epoch", -1)
            history.train = state["history"]["train"]
            history.val = state["history"]["val"]
            no_improve = max(0, start_epoch - 1 - best_epoch)
            log.info(f"  start epoch={start_epoch}  best_val_dice={best_val_dice:.4f}  best_epoch={best_epoch}")
        else:
            log.info("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        log.info(f"=== Epoch {epoch+1}/{epochs} ===")
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn, device, use_amp)
        val_m = validate(model, val_loader, loss_fn, device)
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_m["dice"])
            else:
                scheduler.step()

        train_m["epoch"] = epoch
        val_m["epoch"] = epoch
        history.train.append(train_m)
        history.val.append(val_m)

        improved = val_m["dice"] > best_val_dice
        if improved:
            best_val_dice = val_m["dice"]
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1

        save_checkpoint(run_dir / "last.pt", model=model, optimizer=optimizer,
                        scheduler=scheduler, epoch=epoch,
                        best_val_dice=best_val_dice, best_epoch=best_epoch,
                        history=history, config=cfg)
        if improved:
            save_checkpoint(run_dir / "best.pt", model=model, optimizer=optimizer,
                            scheduler=scheduler, epoch=epoch,
                            best_val_dice=best_val_dice, best_epoch=best_epoch,
                            history=history, config=cfg)

        dt = time.time() - t0
        log.info(f"  train  loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}  iou={train_m['iou']:.4f}")
        log.info(f"  val    loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}"
                 f"  {'(best)' if improved else ''}  ({dt:.1f}s)")

        if no_improve >= early_stop_patience:
            log.info(f"Early stopping: no val Dice improvement for {early_stop_patience} epochs")
            break

    with open(run_dir / "history.json", "w") as f:
        json.dump({"train": history.train, "val": history.val,
                   "best_val_dice": best_val_dice, "best_epoch": best_epoch}, f, indent=2)
    # Save training curves
    try:
        from src.training.plots import plot_run_curves
        plot_run_curves(history, run_dir / "curves.png", title=run_name)
        log.info(f"Curves saved to {run_dir / 'curves.png'}")
    except Exception as e:
        log.warning(f"Failed to save curves: {e}")

    log.info(f"Done. Best val Dice = {best_val_dice:.4f} at epoch {best_epoch+1}")


if __name__ == "__main__":
    main()
