"""Quick smoke test: train each model for 2 epochs to verify end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.train import (load_config, make_dataloaders, make_optimizer,
                                make_scheduler, train_one_epoch, validate)
from src.models import custom_seg, unet_smp  # noqa
from src.models.registry import build_model
from src.training.losses import DiceBCELoss
from src.utils.device import device_info, get_device
from src.utils.seed import set_seed


MODELS = {
    "unet_smp":   {},
    "bafunet":    {"aggre_depth": 2},
    "fusion_unet": {"aggre_depth": 2},
    "resunet_pp": {},
}


def main():
    cfg = load_config(Path("configs/smoke.yaml"))
    cfg["training"]["epochs"] = 1
    set_seed(cfg.get("seed", 42))
    device = get_device()
    print(f"Device: {device_info(device)}")

    train_loader, val_loader = make_dataloaders(cfg)
    print(f"Train tiles: {len(train_loader.dataset)}  Val tiles: {len(val_loader.dataset)}")

    loss_fn = DiceBCELoss(alpha=0.5)
    use_amp = (device.type == "cuda")

    results = {}
    for name, params in MODELS.items():
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        model = build_model(name, encoder_name="resnet34", encoder_weights="imagenet",
                            in_channels=3, classes=1, **params).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  params: {n_params:,}")
        optimizer = make_optimizer(model, cfg)
        scheduler = make_scheduler(optimizer, cfg, cfg["training"]["epochs"])
        train_m = train_one_epoch(model, train_loader, optimizer, loss_fn, device, use_amp)
        val_m = validate(model, val_loader, loss_fn, device)
        print(f"  train: loss={train_m['loss']:.4f}  dice={train_m['dice']:.4f}")
        print(f"  val:   loss={val_m['loss']:.4f}  dice={val_m['dice']:.4f}")
        results[name] = {"params": n_params, "train": train_m, "val": val_m}
        del model, optimizer, scheduler

    print("\n=== Summary (1 epoch each) ===")
    print(f"{'model':15s}  {'params':>12s}  {'train_dice':>10s}  {'val_dice':>10s}")
    for name, r in results.items():
        print(f"{name:15s}  {r['params']:>12,}  {r['train']['dice']:>10.4f}  {r['val']['dice']:>10.4f}")


if __name__ == "__main__":
    main()
