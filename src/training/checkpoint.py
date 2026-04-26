from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

from src.utils.seed import get_rng_states, set_rng_states


@dataclass
class TrainHistory:
    train: list = field(default_factory=list)  # list of dicts per epoch
    val: list = field(default_factory=list)


def save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch: int,
                    best_val_dice: float, best_epoch: int,
                    history: TrainHistory, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "best_epoch": best_epoch,
        "history": {"train": history.train, "val": history.val},
        "rng_states": get_rng_states(),
        "config": config,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, *, model, optimizer=None, scheduler=None,
                    device: torch.device = None, strict: bool = True) -> dict:
    state = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(state["model_state"], strict=strict)
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])
    if state.get("rng_states") is not None:
        try:
            set_rng_states(state["rng_states"])
        except Exception:
            pass
    return state


def find_resume_checkpoint(run_dir: Path) -> Optional[Path]:
    last = run_dir / "last.pt"
    return last if last.exists() else None
