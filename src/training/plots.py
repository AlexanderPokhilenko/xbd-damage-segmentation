"""Training curve plots and result aggregation."""
import json
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt

from src.training.checkpoint import TrainHistory


METRICS_TO_PLOT = ["loss", "dice", "iou"]


def plot_run_curves(history: TrainHistory, output_path: Path,
                    title: Optional[str] = None) -> None:
    """Plot train/val curves for one run on a single figure with subplots."""
    if not history.train or not history.val:
        return

    epochs_train = [e["epoch"] + 1 for e in history.train]
    epochs_val = [e["epoch"] + 1 for e in history.val]

    fig, axes = plt.subplots(1, len(METRICS_TO_PLOT),
                             figsize=(5 * len(METRICS_TO_PLOT), 4))
    if len(METRICS_TO_PLOT) == 1:
        axes = [axes]

    for ax, metric in zip(axes, METRICS_TO_PLOT):
        train_vals = [e.get(metric) for e in history.train]
        val_vals = [e.get(metric) for e in history.val]
        ax.plot(epochs_train, train_vals, "o-", label=f"train {metric}", markersize=3)
        ax.plot(epochs_val, val_vals, "s-", label=f"val {metric}", markersize=3)
        ax.set_xlabel("epoch")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_title(metric)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_run_history(run_dir: Path) -> Optional[dict]:
    """Load history.json from a run directory; returns None if missing."""
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        return None
    with open(hist_path) as f:
        return json.load(f)


def plot_compare_runs(run_dirs: List[Path], output_path: Path,
                      metric: str = "dice", split: str = "val",
                      title: Optional[str] = None) -> None:
    """Compare a single metric across multiple runs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for run_dir in run_dirs:
        hist = load_run_history(run_dir)
        if hist is None:
            continue
        entries = hist[split]
        epochs = [e["epoch"] + 1 for e in entries]
        vals = [e.get(metric) for e in entries]
        ax.plot(epochs, vals, "o-", label=run_dir.name, markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{split} {metric}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def build_summary_table(run_dirs: List[Path]) -> List[dict]:
    """Aggregate best val metrics across runs into a list of dicts."""
    rows = []
    for run_dir in run_dirs:
        hist = load_run_history(run_dir)
        if hist is None:
            continue
        # find best val epoch by dice
        val = hist["val"]
        if not val:
            continue
        best = max(val, key=lambda e: e.get("dice", -1))
        rows.append({
            "run": run_dir.name,
            "best_epoch": best["epoch"] + 1,
            "val_dice": best["dice"],
            "val_iou": best["iou"],
            "val_pixel_acc": best["pixel_acc"],
            "val_precision": best["precision"],
            "val_recall": best["recall"],
            "val_loss": best["loss"],
            "n_epochs_trained": len(val),
        })
    return rows
