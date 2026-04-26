"""BAFUNet ablation over aggre_depth.

Runs three trainings with aggre_depth in {1, 2, 3}.
Each is logged to checkpoints/bafunet_ablation_d{N}/.
"""
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def make_config(base_path: Path, aggre_depth: int, run_name: str, out_path: Path):
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    cfg["run_name"] = run_name
    cfg["model"]["params"]["aggre_depth"] = aggre_depth
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    base_cfg = ROOT / "configs" / "bafunet.yaml"
    tmp_dir = ROOT / "configs" / "_ablation"

    for depth in [1, 2, 3]:
        run_name = f"bafunet_ablation_d{depth}"
        cfg_path = tmp_dir / f"{run_name}.yaml"
        make_config(base_cfg, aggre_depth=depth, run_name=run_name, out_path=cfg_path)
        print(f"\n{'='*70}\nAblation: aggre_depth={depth}\n{'='*70}")
        result = subprocess.run(
            ["python", "-m", "src.training.train",
             "--config", str(cfg_path), "--resume", "auto"],
            cwd=str(ROOT),
        )
        if result.returncode != 0:
            print(f"!! aggre_depth={depth} failed")
            break
        # Also evaluate on test
        subprocess.run(
            ["python", str(ROOT / "scripts" / "evaluate.py"),
             "--config", str(cfg_path)],
            cwd=str(ROOT),
        )


if __name__ == "__main__":
    main()
