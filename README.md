# xBD Damage Segmentation for Semantic Consistency Verification

PyTorch implementation comparing four U-Net family architectures (U-Net, BAFUNet, FusionUNet, ResUNet++) on the xBD building-damage segmentation task. All models share a ResNet34 ImageNet-pretrained encoder for fair architectural comparison. Includes structural-parametric ablation of BAFUNet's `aggre_depth` parameter.

This module is designed for integration into a semantic consistency framework for verifying news reports about emergency events (floods, wildfires, building damage) using satellite imagery — a sub-task of broader research on intelligent text analysis in electronic media.

## Repository structure

```
.
├── configs/             # YAML configs for each model
├── src/
│   ├── data/            # xBD preprocessing and PyTorch Dataset
│   ├── models/          # U-Net (SMP), BAFUNet, FusionUNet, ResUNet++
│   ├── training/        # train loop, losses, metrics, checkpoint, plots
│   └── utils/           # device, seed, logging
├── scripts/             # evaluate, ablation, visualization, error analysis
├── notebooks/
│   └── demo.ipynb       # end-to-end reproducible experiment
└── results/             # generated tables and plots (gitignored)
```

## Installation

Requires Python 3.9+. Tested on PyTorch 2.1+ (CUDA, MPS, CPU).

```bash
# For training (Linux/macOS/Windows, with optional CUDA)
pip install -r requirements.txt
pip install -e .

# Additional dependencies for local data preprocessing only
pip install -r requirements-prep.txt
```

## Data preparation

1. Register at https://xview2.org/ and download the **Challenge training set** (~7.8 GB).
2. Extract into `./train/` so that `train/images/` and `train/labels/` exist.
3. Run preprocessing (filters wildfire and flood disasters, tiles into 256×256, splits 70/15/15 per image within each disaster):

```bash
python -m src.data.prepare_xbd \
    --xbd-root ./train \
    --output ./xbd_processed \
    --tile-size 256 \
    --min-damage-ratio 0.005 \
    --negative-ratio 0.15 \
    --seed 42
```

Inspect overlays in `xbd_processed/previews/` to verify mask alignment.

## Reproducing the full experiment

The simplest path is the demo notebook:

```bash
jupyter lab notebooks/demo.ipynb
```

The notebook performs: training of 4 main models, BAFUNet ablation (3 runs), test-set evaluation, comparison plots, prediction visualization, error analysis. Resume is automatic — interrupting and rerunning continues from the last checkpoint.

### Step-by-step alternative

```bash
# Train each model individually
python -m src.training.train --config configs/unet.yaml
python -m src.training.train --config configs/bafunet.yaml
python -m src.training.train --config configs/fusion_unet.yaml
python -m src.training.train --config configs/resunet_pp.yaml

# Ablation
python scripts/run_ablation.py

# Evaluate any config on test split
python scripts/evaluate.py --config configs/bafunet.yaml

# Visualize predictions
python scripts/visualize_predictions.py --config configs/bafunet.yaml \
    --output results/predictions.png --n-samples 12 --strategy mixed

# Error analysis
python scripts/error_analysis.py --config configs/bafunet.yaml \
    --output-dir results/errors_bafunet
```

## Method overview

**Task**: binary semantic segmentation of post-disaster damage on RGB satellite imagery (256×256 tiles).

**Encoder**: ResNet34 pretrained on ImageNet (via `segmentation_models_pytorch`), used identically by all four models.

**Decoders**:
- *U-Net* (baseline): standard SMP U-Net decoder.
- *BAFUNet*: ASPP bottleneck + multi-round bidirectional FuseModule between encoder skip features (with ECA channel attention) + plain U-Net upsample chain.
- *FusionUNet*: two-round FuseModule + CCA channel-cross-attention skip connections (no ASPP).
- *ResUNet++* (adapted): SMP UnetPlusPlus with SCSE attention. ResNet34 encoder already provides residual blocks and ImageNet representation, preserving the spirit of the original.

**Loss**: composite Dice + BCE (equal weights).
**Optimizer**: AdamW, lr=1e-4, weight decay 1e-4.
**Scheduler**: CosineAnnealingLR over max epochs.
**Regularization**: ImageNet pretraining, weight decay, BatchNorm in custom blocks, training augmentations (flips, rotations, brightness/contrast, gaussian noise), early stopping on val Dice (patience=7).

**Data split**: 70/15/15 per image within each disaster event. Per-tile leakage is avoided because tiles inherit their parent image's split.

## Results

After running `notebooks/demo.ipynb`, see:
- `results/main_comparison.csv` — final test metrics for all 4 models
- `results/ablation.csv` — BAFUNet over `aggre_depth ∈ {1, 2, 3}`
- `results/compare_val_dice.png`, `results/compare_val_loss.png` — training curves
- `results/predictions_grid.png` — qualitative examples
- `results/errors_<run>/` — per-tile distributions and per-disaster breakdown

## Citation and references

The custom decoder modules (FuseModule, CCA, ECA, ASPP) are adapted from:
- BAFUNet (own prior work on spine MRI segmentation, see [BAFUNet: Hybrid U-Net for Segmentation of Spine MR Images](https://doi.org/10.18372/1990-5548.82.19365))
- FusionU-Net ([FusionU-Net: U-Net with Enhanced Skip Connection for Pathology Image Segmentation](https://doi.org/10.48550/arXiv.2310.10951))
- ResUNet++ ([ResUNet++: An Advanced Architecture for Medical Image Segmentation](https://doi.org/10.48550/arXiv.1911.07067))

xBD dataset: https://xview2.org/
