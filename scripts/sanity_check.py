import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.registry import build_model, list_models
from src.models import custom_seg, unet_smp  # noqa

print("Registered models:", list_models())
x = torch.randn(2, 3, 256, 256)
for name in ["unet_smp", "bafunet", "fusion_unet", "resunet_pp"]:
    m = build_model(name, encoder_name="resnet34", encoder_weights=None,
                    in_channels=3, classes=1)
    m.eval()
    with torch.no_grad():
        y = m(x)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {name:15s}  out={tuple(y.shape)}  params={n_params:,}")
