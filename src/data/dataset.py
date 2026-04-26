import json
from pathlib import Path
from typing import Optional, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class XBDDataset(Dataset):
    """xBD damage segmentation dataset (binary).

    Reads tiles prepared by src.data.prepare_xbd. Returns (image, mask) pairs:
        image: HxWx3 uint8 numpy array (or transformed tensor)
        mask:  HxW   uint8 numpy array with values {0, 1}
    """
    def __init__(self, root: Path, split: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.tile_size = manifest["tile_size"]
        self.items = [t for t in manifest["tiles"] if t["split"] == split]
        if not self.items:
            raise ValueError(f"No tiles found for split='{split}' in {manifest_path}")
        self.transform = transform
        self.split = split

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        image = np.array(Image.open(self.root / item["image_path"]).convert("RGB"))
        mask = np.array(Image.open(self.root / item["mask_path"]))
        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        return image, mask

    def get_meta(self, idx: int) -> dict:
        return self.items[idx]
