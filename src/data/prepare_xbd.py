"""
Preprocess xBD training set into a tiled binary damage segmentation dataset.

Split strategy: 70/15/15 PER IMAGE within each disaster event (no tile-level leakage).

Usage:
    python -m src.data.prepare_xbd \
        --xbd-root ./train \
        --output ./xbd_processed \
        --tile-size 256 \
        --min-damage-ratio 0.005 \
        --negative-ratio 0.15 \
        --seed 42
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.features import rasterize
from shapely import wkt
from tqdm import tqdm

DAMAGED_SUBTYPES = {"minor-damage", "major-damage", "destroyed"}
KEYWORDS = ("fire", "flood")
TILE_SIZE_DEFAULT = 256


def is_relevant(filename: str) -> bool:
    name = filename.lower()
    return "post_disaster" in name and any(k in name for k in KEYWORDS)


def disaster_from_filename(filename: str) -> str:
    return Path(filename).stem.split("_")[0]


def parse_polygons(label_json: dict) -> list:
    """Return list of (shapely_polygon, is_damaged, subtype)."""
    polys = []
    features = label_json.get("features", {}).get("xy", [])
    for feat in features:
        wkt_str = feat.get("wkt")
        if not wkt_str:
            continue
        try:
            geom = wkt.loads(wkt_str)
        except Exception:
            continue
        if geom.is_empty:
            continue
        subtype = feat.get("properties", {}).get("subtype", "un-classified")
        is_damaged = subtype in DAMAGED_SUBTYPES
        polys.append((geom, is_damaged, subtype))
    return polys


def rasterize_mask(polys: list, height: int, width: int) -> np.ndarray:
    damaged_shapes = [(geom, 1) for geom, is_dmg, _ in polys if is_dmg]
    if not damaged_shapes:
        return np.zeros((height, width), dtype=np.uint8)
    return rasterize(
        damaged_shapes,
        out_shape=(height, width),
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )


def tile_image_and_mask(image: np.ndarray, mask: np.ndarray, tile_size: int):
    h, w = mask.shape
    for r in range(0, h - tile_size + 1, tile_size):
        for c in range(0, w - tile_size + 1, tile_size):
            yield r, c, image[r:r + tile_size, c:c + tile_size], mask[r:r + tile_size, c:c + tile_size]


def split_per_image(images_by_disaster: dict, seed: int, ratios=(0.70, 0.15, 0.15)) -> dict:
    """For each disaster, split its images into train/val/test by ratios.
    Returns mapping {image_path: split_name}.
    """
    rng = random.Random(seed)
    image_split = {}
    for disaster, files in sorted(images_by_disaster.items()):
        files_sorted = sorted(files)
        rng.shuffle(files_sorted)
        n = len(files_sorted)
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test < 1 and n >= 3:
            n_val -= 1
            n_test = 1
        for i, p in enumerate(files_sorted):
            if i < n_train:
                image_split[p] = "train"
            elif i < n_train + n_val:
                image_split[p] = "val"
            else:
                image_split[p] = "test"
    return image_split


def generate_previews(manifest: dict, output_dir: Path, n: int = 60, seed: int = 42) -> None:
    rng = random.Random(seed)
    tiles = manifest["tiles"]
    with_dmg = sorted([t for t in tiles if t["damage_ratio"] > 0.01], key=lambda t: -t["damage_ratio"])
    no_dmg = [t for t in tiles if t["damage_ratio"] == 0.0]

    n_dmg = min(len(with_dmg), int(n * 0.8))
    n_neg = min(len(no_dmg), n - n_dmg)
    sample = rng.sample(with_dmg, n_dmg) if n_dmg else []
    sample += rng.sample(no_dmg, n_neg) if n_neg else []

    preview_dir = output_dir / "previews"
    raw_mask_dir = preview_dir / "raw_masks"
    preview_dir.mkdir(parents=True, exist_ok=True)
    raw_mask_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(sample):
        img = np.array(Image.open(output_dir / t["image_path"]).convert("RGB"))
        msk = np.array(Image.open(output_dir / t["mask_path"]))
        overlay = img.copy()
        sel = msk > 0
        if sel.any():
            overlay[sel, 0] = np.clip(overlay[sel, 0].astype(int) + 120, 0, 255)
            overlay[sel, 1] = (overlay[sel, 1] * 0.4).astype(np.uint8)
            overlay[sel, 2] = (overlay[sel, 2] * 0.4).astype(np.uint8)
        side_by_side = np.concatenate([img, overlay], axis=1)
        ratio_str = f"{t['damage_ratio']*100:.2f}pct"
        Image.fromarray(side_by_side).save(preview_dir / f"overlay_{i:03d}_{ratio_str}.png")
        # also save the raw mask scaled to 0-255 for visual inspection
        Image.fromarray((msk * 255).astype(np.uint8)).save(raw_mask_dir / f"mask_{i:03d}_{ratio_str}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xbd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=TILE_SIZE_DEFAULT)
    parser.add_argument("--min-damage-ratio", type=float, default=0.005)
    parser.add_argument("--negative-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-first-n", type=int, default=0,
                        help="Print detailed polygon stats for first N images (0 = off)")
    args = parser.parse_args()

    images_dir = args.xbd_root / "images"
    labels_dir = args.xbd_root / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {labels_dir} to exist")

    out_images = args.output / "images"
    out_masks = args.output / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    image_files = sorted([p for p in images_dir.glob("*.png") if is_relevant(p.name)])
    if not image_files:
        raise SystemExit("No relevant post-disaster images found")

    images_by_disaster = defaultdict(list)
    for p in image_files:
        images_by_disaster[disaster_from_filename(p.name)].append(p)

    image_split_map = split_per_image(images_by_disaster, seed=args.seed)

    print(f"Found {len(image_files)} post images across {len(images_by_disaster)} disasters")
    for d, files in sorted(images_by_disaster.items()):
        counts = defaultdict(int)
        for f in files:
            counts[image_split_map[f]] += 1
        print(f"  {d:30s}  total={len(files):4d}  "
              f"train={counts['train']:4d}  val={counts['val']:4d}  test={counts['test']:4d}")

    tiles_meta = []
    stats = defaultdict(lambda: defaultdict(int))
    polygon_stats = {"images_with_polys": 0, "images_with_damaged": 0,
                     "total_polys": 0, "total_damaged": 0,
                     "subtypes": defaultdict(int)}

    for idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
        disaster = disaster_from_filename(img_path.name)
        split = image_split_map[img_path]
        label_path = labels_dir / (img_path.stem + ".json")
        if not label_path.exists():
            continue
        with open(label_path) as f:
            label_json = json.load(f)
        polys = parse_polygons(label_json)

        n_total = len(polys)
        n_damaged = sum(1 for _, dmg, _ in polys if dmg)
        polygon_stats["total_polys"] += n_total
        polygon_stats["total_damaged"] += n_damaged
        if n_total > 0:
            polygon_stats["images_with_polys"] += 1
        if n_damaged > 0:
            polygon_stats["images_with_damaged"] += 1
        for _, _, st in polys:
            polygon_stats["subtypes"][st] += 1

        image = np.array(Image.open(img_path).convert("RGB"))
        h, w = image.shape[:2]
        mask = rasterize_mask(polys, h, w)

        if idx < args.debug_first_n:
            print(f"\n[DEBUG] {img_path.name}: {n_total} polys ({n_damaged} damaged), "
                  f"image={image.shape}, mask sum={int(mask.sum())}, "
                  f"max ratio={float(mask.mean()):.4f}")

        for r, c, img_tile, msk_tile in tile_image_and_mask(image, mask, args.tile_size):
            ratio = float(msk_tile.mean())
            keep = ratio >= args.min_damage_ratio or (ratio == 0.0 and rng.random() < args.negative_ratio)
            if not keep:
                continue

            tile_id = f"{img_path.stem}_r{r}_c{c}"
            img_rel = f"images/{tile_id}.png"
            msk_rel = f"masks/{tile_id}.png"
            Image.fromarray(img_tile).save(args.output / img_rel, optimize=True)
            Image.fromarray(msk_tile).save(args.output / msk_rel, optimize=True)

            tiles_meta.append({
                "tile_id": tile_id,
                "image_path": img_rel,
                "mask_path": msk_rel,
                "disaster": disaster,
                "split": split,
                "row": r,
                "col": c,
                "damage_ratio": ratio,
            })
            stats[split][disaster] += 1

    manifest = {
        "tile_size": args.tile_size,
        "min_damage_ratio": args.min_damage_ratio,
        "negative_ratio": args.negative_ratio,
        "seed": args.seed,
        "tiles": tiles_meta,
    }
    with open(args.output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Polygon stats ===")
    print(f"  Images with any polygons:     {polygon_stats['images_with_polys']}")
    print(f"  Images with damaged polygons: {polygon_stats['images_with_damaged']}")
    print(f"  Total polygons:               {polygon_stats['total_polys']}")
    print(f"  Total damaged polygons:       {polygon_stats['total_damaged']}")
    print("  Subtype distribution:")
    for st, n in sorted(polygon_stats["subtypes"].items(), key=lambda x: -x[1]):
        print(f"    {st:20s} {n}")

    print("\n=== Tiles per split / disaster ===")
    for split in ("train", "val", "test"):
        total = sum(stats[split].values())
        print(f"  {split}: {total}")
        for d, n in sorted(stats[split].items()):
            print(f"    {d:30s} {n}")

    print("\nGenerating previews...")
    generate_previews(manifest, args.output, n=60, seed=args.seed)
    print(f"\nDone. Output: {args.output}")
    print(f"  -> previews/overlay_*.png  : image | image+red_overlay")
    print(f"  -> previews/raw_masks/*.png: mask scaled to 0-255 (white = damaged)")


if __name__ == "__main__":
    main()
