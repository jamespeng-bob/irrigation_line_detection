"""Compute per-channel mean / std on a random subsample of a split.

Why bother (vs. using ImageNet stats):
- Blueprint pages are essentially grayscale (R = G = B) and have notably
  higher contrast variance than natural images. lateral_detection measured
  (~0.20, ~0.40) on its v56 export and noted "using ImageNet stats would
  stretch post-norm input std to ~1.74 instead of ~1.0, which the encoder's
  first layers then have to absorb."
- The v6 export may differ in resolution / rendering, so recompute.

Usage:
    python scripts/compute_dataset_stats.py \
        --root ../datasets/poly-irrigation.v6-v2_w_boboflow.coco.merged \
        --split train --n-images 50
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path, help="Dataset root.")
    p.add_argument("--split", default="train", choices=("train", "valid", "test"))
    p.add_argument(
        "--n-images",
        type=int,
        default=50,
        help="Sample size. ``-1`` = all images in the split.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    split_dir = args.root / args.split
    if not split_dir.is_dir():
        raise SystemExit(f"Split dir not found: {split_dir}")

    paths = sorted(
        p for p in split_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not paths:
        raise SystemExit(f"No images found under {split_dir}")
    if args.n_images >= 0 and args.n_images < len(paths):
        paths = rng.sample(paths, args.n_images)

    sum_  = np.zeros(3, dtype=np.float64)
    sumsq = np.zeros(3, dtype=np.float64)
    n_px = 0
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0
        flat = arr.reshape(-1, 3)
        sum_  += flat.sum(axis=0)
        sumsq += (flat ** 2).sum(axis=0)
        n_px  += flat.shape[0]
        print(f"  {p.name}  {arr.shape[1]}x{arr.shape[0]}")
    mean = sum_ / n_px
    var = sumsq / n_px - mean ** 2
    std = np.sqrt(np.maximum(var, 0.0))

    print()
    print(f"# n_images = {len(paths)}, total pixels = {n_px:,}")
    print(f"normalization:")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
