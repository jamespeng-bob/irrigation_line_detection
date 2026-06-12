"""Tile-based PyTorch dataset for multi-class irrigation-line segmentation.

Inputs are huge architectural drawings (typically 6.8k–14.4k px wide) and
the model is a U-Net trained at 1024² tiles. The dataset reads polylines
once at ``__init__``, then opens each image lazily per ``__getitem__``,
crops a tile, and rasterizes the affected polylines into a tile-local
``(K, T, T)`` mask. We deliberately do **not** preload the full-resolution
images (one 14.4k × 10.8k RGB image is ~470 MB).

Modes
-----
``random``
    Pick a random image with annotations, then a random polyline (uniform
    across classes), then center a tile on a random point along that
    polyline (with jitter). Foreground pixels are guaranteed.
``random_class_balanced``
    Class-stratified sampler: pick a class uniformly first, then a polyline
    of that class uniformly across all images, then center+jitter the tile
    on a random point of that polyline. Effectively oversamples rare-class
    tiles so each class gets ~equal gradient signal per epoch — Phase-2
    intervention for the rare-class collapse observed in v1.
``grid``
    Deterministic sliding window over every image. Used for whole-image
    inference / per-image Dice.
``pos_only_grid``
    Same as ``grid`` but tiles with zero foreground in *any* class are
    skipped at ``__init__`` time. Used for fast validation feedback during
    training.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentation import TileAugmenter
from .coco_loader import ClassPolyline, ImageRecord, load_split

# Architectural drawings routinely exceed PIL's default decompression-bomb
# guard. We're loading our own trusted dataset, so disable it.
Image.MAX_IMAGE_PIXELS = None


@dataclass(frozen=True)
class TileSample:
    """One training/validation tile."""

    image: torch.Tensor          # [3, T, T], normalized float32
    mask: torch.Tensor           # [K, T, T], float32 in {0, 1}
    img_id: int
    tile_origin: tuple[int, int]  # (row, col) in original image pixels


class TileDataset(Dataset):
    """Multi-class binary segmentation tile dataset (one channel per class).

    Parameters
    ----------
    split_dir
        Directory containing ``_annotations.coco.json`` plus the image files.
    num_classes
        Output channel count ``K``. Must match the number of foreground
        categories in the loaded COCO file; we assert this.
    tile_size, stride
        Square tile side length, and sliding-window stride for grid modes.
    mode
        ``random`` | ``grid`` | ``pos_only_grid``.
    thickness
        Stroke thickness in pixels when rasterizing polylines into the mask.
    augmenter
        Optional :class:`TileAugmenter` applied to ``(image, mask)``.
    mean, std
        Per-channel ImageNet-like normalization stats applied to the image.
    samples_per_epoch_per_image
        Used only in ``random`` mode: epoch length =
        ``samples_per_epoch_per_image * num_images_with_polylines``.
    jitter_frac
        In ``random`` mode, jitter the tile center by up to
        ``jitter_frac * tile_size`` pixels in each direction so foreground
        isn't always at the exact tile center. Default ``0.25``.
    seed
        Optional seed for the ``random``-mode sampler.
    """

    def __init__(
        self,
        split_dir: Path | str,
        num_classes: int,
        tile_size: int = 1024,
        stride: int = 768,
        mode: str = "random",
        thickness: int = 4,
        augmenter: Optional[TileAugmenter] = None,
        mean: tuple[float, float, float] = (0.9548, 0.9548, 0.9548),
        std: tuple[float, float, float] = (0.1850, 0.1850, 0.1850),
        samples_per_epoch_per_image: int = 8,
        jitter_frac: float = 0.25,
        seed: Optional[int] = None,
        class_allowlist: Optional[list[str]] = None,
    ) -> None:
        if mode not in ("random", "random_class_balanced", "grid", "pos_only_grid"):
            raise ValueError(f"Unknown mode: {mode!r}")
        self.split_dir = Path(split_dir)
        self.num_classes = int(num_classes)
        self.tile_size = int(tile_size)
        self.stride = int(stride)
        self.mode = mode
        self.thickness = int(thickness)
        self.augmenter = augmenter
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.jitter_frac = float(jitter_frac)
        self.rng = random.Random(seed)

        # ── Load annotations ─────────────────────────────────────────────
        (
            self.images,
            self.polylines_by_image,
            self.class_names,
            self.cat_id_to_channel,
            self._load_stats,
        ) = load_split(self.split_dir, class_allowlist=class_allowlist)

        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"num_classes={self.num_classes} but {self.split_dir} contains "
                f"{len(self.class_names)} foreground classes (after allowlist): "
                f"{self.class_names}"
            )

        # ── Per-class polyline pools (built lazily by the class-balanced
        #    sampler; harmless to leave None for other modes).
        self._polys_by_class: Optional[list[list[tuple[int, ClassPolyline]]]] = None

        # ── Mode-dependent indexing ──────────────────────────────────────
        if mode in ("random", "random_class_balanced"):
            n_with = sum(1 for p in self.polylines_by_image.values() if p)
            self._length = max(1, samples_per_epoch_per_image * max(1, n_with))
            self._grid: Optional[list[tuple[int, int, int]]] = None
        else:
            self._grid = self._build_grid(skip_empty=(mode == "pos_only_grid"))
            self._length = len(self._grid)

    # ------------------------------------------------------------------
    # Standard Dataset API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> TileSample:
        if self.mode == "random":
            img_id, row, col = self._sample_random_tile()
        elif self.mode == "random_class_balanced":
            img_id, row, col = self._sample_random_class_balanced_tile()
        else:
            assert self._grid is not None
            img_id, row, col = self._grid[idx % len(self._grid)]

        image_np, mask_np = self._extract_tile(img_id, row, col)

        if self.augmenter is not None:
            image_np, mask_np = self.augmenter(image_np, mask_np)

        # Image: HWC uint8 → CHW float32, normalized.
        image_t = (image_np.astype(np.float32) / 255.0).transpose(2, 0, 1)
        image_t = (image_t - self.mean) / self.std

        # Mask: (K, T, T) uint8 in {0, 255} → float32 in {0, 1}.
        mask_t = mask_np.astype(np.float32) / 255.0

        return TileSample(
            image=torch.from_numpy(image_t.copy()),
            mask=torch.from_numpy(mask_t.copy()),
            img_id=int(img_id),
            tile_origin=(int(row), int(col)),
        )

    # ------------------------------------------------------------------
    # Grid construction (deterministic modes)
    # ------------------------------------------------------------------

    def _build_grid(self, skip_empty: bool) -> list[tuple[int, int, int]]:
        """Sliding-window origins for every image. ``skip_empty`` drops tiles
        whose union-of-classes mask is all zero."""
        grid: list[tuple[int, int, int]] = []

        # For pos_only_grid we need the full-image union mask once per image
        # so we can cheaply check which tiles contain foreground.
        full_union: dict[int, np.ndarray] = {}
        if skip_empty:
            for img_id, rec in self.images.items():
                full_union[img_id] = self._rasterize_full_union(
                    img_id, rec.height, rec.width
                )

        for img_id, rec in self.images.items():
            n_rows = max(
                1, math.ceil(max(rec.height - self.tile_size, 0) / self.stride) + 1
            )
            n_cols = max(
                1, math.ceil(max(rec.width - self.tile_size, 0) / self.stride) + 1
            )
            full = full_union.get(img_id) if skip_empty else None
            for tr in range(n_rows):
                for tc in range(n_cols):
                    row = min(tr * self.stride, max(0, rec.height - self.tile_size))
                    col = min(tc * self.stride, max(0, rec.width - self.tile_size))
                    if full is not None:
                        sub = full[row : row + self.tile_size, col : col + self.tile_size]
                        if not sub.any():
                            continue
                    grid.append((img_id, row, col))
        return grid

    def _rasterize_full_union(self, img_id: int, height: int, width: int) -> np.ndarray:
        """Full-image binary mask: pixel is 255 if ANY class polyline covers it."""
        mask = np.zeros((height, width), dtype=np.uint8)
        for pl in self.polylines_by_image.get(img_id, []):
            pts_i = pl.points.astype(np.int32).reshape(-1, 1, 2)
            if pts_i.shape[0] < 2:
                continue
            cv2.polylines(
                mask, [pts_i], isClosed=False, color=255,
                thickness=self.thickness, lineType=cv2.LINE_8,
            )
        return mask

    # ------------------------------------------------------------------
    # Random sampling (training)
    # ------------------------------------------------------------------

    def _sample_random_tile(self) -> tuple[int, int, int]:
        """Pick a tile centered (with jitter) on a random GT polyline point.

        Sampling is "uniform over polylines": pick an image with polylines
        uniformly, then a polyline uniformly within that image, then a point
        along that polyline uniformly. ``lateral_solid_0`` (the dominant
        class) is therefore *naturally* oversampled in proportion to its
        polyline-count majority. A class-stratified sampler is a separate
        Phase-2 lever.
        """
        ids_with_polys = [iid for iid, p in self.polylines_by_image.items() if p]
        if not ids_with_polys:
            # Fallback: any image, uniform origin.
            img_id = self.rng.choice(list(self.images.keys()))
            rec = self.images[img_id]
            row = self.rng.randint(0, max(0, rec.height - self.tile_size))
            col = self.rng.randint(0, max(0, rec.width - self.tile_size))
            return img_id, row, col

        img_id = self.rng.choice(ids_with_polys)
        rec = self.images[img_id]
        polys = self.polylines_by_image[img_id]
        pl = self.rng.choice(polys)
        return self._center_tile_on_polyline(img_id, rec, pl)

    def _build_polys_by_class(self) -> list[list[tuple[int, ClassPolyline]]]:
        """Per-class flat list of ``(image_id, ClassPolyline)`` pairs. Cached."""
        pools: list[list[tuple[int, ClassPolyline]]] = [
            [] for _ in range(self.num_classes)
        ]
        for img_id, polys in self.polylines_by_image.items():
            for pl in polys:
                pools[pl.class_idx].append((img_id, pl))
        return pools

    def _sample_random_class_balanced_tile(self) -> tuple[int, int, int]:
        """Class-stratified random sampler.

        Pick a class uniformly among classes that have ≥ 1 polyline anywhere
        in the dataset, then a ``(image, polyline)`` pair uniformly within
        that class, then center+jitter a tile on a random point of the
        polyline. This evens out per-class gradient signal regardless of
        how skewed the polyline-count distribution is.
        """
        if self._polys_by_class is None:
            self._polys_by_class = self._build_polys_by_class()

        non_empty = [k for k, pool in enumerate(self._polys_by_class) if pool]
        if not non_empty:
            # Should be unreachable: __init__ checks num_classes against
            # the loaded class set. Defensive fallback to uniform random.
            return self._sample_random_tile()

        k = self.rng.choice(non_empty)
        img_id, pl = self.rng.choice(self._polys_by_class[k])
        rec = self.images[img_id]
        return self._center_tile_on_polyline(img_id, rec, pl)

    def _center_tile_on_polyline(
        self,
        img_id: int,
        rec: ImageRecord,
        pl: ClassPolyline,
    ) -> tuple[int, int, int]:
        """Shared centering + jitter for both random samplers."""
        pt = pl.points[self.rng.randint(0, len(pl.points) - 1)]
        col_center = int(round(pt[0]))
        row_center = int(round(pt[1]))
        half = self.tile_size // 2
        jitter = int(self.jitter_frac * self.tile_size)
        rj = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
        cj = self.rng.randint(-jitter, jitter) if jitter > 0 else 0
        row = max(0, min(max(0, rec.height - self.tile_size), row_center - half + rj))
        col = max(0, min(max(0, rec.width - self.tile_size), col_center - half + cj))
        return img_id, row, col

    # ------------------------------------------------------------------
    # Tile extraction
    # ------------------------------------------------------------------

    def _extract_tile(self, img_id: int, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
        """Read a tile from disk (RGB) and rasterize the per-tile (K, T, T) mask."""
        rec = self.images[img_id]
        T = self.tile_size

        with Image.open(rec.path) as im:
            im = im.convert("RGB")
            # PIL crop uses (left, upper, right, lower) → (col0, row0, col1, row1).
            # Out-of-bounds regions are padded with zeros by PIL.
            crop = im.crop((col, row, col + T, row + T))
        image_np = np.array(crop, dtype=np.uint8)  # (T, T, 3)

        mask_np = self._rasterize_tile(img_id, row, col)  # (K, T, T)

        # Safety pad to (T, T) if PIL gave back a smaller crop (rare).
        if image_np.shape[:2] != (T, T):
            padded = np.zeros((T, T, 3), dtype=np.uint8)
            h, w = image_np.shape[:2]
            padded[:h, :w] = image_np
            image_np = padded
        if mask_np.shape != (self.num_classes, T, T):
            padded_m = np.zeros((self.num_classes, T, T), dtype=np.uint8)
            _, h, w = mask_np.shape
            padded_m[:, :h, :w] = mask_np
            mask_np = padded_m

        return image_np, mask_np

    def _rasterize_tile(self, img_id: int, row: int, col: int) -> np.ndarray:
        """Rasterize per-class polylines into a tile-local ``(K, T, T)`` mask.

        Polylines are translated by ``(-col, -row)`` so original-image
        coordinates become tile-local; cv2 clips against the canvas bounds.
        """
        T = self.tile_size
        K = self.num_classes
        masks = np.zeros((K, T, T), dtype=np.uint8)
        for pl in self.polylines_by_image.get(img_id, []):
            pts = pl.points.copy()
            pts[:, 0] -= col
            pts[:, 1] -= row
            pts_i = pts.astype(np.int32).reshape(-1, 1, 2)
            if pts_i.shape[0] < 2:
                continue
            cv2.polylines(
                masks[pl.class_idx], [pts_i], isClosed=False, color=255,
                thickness=self.thickness, lineType=cv2.LINE_8,
            )
        return masks


# ---------------------------------------------------------------------------
# DataLoader collation + worker initialization
# ---------------------------------------------------------------------------


def collate_tile_samples(samples: list[TileSample]) -> dict[str, torch.Tensor]:
    """Custom collate that stacks :class:`TileSample` instances into a dict."""
    return {
        "image": torch.stack([s.image for s in samples], dim=0),
        "mask": torch.stack([s.mask for s in samples], dim=0),
        "img_ids": torch.tensor([s.img_id for s in samples], dtype=torch.long),
        "tile_origins": torch.tensor([s.tile_origin for s in samples], dtype=torch.long),
    }


def worker_init_fn(worker_id: int) -> None:
    """Re-seed per-worker RNGs after fork (Linux) or spawn (macOS).

    On Linux ``DataLoader`` workers fork from the main process and inherit
    the full Python ``random``, NumPy, and our :class:`TileDataset`'s
    ``self.rng`` state. Without this hook every worker would produce the
    same sequence of "random" tiles, collapsing effective batch diversity.
    Each epoch torch rolls a fresh base seed, so this also re-seeds across
    epochs.
    """
    import random as _py_random

    import numpy as _np

    info = torch.utils.data.get_worker_info()
    base = torch.initial_seed() % (2**31)
    seed = (base + worker_id) % (2**31)
    _py_random.seed(seed)
    _np.random.seed(seed)
    if info is not None and hasattr(info.dataset, "rng"):
        info.dataset.rng = _py_random.Random(seed)
