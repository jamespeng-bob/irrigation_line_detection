"""COCO loader for the v6 polyline schema (poly-irrigation dataset).

Schema (Roboflow ``_annotations.coco.json``):

- ``categories``: ``[{id, name, supercategory}]``. The first non-background
  category id is 1; we use ``supercategory == "none"`` to recognize and skip
  the Roboflow placeholder if it's still present.
- ``images``:     ``[{id, file_name, height, width, ...}]``.
- ``annotations``: each entry has

      {
        "id":          int,
        "image_id":    int,
        "category_id": int,
        "bbox":        [x, y, w, h],          # always present (pixel coords)
        "iscrowd":     0/1,
        "polyline":    [[x1, y1, x2, y2, ...]],   # list of polylines
        "length":      float,                 # polyline length in px (Roboflow)
      }

Per Roboflow's convention each annotation usually carries exactly one polyline
(the inner list), but the outer list lets a single annotation describe several
disjoint pieces. We expand them so that the loader's output is a *flat* list
of :class:`ClassPolyline` objects regardless of how the source grouped them.

Annotations that lack a usable polyline (no ``polyline`` field, or all inner
lists have < 2 points) are skipped with a debug counter — see
:func:`load_split`'s return ``stats`` dict.

This loader is deliberately self-contained and avoids ``pycocotools``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ImageRecord:
    """Metadata for a single image in a split."""

    id: int
    file_name: str
    path: Path
    height: int
    width: int


@dataclass(frozen=True)
class ClassPolyline:
    """One polyline tagged with its train-time class index.

    ``points`` are absolute (x_col, y_row) pixel coordinates at the image's
    native resolution. ``class_idx`` is in ``[0, K)`` — the channel index this
    polyline maps to in the rasterized ``(K, H, W)`` mask.
    """

    image_id: int
    class_idx: int
    points: np.ndarray  # shape (N, 2), float64


def _is_background_category(cat: dict) -> bool:
    """Detect Roboflow's ``supercategory: none`` placeholder."""
    return cat.get("supercategory") == "none"


def build_class_mapping(
    raw_categories: list[dict],
    class_allowlist: list[str] | None = None,
) -> tuple[list[str], dict[int, int]]:
    """Return ``(class_names, cat_id_to_channel)`` from the COCO categories list.

    Background categories (``supercategory == "none"``) are always skipped.

    ``class_allowlist`` controls which foreground classes are kept and in
    what channel order:

    - ``None`` (default): keep every real category, in the order they appear
      in the COCO file. The remap script writes them in descending-train-
      frequency order, so channel 0 == most common.
    - ``list[str]``: keep only those classes, and assign channels in the
      order they appear in the allowlist. Other classes are dropped from
      ``cat_id_to_channel``, which means their annotations get skipped by
      ``load_split`` downstream. Raises ``ValueError`` if any allowlist name
      isn't a real category in the COCO file (typo guard).

    The returned mapping ``cat_id_to_channel`` lets us translate a raw COCO
    ``category_id`` into the ``[0, K)`` channel index used everywhere
    downstream (model output, masks, metrics).
    """
    real_categories = [c for c in raw_categories if not _is_background_category(c)]
    name_to_cat_id = {str(c["name"]): int(c["id"]) for c in real_categories}

    if class_allowlist is None:
        ordered_names = [str(c["name"]) for c in real_categories]
    else:
        unknown = [n for n in class_allowlist if n not in name_to_cat_id]
        if unknown:
            raise ValueError(
                f"class_allowlist contains classes not in this COCO: {unknown}. "
                f"Available real classes: {list(name_to_cat_id)}"
            )
        ordered_names = list(class_allowlist)

    class_names: list[str] = []
    cat_id_to_channel: dict[int, int] = {}
    for n in ordered_names:
        ch = len(class_names)
        class_names.append(n)
        cat_id_to_channel[name_to_cat_id[n]] = ch
    return class_names, cat_id_to_channel


def _expand_polyline_field(poly_field) -> list[np.ndarray]:
    """Convert Roboflow's ``polyline`` value into a list of ``(N, 2)`` arrays.

    Accepts both shapes Roboflow has historically used:
    - ``[[x1, y1, x2, y2, ...]]``  — nested list of flat coord lists.
    - ``[x1, y1, x2, y2, ...]``    — a single flat coord list (no outer wrap).

    Polylines with fewer than 2 points are dropped.
    """
    if not poly_field:
        return []
    # Detect nested vs flat by inspecting the first element.
    first = poly_field[0]
    if isinstance(first, (list, tuple)):
        groups = poly_field
    else:
        groups = [poly_field]

    out: list[np.ndarray] = []
    for grp in groups:
        if grp is None or len(grp) < 4 or len(grp) % 2 != 0:
            continue
        arr = np.asarray(grp, dtype=np.float64).reshape(-1, 2)
        if arr.shape[0] < 2:
            continue
        out.append(arr)
    return out


def load_split(
    split_dir: Path | str,
    class_allowlist: list[str] | None = None,
) -> tuple[
    dict[int, ImageRecord],
    dict[int, list[ClassPolyline]],
    list[str],
    dict[int, int],
    dict[str, int],
]:
    """Parse a v6 polyline COCO split.

    Parameters
    ----------
    split_dir
        Directory containing ``_annotations.coco.json`` and the image files.
    class_allowlist
        Optional list of class names to keep. When provided, only annotations
        of those classes are returned, and the channel order in ``class_names``
        follows the allowlist (not the COCO file). Use this to spin up a
        specialist model that targets a subset of classes from the same
        underlying dataset.

    Returns
    -------
    images
        ``image_id -> ImageRecord``.
    polylines_by_image
        ``image_id -> list[ClassPolyline]`` (possibly empty).
    class_names
        Ordered list of foreground class names, length ``K``. The channel
        index of a class is its position in this list.
    cat_id_to_channel
        Raw COCO ``category_id`` → ``[0, K)`` channel index. When an
        allowlist is provided, only allowlisted category ids appear here.
    stats
        Diagnostic counters useful for sanity-checking the load:
        ``n_images``, ``n_annotations``, ``n_polylines``, ``n_skipped_bbox_only``,
        ``n_skipped_unknown_cat`` (which now includes allowlist-filtered
        annotations).
    """
    split_dir = Path(split_dir)
    ann_path = split_dir / "_annotations.coco.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {ann_path}")
    with ann_path.open() as f:
        raw = json.load(f)

    class_names, cat_id_to_channel = build_class_mapping(
        raw.get("categories", []), class_allowlist=class_allowlist
    )

    images: dict[int, ImageRecord] = {}
    for img in raw.get("images", []):
        images[int(img["id"])] = ImageRecord(
            id=int(img["id"]),
            file_name=str(img["file_name"]),
            path=split_dir / img["file_name"],
            height=int(img["height"]),
            width=int(img["width"]),
        )

    polylines_by_image: dict[int, list[ClassPolyline]] = {iid: [] for iid in images}
    stats = {
        "n_images": len(images),
        "n_annotations": 0,
        "n_polylines": 0,
        "n_skipped_bbox_only": 0,
        "n_skipped_unknown_cat": 0,
    }

    for ann in raw.get("annotations", []):
        stats["n_annotations"] += 1
        cat_id = int(ann.get("category_id", -1))
        if cat_id not in cat_id_to_channel:
            stats["n_skipped_unknown_cat"] += 1
            continue
        ch = cat_id_to_channel[cat_id]
        polys = _expand_polyline_field(ann.get("polyline"))
        if not polys:
            stats["n_skipped_bbox_only"] += 1
            continue
        img_id = int(ann["image_id"])
        bucket = polylines_by_image.setdefault(img_id, [])
        for pts in polys:
            bucket.append(
                ClassPolyline(image_id=img_id, class_idx=ch, points=pts)
            )
            stats["n_polylines"] += 1

    return images, polylines_by_image, class_names, cat_id_to_channel, stats


def summarize_polylines(
    polylines_by_image: dict[int, list[ClassPolyline]],
    class_names: list[str],
) -> dict:
    """Per-class polyline-count summary, useful for logging."""
    counts = [0] * len(class_names)
    images_with_class = [set() for _ in class_names]
    for img_id, polys in polylines_by_image.items():
        for pl in polys:
            counts[pl.class_idx] += 1
            images_with_class[pl.class_idx].add(img_id)
    return {
        "per_class_polylines": {
            class_names[k]: counts[k] for k in range(len(class_names))
        },
        "per_class_image_coverage": {
            class_names[k]: len(images_with_class[k]) for k in range(len(class_names))
        },
        "n_images_any_polyline": sum(
            1 for polys in polylines_by_image.values() if polys
        ),
    }
