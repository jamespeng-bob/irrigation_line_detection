"""Exploratory analysis for the poly-irrigation dataset.

Supports both export formats:

- **YOLO** (e.g. ``poly-irrigation.v6-v2_w_boboflow.yolo26``): a ``data.yaml`` at
  the root, plus ``train/{images,labels}`` and ``valid/{images,labels}``.
- **COCO** (e.g. ``poly-irrigation.v6-v2_w_boboflow.coco``): each split is a
  flat folder of images plus an ``_annotations.coco.json`` file.

The script is path-agnostic: pass ``--root`` so it can run identically on the
MacBook (local) and on the RTX 6000 server (remote).

Local:
    python scripts/explore_dataset.py \\
        --root /Users/james.peng/Desktop/Irrigation/datasets/poly-irrigation.v6-v2_w_boboflow.coco

Server:
    python scripts/explore_dataset.py \\
        --root /home/rtx6000/james/datasets/poly-irrigation.v6-v2_w_boboflow.coco

The script writes ``DATASET_REPORT.md``, ``summary.json``, and PNG figures to
``--out-dir`` (default ``reports/dataset/``).
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

# Architectural drawings in this dataset routinely exceed 10k×10k pixels;
# raise PIL's decompression-bomb ceiling so the loader doesn't refuse them.
Image.MAX_IMAGE_PIXELS = None


SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Dataset root. Either a YOLO root (with data.yaml) or a COCO root "
             "(with split sub-folders containing _annotations.coco.json).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/dataset"),
        help="Where to write the report + figures.",
    )
    p.add_argument(
        "--num-samples",
        type=int,
        default=6,
        help="Number of annotated samples to render in the visualization grid.",
    )
    p.add_argument(
        "--format",
        choices=["auto", "yolo", "coco"],
        default="auto",
        help="Override format detection. 'auto' looks for data.yaml (YOLO) "
             "or _annotations.coco.json (COCO).",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(root: Path) -> str:
    if (root / "data.yaml").exists():
        return "yolo"
    for s in SPLITS:
        if (root / s / "_annotations.coco.json").exists():
            return "coco"
    raise SystemExit(
        f"Could not auto-detect dataset format under {root}. "
        "Expected either data.yaml (YOLO) or */_annotations.coco.json (COCO)."
    )


# ===========================================================================
# YOLO support
# ===========================================================================

def load_data_yaml(root: Path) -> dict:
    with (root / "data.yaml").open() as f:
        return yaml.safe_load(f)


def discover_yolo_split(root: Path, split: str) -> Tuple[List[Path], List[Path]]:
    img_dir = root / split / "images"
    lbl_dir = root / split / "labels"
    if not img_dir.exists():
        return [], []
    images = sorted(
        p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    labels = [lbl_dir / (p.stem + ".txt") for p in images]
    return images, labels


def parse_yolo_label_file(path: Path) -> List[Tuple[int, List[float]]]:
    if not path.exists():
        return []
    out: List[Tuple[int, List[float]]] = []
    with path.open() as f:
        for raw in f:
            tokens = raw.strip().split()
            if not tokens:
                continue
            cls = int(tokens[0])
            coords = [float(t) for t in tokens[1:]]
            out.append((cls, coords))
    return out


def classify_yolo_line(coords: List[float]) -> str:
    """4 coords -> bbox; >= 6 coords (even count) -> polygon."""
    n = len(coords)
    if n == 4:
        return "bbox"
    if n >= 6 and n % 2 == 0:
        return "polygon"
    return "other"


def summarize_yolo_split(
    root: Path,
    split: str,
    class_names: List[str],
) -> dict:
    images, labels = discover_yolo_split(root, split)
    n_imgs = len(images)
    n_lbl_files = sum(1 for p in labels if p.exists())
    n_empty = sum(1 for p in labels if p.exists() and p.stat().st_size == 0)
    n_missing = sum(1 for p in labels if not p.exists())
    n_nonempty = n_lbl_files - n_empty

    class_counts: Counter = Counter()
    fmt_counts: Counter = Counter()
    per_image_inst: List[int] = []
    annotated_image_paths: List[Path] = []

    for img_path, lbl_path in zip(images, labels):
        anns = parse_yolo_label_file(lbl_path)
        per_image_inst.append(len(anns))
        if anns:
            annotated_image_paths.append(img_path)
        for cls, coords in anns:
            if 0 <= cls < len(class_names):
                class_counts[class_names[cls]] += 1
            else:
                class_counts[f"<oob:{cls}>"] += 1
            fmt_counts[classify_yolo_line(coords)] += 1

    image_sizes: List[Tuple[int, int]] = []
    for p in images[: min(len(images), 60)]:
        try:
            with Image.open(p) as im:
                image_sizes.append(im.size)
        except Exception:
            pass
    widths = [w for (w, _) in image_sizes]
    heights = [h for (_, h) in image_sizes]

    return {
        "split": split,
        "format": "yolo",
        "num_images": n_imgs,
        "num_label_files": n_lbl_files,
        "num_label_files_missing": n_missing,
        "num_label_files_empty": n_empty,
        "num_label_files_nonempty": n_nonempty,
        "num_annotated_images": len(annotated_image_paths),
        "num_instances_total": int(sum(per_image_inst)),
        "instances_per_image_mean": float(np.mean(per_image_inst)) if per_image_inst else 0.0,
        "instances_per_image_max": int(max(per_image_inst)) if per_image_inst else 0,
        "class_counts": dict(class_counts),
        "format_counts": dict(fmt_counts),
        "image_size_sample_n": len(image_sizes),
        "image_width_min": int(min(widths)) if widths else None,
        "image_width_max": int(max(widths)) if widths else None,
        "image_height_min": int(min(heights)) if heights else None,
        "image_height_max": int(max(heights)) if heights else None,
        "annotated_image_paths": [str(p) for p in annotated_image_paths],
    }


# ===========================================================================
# COCO support
# ===========================================================================

def discover_coco_split(root: Path, split: str) -> Optional[dict]:
    """Return the parsed COCO JSON for ``split`` if it exists, else ``None``."""
    p = root / split / "_annotations.coco.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _polyline_length(coords: List[float]) -> float:
    if len(coords) < 4:
        return 0.0
    xs = np.asarray(coords[0::2], dtype=float)
    ys = np.asarray(coords[1::2], dtype=float)
    dx = np.diff(xs)
    dy = np.diff(ys)
    return float(np.sqrt(dx * dx + dy * dy).sum())


def _polygon_area(coords: List[float]) -> float:
    if len(coords) < 6:
        return 0.0
    xs = np.asarray(coords[0::2], dtype=float)
    ys = np.asarray(coords[1::2], dtype=float)
    return float(0.5 * np.abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))


def _has_polyline(ann: dict) -> bool:
    pl = ann.get("polyline")
    return bool(pl) and any(len(p) >= 4 for p in pl)


def _has_polygon(ann: dict) -> bool:
    seg = ann.get("segmentation")
    if not seg:
        return False
    if isinstance(seg, dict):
        return True
    return bool(seg) and any(len(p) >= 6 for p in seg)


def classify_coco_ann(ann: dict) -> str:
    if _has_polyline(ann):
        return "polyline"
    if _has_polygon(ann):
        return "polygon"
    if ann.get("bbox"):
        return "bbox_only"
    return "other"


def _is_background_category(cat: dict) -> bool:
    """Roboflow inserts a 'supercategory: none' placeholder we should ignore."""
    return cat.get("supercategory") == "none"


def summarize_coco_split(root: Path, split: str) -> Optional[dict]:
    doc = discover_coco_split(root, split)
    if doc is None:
        return None

    cats: List[dict] = doc["categories"]
    id_to_name: Dict[int, str] = {c["id"]: c["name"] for c in cats}
    bg_ids = {c["id"] for c in cats if _is_background_category(c)}

    images: List[dict] = doc["images"]
    anns: List[dict] = doc["annotations"]

    class_counts: Counter = Counter()
    fmt_counts: Counter = Counter()
    per_image_inst: Counter = Counter()
    polyline_lengths_px: List[float] = []
    bbox_widths: List[float] = []
    bbox_heights: List[float] = []
    bbox_areas_norm: List[float] = []

    image_by_id = {im["id"]: im for im in images}

    for a in anns:
        cid = a.get("category_id")
        if cid in bg_ids:
            continue
        name = id_to_name.get(cid, f"<unk:{cid}>")
        class_counts[name] += 1
        fmt_counts[classify_coco_ann(a)] += 1
        per_image_inst[a["image_id"]] += 1

        if _has_polyline(a):
            for pl in a["polyline"]:
                polyline_lengths_px.append(_polyline_length(pl))

        bb = a.get("bbox")
        im = image_by_id.get(a["image_id"])
        if bb and im and im.get("width") and im.get("height"):
            _, _, w, h = bb
            bbox_widths.append(float(w))
            bbox_heights.append(float(h))
            bbox_areas_norm.append(
                (float(w) * float(h)) / (float(im["width"]) * float(im["height"]))
            )

    annotated_ids = set(per_image_inst.keys())
    n_annotated = len(annotated_ids)

    widths = [im["width"] for im in images if im.get("width")]
    heights = [im["height"] for im in images if im.get("height")]

    image_dir = root / split

    annotated_image_paths: List[Path] = []
    for im in images:
        if im["id"] in annotated_ids:
            p = image_dir / im["file_name"]
            if p.exists():
                annotated_image_paths.append(p)

    def _stats(arr: List[float]) -> Optional[dict]:
        if not arr:
            return None
        a = np.asarray(arr, dtype=float)
        return {
            "n": int(a.size),
            "min": float(a.min()),
            "p50": float(np.percentile(a, 50)),
            "mean": float(a.mean()),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
        }

    return {
        "split": split,
        "format": "coco",
        "num_images": len(images),
        "num_annotations": len(anns),
        "num_annotated_images": n_annotated,
        "num_instances_total": int(sum(per_image_inst.values())),
        "instances_per_image_mean": float(np.mean(list(per_image_inst.values()))) if per_image_inst else 0.0,
        "instances_per_image_max": int(max(per_image_inst.values())) if per_image_inst else 0,
        "class_counts": dict(class_counts),
        "format_counts": dict(fmt_counts),
        "image_width_min": int(min(widths)) if widths else None,
        "image_width_max": int(max(widths)) if widths else None,
        "image_height_min": int(min(heights)) if heights else None,
        "image_height_max": int(max(heights)) if heights else None,
        "polyline_length_px": _stats(polyline_lengths_px),
        "bbox_width_px": _stats(bbox_widths),
        "bbox_height_px": _stats(bbox_heights),
        "bbox_area_norm": _stats(bbox_areas_norm),
        "annotated_image_paths": [str(p) for p in annotated_image_paths],
        "_coco_doc": doc,
    }


# ---------------------------------------------------------------------------
# Plotting (shared)
# ---------------------------------------------------------------------------

def plot_class_distribution(
    summaries: Dict[str, dict],
    class_names: List[str],
    out_path: Path,
) -> None:
    splits_present = [s for s in SPLITS if s in summaries]
    counts_by_split = {
        s: [summaries[s]["class_counts"].get(c, 0) for c in class_names]
        for s in splits_present
    }
    x = np.arange(len(class_names))
    width = 0.8 / max(len(splits_present), 1)

    fig, ax = plt.subplots(figsize=(max(10, 0.55 * len(class_names)), 5))
    for i, s in enumerate(splits_present):
        ax.bar(x + i * width, counts_by_split[s], width, label=s)
    ax.set_xticks(x + width * (len(splits_present) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel("# instances")
    ax.set_title("Instances per class, per split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_bbox_size_histogram(summaries: Dict[str, dict], out_path: Path) -> None:
    """Plot histograms of bbox widths/heights and a scatter of (w,h)."""
    train = summaries.get("train")
    if not train or train.get("bbox_width_px") is None:
        return
    doc = train.get("_coco_doc")
    if not doc:
        return

    widths_px: List[float] = []
    heights_px: List[float] = []
    img_by_id = {im["id"]: im for im in doc["images"]}
    cats = {c["id"]: c["name"] for c in doc["categories"]}
    bg_ids = {c["id"] for c in doc["categories"] if _is_background_category(c)}
    for a in doc["annotations"]:
        if a.get("category_id") in bg_ids:
            continue
        bb = a.get("bbox")
        if not bb:
            continue
        widths_px.append(float(bb[2]))
        heights_px.append(float(bb[3]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(np.log10(np.clip(widths_px, 1, None)), bins=40)
    axes[0].set_title("bbox width (px), log10")
    axes[0].set_xlabel("log10(width [px])")

    axes[1].hist(np.log10(np.clip(heights_px, 1, None)), bins=40)
    axes[1].set_title("bbox height (px), log10")
    axes[1].set_xlabel("log10(height [px])")

    axes[2].scatter(widths_px, heights_px, s=2, alpha=0.3)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("width [px]")
    axes[2].set_ylabel("height [px]")
    axes[2].set_title("bbox (w, h) scatter (train)")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle("Bounding-box size distribution (train, COCO)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def render_sample_grid_yolo(
    summary: dict,
    class_names: List[str],
    out_path: Path,
    num_samples: int,
    rng: random.Random,
) -> None:
    annotated = summary["annotated_image_paths"]
    if not annotated:
        return
    chosen = rng.sample(annotated, k=min(num_samples, len(annotated)))

    cmap = plt.get_cmap("tab20")
    cols = min(3, len(chosen))
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    used_classes: set = set()

    for ax_i, img_path_str in enumerate(chosen):
        ax = axes[ax_i // cols][ax_i % cols]
        img_path = Path(img_path_str)
        lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
        with Image.open(img_path) as im:
            W, H = im.size
            ax.imshow(im)
        for cls, coords in parse_yolo_label_file(lbl_path):
            fmt = classify_yolo_line(coords)
            color = cmap(cls % cmap.N)
            used_classes.add(cls)
            if fmt == "bbox":
                cx, cy, w, h = coords
                x0 = (cx - w / 2) * W
                y0 = (cy - h / 2) * H
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), w * W, h * H,
                    fill=False, edgecolor=color, linewidth=1.5,
                ))
            elif fmt == "polygon":
                xs = np.array(coords[0::2]) * W
                ys = np.array(coords[1::2]) * H
                ax.add_patch(mpatches.Polygon(
                    np.column_stack([xs, ys]),
                    closed=True, fill=False,
                    edgecolor=color, linewidth=1.5,
                ))
        ax.set_title(img_path.name, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for ax_i in range(len(chosen), rows * cols):
        axes[ax_i // cols][ax_i % cols].axis("off")

    if used_classes:
        legend_handles = [
            mpatches.Patch(color=cmap(c % cmap.N), label=class_names[c])
            for c in sorted(used_classes)
            if 0 <= c < len(class_names)
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(6, len(legend_handles)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sample annotated images (train)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_sample_grid_coco(
    root: Path,
    summary: dict,
    out_path: Path,
    num_samples: int,
    rng: random.Random,
) -> None:
    doc = summary.get("_coco_doc")
    if not doc:
        return
    annotated_paths = summary["annotated_image_paths"]
    if not annotated_paths:
        return

    chosen_paths = rng.sample(annotated_paths, k=min(num_samples, len(annotated_paths)))
    chosen_names = {Path(p).name for p in chosen_paths}

    file_to_id = {im["file_name"]: im["id"] for im in doc["images"]}
    chosen_image_ids = {file_to_id[n] for n in chosen_names if n in file_to_id}

    cats = {c["id"]: c["name"] for c in doc["categories"]}
    bg_ids = {c["id"] for c in doc["categories"] if _is_background_category(c)}
    real_cat_ids = sorted(set(cats) - bg_ids)
    color_index = {cid: i for i, cid in enumerate(real_cat_ids)}
    cmap = plt.get_cmap("tab20")
    color_for_cid = lambda cid: cmap(color_index.get(cid, 0) % cmap.N)

    anns_by_img: Dict[int, List[dict]] = {}
    for a in doc["annotations"]:
        if a.get("category_id") in bg_ids:
            continue
        if a["image_id"] in chosen_image_ids:
            anns_by_img.setdefault(a["image_id"], []).append(a)

    cols = min(3, len(chosen_paths))
    rows = (len(chosen_paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    used_cids: set = set()

    for ax_i, img_path_str in enumerate(chosen_paths):
        ax = axes[ax_i // cols][ax_i % cols]
        img_path = Path(img_path_str)
        with Image.open(img_path) as im:
            ax.imshow(im)
        img_id = file_to_id.get(img_path.name)
        for a in anns_by_img.get(img_id, []):
            cid = a["category_id"]
            color = color_for_cid(cid)
            used_cids.add(cid)
            drew_geometry = False
            if _has_polyline(a):
                for pl in a["polyline"]:
                    xs = np.array(pl[0::2], dtype=float)
                    ys = np.array(pl[1::2], dtype=float)
                    ax.plot(xs, ys, color=color, linewidth=1.2)
                    drew_geometry = True
            elif _has_polygon(a):
                seg = a["segmentation"]
                if isinstance(seg, list):
                    for poly in seg:
                        xs = np.array(poly[0::2], dtype=float)
                        ys = np.array(poly[1::2], dtype=float)
                        ax.add_patch(mpatches.Polygon(
                            np.column_stack([xs, ys]), closed=True,
                            fill=False, edgecolor=color, linewidth=1.2,
                        ))
                        drew_geometry = True
            if not drew_geometry and a.get("bbox"):
                x, y, w, h = a["bbox"]
                ax.add_patch(mpatches.Rectangle(
                    (x, y), w, h, fill=False, edgecolor=color, linewidth=1.0,
                ))
        ax.set_title(img_path.name, fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])

    for ax_i in range(len(chosen_paths), rows * cols):
        axes[ax_i // cols][ax_i % cols].axis("off")

    if used_cids:
        legend_handles = [
            mpatches.Patch(color=color_for_cid(cid), label=cats[cid])
            for cid in sorted(used_cids)
        ]
        fig.legend(handles=legend_handles, loc="lower center",
                   ncol=min(6, len(legend_handles)), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Sample annotated images (train) — polylines + boxes", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------

def _fmt_stats(s: Optional[dict], precision: int = 1) -> str:
    if not s:
        return "—"
    f = f".{precision}f"
    return (f"n={s['n']:,}, min={s['min']:{f}}, p50={s['p50']:{f}}, "
            f"mean={s['mean']:{f}}, p95={s['p95']:{f}}, max={s['max']:{f}}")


def write_markdown_report(
    root: Path,
    fmt: str,
    class_names: List[str],
    summaries: Dict[str, dict],
    figs: Dict[str, Path],
    out_path: Path,
) -> None:
    lines: List[str] = []
    lines.append(f"# Dataset report — `{root.name}`\n")
    lines.append(f"- Dataset root (local): `{root}`")
    lines.append(f"- Detected format: **{fmt.upper()}**")
    lines.append(f"- Number of (real) classes: **{len(class_names)}**")
    lines.append("")

    lines.append("## Class names\n")
    for i, name in enumerate(class_names):
        lines.append(f"- `{i}` — {name}")
    lines.append("")

    # --- split summary ----------------------------------------------------
    lines.append("## Split summary\n")
    if fmt == "yolo":
        lines.append(
            "| split | images | label files (exist / empty / non-empty) | "
            "annotated images | total instances | mean / max per annotated image |"
        )
        lines.append("| --- | ---: | --- | ---: | ---: | --- |")
        for s in SPLITS:
            d = summaries.get(s)
            if not d:
                continue
            ann = d["num_annotated_images"]
            mean_inst = (d["num_instances_total"] / ann) if ann else 0.0
            lines.append(
                f"| {s} | {d['num_images']} | "
                f"{d['num_label_files']} / {d['num_label_files_empty']} / {d['num_label_files_nonempty']} | "
                f"{ann} | {d['num_instances_total']} | "
                f"{mean_inst:.2f} / {d['instances_per_image_max']} |"
            )
    else:  # coco
        lines.append(
            "| split | images | annotations | annotated images "
            "| mean / max instances per annotated image |"
        )
        lines.append("| --- | ---: | ---: | ---: | --- |")
        for s in SPLITS:
            d = summaries.get(s)
            if not d:
                continue
            ann = d["num_annotated_images"]
            mean_inst = (d["num_instances_total"] / ann) if ann else 0.0
            lines.append(
                f"| {s} | {d['num_images']} | {d['num_annotations']} | "
                f"{ann} | {mean_inst:.2f} / {d['instances_per_image_max']} |"
            )
    lines.append("")

    # --- image dims -------------------------------------------------------
    lines.append("## Image dimensions\n")
    for s in SPLITS:
        d = summaries.get(s)
        if not d:
            continue
        wmin, wmax = d.get("image_width_min"), d.get("image_width_max")
        hmin, hmax = d.get("image_height_min"), d.get("image_height_max")
        if wmin is None:
            continue
        n = d.get("num_images") if fmt == "coco" else d.get("image_size_sample_n")
        lines.append(
            f"- **{s}** ({n} images): "
            f"W ∈ [{wmin}, {wmax}], H ∈ [{hmin}, {hmax}]"
        )
    lines.append("")

    # --- annotation geometry breakdown -----------------------------------
    if fmt == "yolo":
        lines.append("## Annotation geometry (YOLO)\n")
        lines.append("| split | bbox lines | polygon lines | other lines |")
        lines.append("| --- | ---: | ---: | ---: |")
        for s in SPLITS:
            d = summaries.get(s)
            if not d:
                continue
            f = d["format_counts"]
            lines.append(
                f"| {s} | {f.get('bbox', 0)} | {f.get('polygon', 0)} | {f.get('other', 0)} |"
            )
        lines.append("")
    else:
        lines.append("## Annotation geometry (COCO)\n")
        lines.append("| split | polyline | polygon | bbox-only | other |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for s in SPLITS:
            d = summaries.get(s)
            if not d:
                continue
            f = d["format_counts"]
            lines.append(
                f"| {s} | {f.get('polyline', 0)} | {f.get('polygon', 0)} | "
                f"{f.get('bbox_only', 0)} | {f.get('other', 0)} |"
            )
        lines.append("")

    # --- size stats (COCO only) -------------------------------------------
    if fmt == "coco":
        lines.append("## Annotation size statistics (train)\n")
        d = summaries.get("train")
        if d:
            lines.append(f"- Polyline length (px): {_fmt_stats(d.get('polyline_length_px'))}")
            lines.append(f"- BBox width (px):  {_fmt_stats(d.get('bbox_width_px'))}")
            lines.append(f"- BBox height (px): {_fmt_stats(d.get('bbox_height_px'))}")
            lines.append(f"- BBox area (fraction of image): {_fmt_stats(d.get('bbox_area_norm'), precision=5)}")
        lines.append("")

    # --- per-class counts -------------------------------------------------
    lines.append("## Per-class instance counts\n")
    lines.append("| class | " + " | ".join(SPLITS) + " | total |")
    lines.append("| --- | " + " | ".join(["---:"] * (len(SPLITS) + 1)) + " |")
    totals: Counter = Counter()
    for s in SPLITS:
        for c, n in summaries.get(s, {}).get("class_counts", {}).items():
            totals[c] += n
    for c in class_names:
        row = [c]
        for s in SPLITS:
            row.append(str(summaries.get(s, {}).get("class_counts", {}).get(c, 0)))
        row.append(str(totals.get(c, 0)))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # --- figures ----------------------------------------------------------
    if "class_distribution" in figs:
        lines.append(f"![class distribution]({figs['class_distribution'].name})\n")
    if "bbox_sizes" in figs:
        lines.append(f"![bbox size distribution]({figs['bbox_sizes'].name})\n")
    if "samples" in figs:
        lines.append(f"![sample annotated images]({figs['samples'].name})\n")

    # --- notes ------------------------------------------------------------
    lines.append("## Notes & observations\n")
    train = summaries.get("train", {})
    valid = summaries.get("valid", {})
    if train:
        ann_ratio = train["num_annotated_images"] / max(train["num_images"], 1)
        lines.append(
            f"- Train annotation coverage: **{train['num_annotated_images']}/"
            f"{train['num_images']}** images = {ann_ratio:.1%}."
        )
    if valid:
        ann_ratio_v = valid["num_annotated_images"] / max(valid["num_images"], 1)
        lines.append(
            f"- Valid annotation coverage: **{valid['num_annotated_images']}/"
            f"{valid['num_images']}** images = {ann_ratio_v:.1%}."
        )
    used = {c for s in summaries.values() for c in s.get("class_counts", {})}
    unused = [c for c in class_names if c not in used]
    if unused:
        lines.append(
            "- Classes with zero instances anywhere: "
            + ", ".join(f"`{c}`" for c in unused)
        )

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    root: Path = args.root.resolve()
    assert root.exists(), f"Dataset root not found: {root}"

    fmt = args.format if args.format != "auto" else detect_format(root)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: Dict[str, dict] = {}
    class_names: List[str] = []

    if fmt == "yolo":
        cfg = load_data_yaml(root)
        class_names = list(cfg.get("names", []))
        for s in SPLITS:
            if (root / s).exists():
                summaries[s] = summarize_yolo_split(root, s, class_names)
    elif fmt == "coco":
        train_doc = discover_coco_split(root, "train")
        if train_doc is None:
            raise SystemExit("COCO root must contain a train/_annotations.coco.json")
        class_names = [
            c["name"] for c in train_doc["categories"] if not _is_background_category(c)
        ]
        for s in SPLITS:
            sm = summarize_coco_split(root, s)
            if sm is not None:
                summaries[s] = sm
    else:
        raise SystemExit(f"Unknown format: {fmt}")

    # ---- write machine-readable summary ----------------------------------
    json_payload = {
        s: {k: v for k, v in d.items() if k not in {"annotated_image_paths", "_coco_doc"}}
        for s, d in summaries.items()
    }
    json_payload["_meta"] = {
        "format": fmt,
        "root": str(root),
        "class_names": class_names,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(json_payload, indent=2))

    # ---- figures ----------------------------------------------------------
    figs: Dict[str, Path] = {}

    cls_fig = args.out_dir / "class_distribution.png"
    plot_class_distribution(summaries, class_names, cls_fig)
    figs["class_distribution"] = cls_fig

    if fmt == "coco":
        bbox_fig = args.out_dir / "bbox_sizes.png"
        plot_bbox_size_histogram(summaries, bbox_fig)
        if bbox_fig.exists():
            figs["bbox_sizes"] = bbox_fig

    train = summaries.get("train")
    if train and train.get("annotated_image_paths"):
        sample_fig = args.out_dir / "samples_train.png"
        if fmt == "yolo":
            render_sample_grid_yolo(train, class_names, sample_fig,
                                    args.num_samples, rng)
        else:
            render_sample_grid_coco(root, train, sample_fig,
                                    args.num_samples, rng)
        if sample_fig.exists():
            figs["samples"] = sample_fig

    md_path = args.out_dir / "DATASET_REPORT.md"
    write_markdown_report(root, fmt, class_names, summaries, figs, md_path)

    print(f"Wrote: {md_path}")
    print(f"Wrote: {args.out_dir / 'summary.json'}")
    for k, p in figs.items():
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
