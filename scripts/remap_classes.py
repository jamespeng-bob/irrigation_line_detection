"""Apply class merge / drop rules to a Roboflow-style COCO dataset.

Input (read-only):
    {src}/train/_annotations.coco.json   (+ image files)
    {src}/valid/_annotations.coco.json   (+ image files)
    {src}/test/_annotations.coco.json    (optional)

Output (created):
    {dst}/<split>/_annotations.coco.json (re-written with merged categories)
    {dst}/<split>/<image files>          (symlinked by default, --copy to copy)
    {dst}/REMAP_LOG.md                   (human-readable summary)
    {dst}/class_remap.applied.yaml       (a verbatim copy of the rules used)

The script never modifies the source dataset. Every annotation in every split
must be covered by either ``rename`` or ``drop`` in the rules file; the script
fails loudly otherwise to prevent silent label drift.

Usage (local Mac):

    python scripts/remap_classes.py \\
        --src /Users/james.peng/Desktop/Irrigation/datasets/poly-irrigation.v6-v2_w_boboflow.coco \\
        --dst /Users/james.peng/Desktop/Irrigation/datasets/poly-irrigation.v6-v2_w_boboflow.coco.merged \\
        --rules configs/class_remap.yaml

Usage (server):

    python scripts/remap_classes.py \\
        --src /home/rtx6000/james/datasets/poly-irrigation.v6-v2_w_boboflow.coco \\
        --dst /home/rtx6000/james/datasets/poly-irrigation.v6-v2_w_boboflow.coco.merged \\
        --rules configs/class_remap.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


SPLITS = ("train", "valid", "test")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path, help="Source COCO dataset root.")
    p.add_argument("--dst", required=True, type=Path, help="Output dataset root (created).")
    p.add_argument("--rules", required=True, type=Path, help="YAML file with the remap rules.")
    p.add_argument(
        "--image-mode",
        choices=["symlink", "copy", "none"],
        default="symlink",
        help="How to populate images in the output: symlink (fast, default), "
             "copy (portable), or none (annotations-only).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, remove an existing --dst directory before writing.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def load_rules(path: Path) -> dict:
    rules = yaml.safe_load(path.read_text())
    if not isinstance(rules, dict):
        raise SystemExit(f"Rules file {path} did not parse to a mapping.")
    rules.setdefault("rename", {})
    rules.setdefault("drop", [])
    rules.setdefault("drop_degenerate_bboxes", False)
    rules.setdefault("drop_empty_images", False)
    return rules


def _validate_rules_against_categories(
    rules: dict,
    all_source_names: set,
) -> None:
    rename: Dict[str, str] = rules["rename"]
    drop: List[str] = list(rules["drop"])

    rename_keys = set(rename.keys())
    drop_set = set(drop)

    overlap = rename_keys & drop_set
    if overlap:
        raise SystemExit(
            "Rule conflict: classes listed in both rename and drop: "
            + ", ".join(sorted(overlap))
        )

    missing = all_source_names - rename_keys - drop_set
    if missing:
        raise SystemExit(
            "Source classes not covered by rename or drop "
            f"(add them to configs/class_remap.yaml): {sorted(missing)}"
        )

    unknown = (rename_keys | drop_set) - all_source_names
    if unknown:
        # Not fatal -- the rules file can carry classes that don't exist in
        # this particular export -- but we warn so typos surface quickly.
        print(
            "  [warn] rule references classes not present in this dataset: "
            f"{sorted(unknown)}"
        )


# ---------------------------------------------------------------------------
# COCO helpers
# ---------------------------------------------------------------------------

def _is_background_category(cat: dict) -> bool:
    """Roboflow inserts a 'supercategory: none' placeholder we ignore."""
    return cat.get("supercategory") == "none"


def _load_split(src: Path, split: str) -> Optional[dict]:
    p = src / split / "_annotations.coco.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _ann_has_real_polyline(ann: dict) -> bool:
    pl = ann.get("polyline")
    if not pl:
        return False
    return any(isinstance(p, list) and len(p) >= 4 for p in pl)


def _ann_has_real_polygon(ann: dict) -> bool:
    seg = ann.get("segmentation")
    if not seg:
        return False
    if isinstance(seg, dict):  # RLE
        return True
    return any(isinstance(p, list) and len(p) >= 6 for p in seg)


def _ann_is_degenerate(ann: dict) -> bool:
    """An annotation is degenerate only if it carries no usable geometry at all.

    A bbox with ``w == 0`` or ``h == 0`` is *not* degenerate on its own — these
    routinely occur for purely horizontal/vertical pipe segments whose polyline
    is the real annotation. We only drop annotations that have neither a real
    polyline nor a real polygon AND whose bbox has zero area.
    """
    if _ann_has_real_polyline(ann) or _ann_has_real_polygon(ann):
        return False
    bb = ann.get("bbox")
    if not bb:
        return True  # no geometry of any kind
    _, _, w, h = bb
    return float(w) <= 0.0 or float(h) <= 0.0


# ---------------------------------------------------------------------------
# Core remap
# ---------------------------------------------------------------------------

def compute_destination_class_ids(
    rules: dict,
    train_doc: dict,
) -> Tuple[List[str], Dict[str, int]]:
    """Order destination classes by descending train-instance count; return
    (class_names_in_id_order, {dest_name: new_id}).

    New IDs start at 1 (COCO convention; 0 is implicit background).
    """
    rename: Dict[str, str] = rules["rename"]
    drop_set = set(rules["drop"])

    src_id_to_name = {c["id"]: c["name"] for c in train_doc["categories"]}
    dest_counts: Counter = Counter()
    for a in train_doc["annotations"]:
        src_name = src_id_to_name.get(a["category_id"])
        if src_name is None or src_name in drop_set:
            continue
        if src_name not in rename:
            # validation should have caught this, but defend in depth
            continue
        if rules["drop_degenerate_bboxes"] and _ann_is_degenerate(a):
            continue
        dest_counts[rename[src_name]] += 1

    # Deterministic order: by count desc, then by name asc as tiebreaker.
    ordered_names = [
        n for n, _ in sorted(dest_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    # Also include any rename destination that happens to have zero train
    # instances but is referenced by the rules; keep them so the schema is
    # complete. (Order them after the populated classes.)
    referenced = set(rename.values())
    for n in sorted(referenced - set(ordered_names)):
        ordered_names.append(n)

    name_to_id = {n: i + 1 for i, n in enumerate(ordered_names)}
    return ordered_names, name_to_id


def remap_split_doc(
    src_doc: dict,
    rules: dict,
    dest_categories: List[dict],
    name_to_dest_id: Dict[str, int],
) -> Tuple[dict, dict, set]:
    """Return (new_doc, stats_dict, kept_image_filenames) for one split.

    ``kept_image_filenames`` is the set of image basenames that should be
    placed in the output directory (everything else is excluded when
    ``drop_empty_images`` is enabled).
    """
    rename: Dict[str, str] = rules["rename"]
    drop_set = set(rules["drop"])

    src_id_to_name = {c["id"]: c["name"] for c in src_doc["categories"]}

    new_anns: List[dict] = []
    next_id = 1

    stats = {
        "src_total": len(src_doc["annotations"]),
        "dropped_class": 0,
        "dropped_degenerate": 0,
        "kept": 0,
        "by_dest_class": Counter(),
        "by_src_class_kept": Counter(),
        "by_src_class_dropped": Counter(),
    }

    for a in src_doc["annotations"]:
        src_name = src_id_to_name.get(a["category_id"])
        if src_name is None:
            continue
        if src_name in drop_set:
            stats["dropped_class"] += 1
            stats["by_src_class_dropped"][src_name] += 1
            continue
        if src_name not in rename:
            stats["dropped_class"] += 1
            stats["by_src_class_dropped"][src_name] += 1
            continue
        if rules["drop_degenerate_bboxes"] and _ann_is_degenerate(a):
            stats["dropped_degenerate"] += 1
            continue

        dest_name = rename[src_name]
        new_ann = dict(a)
        new_ann["id"] = next_id
        new_ann["category_id"] = name_to_dest_id[dest_name]
        new_anns.append(new_ann)
        next_id += 1

        stats["kept"] += 1
        stats["by_dest_class"][dest_name] += 1
        stats["by_src_class_kept"][src_name] += 1

    src_images: List[dict] = src_doc.get("images", [])
    img_has_ann = {a["image_id"] for a in new_anns}

    n_src_images = len(src_images)
    n_with_anns = len(img_has_ann)
    n_now_background = n_src_images - n_with_anns

    if rules["drop_empty_images"]:
        kept_images = [im for im in src_images if im["id"] in img_has_ann]
        stats["images_dropped_empty"] = n_now_background
        stats["images_now_background"] = 0
    else:
        kept_images = list(src_images)
        stats["images_dropped_empty"] = 0
        stats["images_now_background"] = n_now_background

    stats["images_total_src"] = n_src_images
    stats["images_total_dst"] = len(kept_images)
    stats["images_with_anns"] = n_with_anns

    kept_filenames = {im["file_name"] for im in kept_images}

    new_doc = {
        "info": {
            **src_doc.get("info", {}),
            "remap_note": (
                "Remapped by scripts/remap_classes.py using "
                "configs/class_remap.yaml. See REMAP_LOG.md."
            ),
        },
        "licenses": src_doc.get("licenses", []),
        "categories": dest_categories,
        "images": kept_images,
        "annotations": new_anns,
    }

    return new_doc, stats, kept_filenames


# ---------------------------------------------------------------------------
# Image copying / symlinking
# ---------------------------------------------------------------------------

def _populate_images(
    src_dir: Path,
    dst_dir: Path,
    mode: str,
    allow_filenames: Optional[set] = None,
) -> int:
    if mode == "none":
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src_dir.iterdir():
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if allow_filenames is not None and p.name not in allow_filenames:
            continue
        target = dst_dir / p.name
        if target.exists() or target.is_symlink():
            target.unlink()
        if mode == "symlink":
            os.symlink(p.resolve(), target)
        else:  # copy
            shutil.copy2(p, target)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_remap_log(
    dst: Path,
    rules: dict,
    ordered_dest_names: List[str],
    name_to_dest_id: Dict[str, int],
    per_split_stats: Dict[str, dict],
) -> None:
    lines: List[str] = []
    lines.append("# Class remap log\n")
    lines.append("Produced by `scripts/remap_classes.py` using `configs/class_remap.yaml`.\n")

    lines.append("## Final class set\n")
    lines.append("| new id | name | train kept | valid kept | test kept |")
    lines.append("| ---: | --- | ---: | ---: | ---: |")
    for n in ordered_dest_names:
        row = [str(name_to_dest_id[n]), n]
        for s in SPLITS:
            cnt = per_split_stats.get(s, {}).get("by_dest_class", Counter()).get(n, 0)
            row.append(str(cnt))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-split totals\n")
    lines.append(
        "| split | src anns | kept | dropped (class) | dropped (degenerate) | "
        "src images | kept images | empty-dropped | kept-as-background |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for s in SPLITS:
        st = per_split_stats.get(s)
        if not st:
            continue
        lines.append(
            f"| {s} | {st['src_total']} | {st['kept']} | "
            f"{st['dropped_class']} | {st['dropped_degenerate']} | "
            f"{st['images_total_src']} | {st['images_total_dst']} | "
            f"{st['images_dropped_empty']} | {st['images_now_background']} |"
        )
    lines.append("")

    lines.append("## Source-class fate\n")
    all_src = sorted({
        c for s in per_split_stats.values()
        for c in list(s.get("by_src_class_kept", {})) + list(s.get("by_src_class_dropped", {}))
    })
    rename = rules["rename"]
    drop_set = set(rules["drop"])
    lines.append("| source class | action | destination | train | valid | test |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for c in all_src:
        if c in drop_set:
            action, dest = "drop", "—"
        elif c in rename:
            action, dest = ("rename" if rename[c] != c else "keep"), rename[c]
        else:
            action, dest = "drop", "—"
        counts = []
        for s in SPLITS:
            st = per_split_stats.get(s, {})
            if action == "drop":
                counts.append(str(st.get("by_src_class_dropped", Counter()).get(c, 0)))
            else:
                counts.append(str(st.get("by_src_class_kept", Counter()).get(c, 0)))
        lines.append(f"| {c} | {action} | {dest} | " + " | ".join(counts) + " |")
    lines.append("")

    (dst / "REMAP_LOG.md").write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}")
    if dst == src:
        raise SystemExit("--src and --dst must be different paths.")

    if dst.exists():
        if not args.overwrite:
            raise SystemExit(
                f"--dst already exists: {dst}. Pass --overwrite to replace it."
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    rules = load_rules(args.rules)

    # Load all split docs up-front so we can validate against the *union* of
    # source category names and compute consistent destination IDs.
    split_docs: Dict[str, dict] = {}
    for s in SPLITS:
        doc = _load_split(src, s)
        if doc is not None:
            split_docs[s] = doc
    if "train" not in split_docs:
        raise SystemExit(f"Source has no train/_annotations.coco.json: {src}")

    all_source_names: set = set()
    for s, doc in split_docs.items():
        for c in doc["categories"]:
            if _is_background_category(c):
                continue
            all_source_names.add(c["name"])
    _validate_rules_against_categories(rules, all_source_names)

    ordered_dest_names, name_to_dest_id = compute_destination_class_ids(
        rules, split_docs["train"]
    )

    dest_categories = [
        {"id": name_to_dest_id[n], "name": n, "supercategory": "irrigation"}
        for n in ordered_dest_names
    ]

    per_split_stats: Dict[str, dict] = {}
    for s, doc in split_docs.items():
        new_doc, stats, kept_filenames = remap_split_doc(
            doc, rules, dest_categories, name_to_dest_id
        )
        per_split_stats[s] = stats

        split_out_dir = dst / s
        split_out_dir.mkdir(parents=True, exist_ok=True)
        with (split_out_dir / "_annotations.coco.json").open("w") as f:
            json.dump(new_doc, f)

        n_imgs = _populate_images(
            src / s, split_out_dir, args.image_mode, allow_filenames=kept_filenames
        )
        print(
            f"[{s}] anns: {stats['src_total']:>6} -> {stats['kept']:>6} kept, "
            f"{stats['dropped_class']} class-dropped, "
            f"{stats['dropped_degenerate']} degenerate-dropped; "
            f"images: {stats['images_total_src']} src -> {stats['images_total_dst']} dst "
            f"(empty-dropped: {stats['images_dropped_empty']}, "
            f"kept-as-background: {stats['images_now_background']}); "
            f"{n_imgs} image entries placed ({args.image_mode})."
        )

    # Persist the exact rules that were applied.
    shutil.copy2(args.rules, dst / "class_remap.applied.yaml")
    write_remap_log(dst, rules, ordered_dest_names, name_to_dest_id, per_split_stats)

    print(f"\nFinal classes ({len(ordered_dest_names)}):")
    for n in ordered_dest_names:
        print(f"  {name_to_dest_id[n]:>2}: {n}")
    print(f"\nWrote merged dataset to: {dst}")
    print(f"See {dst / 'REMAP_LOG.md'} for full counts.")


if __name__ == "__main__":
    main()
