"""Multi-model whole-image fusion evaluator.

Run a "base" multi-class model + one or more "specialist" models on the
same valid split, then compute per-class Dice / IoU / clDice both **alone**
(base only) and **fused** (specialist probabilities override the base's
predictions for the specialist's classes).

This is the MoE-at-inference evaluator. Use it to decide whether shipping
``base + specialists`` together actually improves per-class numbers vs.
shipping just the joint model.

Override rule (simple, room to evolve):
    For each pixel and each class in the specialist's class list, the
    fused probability = specialist's sigmoid output. Other classes use
    the base's output unchanged. No averaging, no calibration — we want
    to know if the specialist's raw answer is better than the base's.

Per-class metrics are sum-based across the whole valid split, matching
``inference.whole_image_eval``'s aggregation so the numbers are directly
comparable to ``runs/<run>/history.json``'s ``whole_image_eval`` blocks.

Usage:

    python scripts/eval_fusion.py \\
        --base       runs/v2_general_classbalanced/best_whole_image.pth \\
        --specialist runs/v2_specialist_lateral_pair/best_whole_image.pth \\
        --specialist runs/v2_specialist_main_pair/best_whole_image.pth \\
        --valid-dir  ../datasets/poly-irrigation.v6-v2_w_boboflow.coco.merged/valid \\
        --device     cuda:0 \\
        --out        reports/fusion_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.coco_loader import load_split
from inference.tiled_predict import predict_full_image
from inference.whole_image_eval import _rasterize_class_mask  # type: ignore
from models.unet import build_model


@dataclass
class LoadedModel:
    """A loaded checkpoint plus the metadata we need to fuse with others."""

    tag: str                  # short label for logs (derived from checkpoint dir)
    model: torch.nn.Module
    class_names: list[str]
    encoder_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", required=True, type=Path,
                   help="Path to the base (multi-class) checkpoint .pth.")
    p.add_argument("--specialist", action="append", default=[], type=Path,
                   help="Path to a specialist checkpoint .pth. Repeat for >1 specialist.")
    p.add_argument("--valid-dir", required=True, type=Path,
                   help="COCO valid split directory.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--thickness", type=int, default=4)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=4,
                   help="Tiles per forward pass inside predict_full_image.")
    p.add_argument("--mean", nargs=3, type=float,
                   default=[0.9548, 0.9548, 0.9548])
    p.add_argument("--std", nargs=3, type=float,
                   default=[0.1850, 0.1850, 0.1850])
    p.add_argument("--max-images", type=int, default=None,
                   help="If set, evaluate only the first N images (smoke test).")
    p.add_argument("--out", type=Path, default=None,
                   help="Where to write the JSON summary. Defaults to "
                        "reports/fusion_eval_<base>_<specs>.json")
    return p.parse_args()


def load_checkpoint(path: Path, device: str) -> LoadedModel:
    """Load a checkpoint and build its model with the saved metadata."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    class_names = list(ckpt.get("class_names") or [])
    if not class_names:
        raise ValueError(
            f"Checkpoint {path} has no 'class_names' field — re-train with the "
            "current trainer (post 'stamp encoder name' commit)."
        )
    encoder_name = ckpt.get("encoder_name", "mit_b2")
    model_name = ckpt.get("model_name", "smp_unet")
    model = build_model(
        {"name": model_name, "encoder": encoder_name, "encoder_weights": None},
        num_classes=len(class_names),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    tag = path.parent.name
    print(f"[fusion] loaded {tag}: encoder={encoder_name}  classes={class_names}")
    return LoadedModel(tag=tag, model=model, class_names=class_names,
                       encoder_name=encoder_name)


def build_specialist_overrides(
    base: LoadedModel,
    specialists: list[LoadedModel],
) -> list[tuple[LoadedModel, list[tuple[int, int]]]]:
    """For each specialist, build the list of (specialist_ch, base_ch) pairs.

    Validates that every specialist class is present in the base model and
    that no two specialists try to override the same base channel (we keep
    the override rule simple — first specialist in the CLI list wins on
    conflict, but we emit a clear warning).
    """
    base_lookup = {n: i for i, n in enumerate(base.class_names)}
    used: dict[int, str] = {}
    overrides: list[tuple[LoadedModel, list[tuple[int, int]]]] = []
    for sp in specialists:
        pairs: list[tuple[int, int]] = []
        for spec_ch, name in enumerate(sp.class_names):
            if name not in base_lookup:
                raise ValueError(
                    f"Specialist {sp.tag} predicts class {name!r} which is not "
                    f"in the base's class list {base.class_names}. Fusion is "
                    "only defined when specialist ⊆ base."
                )
            base_ch = base_lookup[name]
            if base_ch in used:
                print(f"[fusion] WARNING: base channel {base_ch} ({name}) already "
                      f"overridden by {used[base_ch]}; {sp.tag}'s prediction "
                      "will be ignored for this channel.")
                continue
            used[base_ch] = sp.tag
            pairs.append((spec_ch, base_ch))
        overrides.append((sp, pairs))
        print(f"[fusion] {sp.tag} overrides base channels "
              f"{[(p[1], base.class_names[p[1]]) for p in pairs]}")
    return overrides


# ---------------------------------------------------------------------------
# Aggregators
# ---------------------------------------------------------------------------


class _PerClassAggregator:
    """Sum-based Dice / IoU / clDice + per-class skel length ratio."""

    def __init__(self, K: int) -> None:
        self.K = K
        self.inter = np.zeros(K, dtype=np.float64)
        self.sum_pred = np.zeros(K, dtype=np.float64)
        self.sum_targ = np.zeros(K, dtype=np.float64)
        self.tprec_num = np.zeros(K, dtype=np.float64)
        self.tprec_den = np.zeros(K, dtype=np.float64)
        self.tsens_num = np.zeros(K, dtype=np.float64)
        self.tsens_den = np.zeros(K, dtype=np.float64)
        self.skel_ratios: list[list[float]] = [[] for _ in range(K)]

    def update(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """``pred`` and ``gt``: ``(K, H, W)`` bool arrays."""
        for k in range(self.K):
            p, t = pred[k], gt[k]
            self.inter[k] += float((p & t).sum())
            self.sum_pred[k] += float(p.sum())
            self.sum_targ[k] += float(t.sum())
            if t.any():
                skel_t = skeletonize(t)
                skel_p = skeletonize(p)
                self.tprec_num[k] += float((skel_p & t).sum())
                self.tprec_den[k] += float(skel_p.sum())
                self.tsens_num[k] += float((skel_t & p).sum())
                self.tsens_den[k] += float(skel_t.sum())
                gt_skel = float(skel_t.sum())
                if gt_skel >= 1.0:
                    self.skel_ratios[k].append(float(skel_p.sum()) / max(gt_skel, 1e-6))

    def compute(self, class_names: list[str]) -> dict:
        eps = 1e-6
        dice = (2.0 * self.inter + eps) / (self.sum_pred + self.sum_targ + eps)
        union = self.sum_pred + self.sum_targ - self.inter
        iou = (self.inter + eps) / (union + eps)
        tprec = self.tprec_num / (self.tprec_den + eps)
        tsens = self.tsens_num / (self.tsens_den + eps)
        cldice = 2.0 * tprec * tsens / (tprec + tsens + eps)
        skel_stats = []
        for ratios in self.skel_ratios:
            if not ratios:
                skel_stats.append({"n": 0, "mean": float("nan"), "median": float("nan")})
                continue
            arr = np.asarray(ratios, dtype=np.float64)
            skel_stats.append({
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
            })
        return {
            "class_names": list(class_names),
            "dice_per_class": dice.tolist(),
            "dice_macro": float(dice.mean()),
            "iou_per_class": iou.tolist(),
            "iou_macro": float(iou.mean()),
            "cldice_per_class": cldice.tolist(),
            "cldice_macro": float(cldice.mean()),
            "skel_length_ratio_per_class": skel_stats,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    device = args.device

    base = load_checkpoint(args.base, device)
    specialists = [load_checkpoint(p, device) for p in args.specialist]
    overrides = build_specialist_overrides(base, specialists)

    K = len(base.class_names)

    (
        images,
        polylines_by_image,
        loaded_class_names,
        _cat_id_to_channel,
        _stats,
    ) = load_split(args.valid_dir, class_allowlist=base.class_names)
    # Sanity: enforce loader returned the exact channel order we asked for.
    if loaded_class_names != base.class_names:
        raise RuntimeError(
            f"Loader class order {loaded_class_names} doesn't match base "
            f"{base.class_names}. Likely a class_allowlist bug."
        )

    image_ids = list(images.keys())
    if args.max_images is not None:
        image_ids = image_ids[: int(args.max_images)]

    base_agg = _PerClassAggregator(K)
    fused_agg = _PerClassAggregator(K)

    mean = tuple(args.mean)
    std = tuple(args.std)

    t0 = time.time()
    for img_id in tqdm(image_ids, desc="fusion-eval"):
        rec = images[img_id]
        gt = _rasterize_class_mask(
            polylines_by_image.get(img_id, []),
            height=rec.height,
            width=rec.width,
            num_classes=K,
            thickness=args.thickness,
        )                                                       # (K, H, W) uint8
        gt_b = gt >= 128

        # Base prediction
        base_probs = predict_full_image(
            base.model, rec.path,
            num_classes=K,
            tile_size=args.tile_size, stride=args.stride,
            device=device, mean=mean, std=std,
            batch_size=args.batch_size, show_progress=False,
        )                                                       # (K, H, W) float32

        # Fused volume starts as a copy of base and is then overwritten.
        fused_probs = base_probs.copy()
        for sp, pairs in overrides:
            if not pairs:
                continue
            sp_probs = predict_full_image(
                sp.model, rec.path,
                num_classes=len(sp.class_names),
                tile_size=args.tile_size, stride=args.stride,
                device=device, mean=mean, std=std,
                batch_size=args.batch_size, show_progress=False,
            )                                                   # (K_sp, H, W)
            for spec_ch, base_ch in pairs:
                fused_probs[base_ch] = sp_probs[spec_ch]

        base_pred = base_probs >= args.threshold
        fused_pred = fused_probs >= args.threshold
        base_agg.update(base_pred, gt_b)
        fused_agg.update(fused_pred, gt_b)

    elapsed_s = time.time() - t0

    base_summary = base_agg.compute(base.class_names)
    fused_summary = fused_agg.compute(base.class_names)

    # --- Pretty-print -----------------------------------------------------
    def print_table(title: str, summary: dict) -> None:
        print()
        print(f"--- {title} ---")
        print(f"  macro:  dice={summary['dice_macro']:.4f}  iou={summary['iou_macro']:.4f}  "
              f"cldice={summary['cldice_macro']:.4f}")
        print(f"  per-class:")
        for k, name in enumerate(summary['class_names']):
            print(f"    {name:<18s} dice={summary['dice_per_class'][k]:.4f}  "
                  f"iou={summary['iou_per_class'][k]:.4f}  "
                  f"cldice={summary['cldice_per_class'][k]:.4f}")

    print_table("BASE alone", base_summary)
    if overrides:
        print_table("FUSED (base + specialists)", fused_summary)
        print()
        print("--- delta (fused - base) ---")
        dpc_b = base_summary["dice_per_class"]
        dpc_f = fused_summary["dice_per_class"]
        print(f"  macro dice delta = {fused_summary['dice_macro'] - base_summary['dice_macro']:+.4f}")
        for k, name in enumerate(base_summary["class_names"]):
            print(f"    {name:<18s} ΔDice = {dpc_f[k] - dpc_b[k]:+.4f}")

    print()
    print(f"[fusion] {len(image_ids)} images in {elapsed_s/60:.1f} min "
          f"({elapsed_s/max(1,len(image_ids)):.1f} s/image)")

    # --- Persist ----------------------------------------------------------
    out_path = args.out
    if out_path is None:
        out_path = ROOT / "reports" / (
            "fusion_eval__" + base.tag + "__" + "+".join(s.tag for s in specialists) + ".json"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "base": {
            "tag": base.tag,
            "checkpoint": str(args.base),
            "class_names": base.class_names,
            "encoder_name": base.encoder_name,
        },
        "specialists": [
            {"tag": s.tag, "checkpoint": str(p),
             "class_names": s.class_names, "encoder_name": s.encoder_name}
            for s, p in zip(specialists, args.specialist)
        ],
        "n_images": len(image_ids),
        "elapsed_s": elapsed_s,
        "base_alone": base_summary,
        "fused": fused_summary if overrides else None,
    }, indent=2))
    print(f"[fusion] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
