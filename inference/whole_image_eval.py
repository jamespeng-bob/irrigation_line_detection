"""Whole-image, per-class Dice evaluator (run every K epochs).

The tile-level validation accumulators are fast (cheap to run every epoch)
but they only see ``pos_only_grid`` tiles — tiles that contain at least one
GT pixel. They therefore miss the most common production-time failure mode:
false positives on completely empty background regions. The mismatch shows
up dramatically in the length-ratio mean (the v3 lateral_detection ladder
hit skel-length-ratio means of 3–5 even with median ≈ 1.0).

This module runs *true* full-image inference (via ``inference.tiled_predict``)
on every valid image, rasterizes the full-image GT, and computes per-class
Dice / IoU / clDice / length-ratio at the image level. Aggregates are
sum-based across images (Dice = ``2 * sum_inter / (sum_pred + sum_target)``)
so a few small/easy plans can't dominate a few large/hard ones.

Called by the trainer once every ``whole_image_eval_every`` epochs (default
every 10) and at the final epoch. Skipped on non-rank-0 ranks under DDP.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import skeletonize
from tqdm import tqdm

from data.coco_loader import ClassPolyline, ImageRecord, load_split
from inference.tiled_predict import predict_full_image


@dataclass
class WholeImageEvalConfig:
    valid_dir: Path
    num_classes: int
    class_names: list[str]
    tile_size: int = 1024
    stride: int = 768
    thickness: int = 4
    device: str = "cuda:0"
    mean: tuple[float, float, float] = (0.9548, 0.9548, 0.9548)
    std: tuple[float, float, float] = (0.1850, 0.1850, 0.1850)
    batch_size: int = 4
    threshold: float = 0.5
    # If set, run on only the first N images. Useful for local smoke tests.
    max_images: Optional[int] = None
    # Cache full-image GT masks across calls to avoid re-rasterizing each time.
    cache_gt: bool = True
    # Optional: restrict the loader (and therefore the per-class metrics) to
    # a subset of the COCO classes. The channel order follows the allowlist,
    # matching what the trained model was fed.
    class_allowlist: Optional[list[str]] = None


def _rasterize_class_mask(
    polylines: list[ClassPolyline],
    height: int,
    width: int,
    num_classes: int,
    thickness: int,
) -> np.ndarray:
    """Rasterize a list of class-tagged polylines into ``(K, H, W) uint8``."""
    mask = np.zeros((num_classes, height, width), dtype=np.uint8)
    for pl in polylines:
        pts_i = pl.points.astype(np.int32).reshape(-1, 1, 2)
        if pts_i.shape[0] < 2:
            continue
        cv2.polylines(
            mask[pl.class_idx], [pts_i], isClosed=False, color=255,
            thickness=thickness, lineType=cv2.LINE_8,
        )
    return mask


class WholeImageEvaluator:
    """Stateful evaluator that caches GT masks across epochs.

    Build once at the start of training; call ``evaluate(model, epoch)``
    every N epochs. Returns a dict with per-class and macro statistics
    plus the per-image breakdown.
    """

    def __init__(self, config: WholeImageEvalConfig) -> None:
        self.config = config
        (
            self.images,
            self.polylines_by_image,
            self.class_names,
            self._cat_id_to_channel,
            _stats,
        ) = load_split(
            self.config.valid_dir,
            class_allowlist=self.config.class_allowlist,
        )
        # Defensive — the trainer also validates this, but a mismatch here
        # would silently produce wrong metrics, so re-check.
        assert len(self.class_names) == self.config.num_classes, (
            f"WholeImageEvaluator: valid split has {len(self.class_names)} classes "
            f"but config.num_classes={self.config.num_classes}"
        )
        self._gt_cache: dict[int, np.ndarray] = {}

    def _get_gt(self, img_id: int, rec: ImageRecord) -> np.ndarray:
        if self.config.cache_gt and img_id in self._gt_cache:
            return self._gt_cache[img_id]
        mask = _rasterize_class_mask(
            self.polylines_by_image.get(img_id, []),
            height=rec.height,
            width=rec.width,
            num_classes=self.config.num_classes,
            thickness=self.config.thickness,
        )
        if self.config.cache_gt:
            self._gt_cache[img_id] = mask
        return mask

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        epoch: int,
        show_progress: bool = True,
    ) -> dict:
        """Run whole-image inference + per-class accumulation over all valid images."""
        cfg = self.config
        K = cfg.num_classes

        # Sum-based accumulators (per class)
        inter = np.zeros(K, dtype=np.float64)
        sum_pred = np.zeros(K, dtype=np.float64)
        sum_targ = np.zeros(K, dtype=np.float64)
        # clDice components (per class)
        tprec_num = np.zeros(K, dtype=np.float64)
        tprec_den = np.zeros(K, dtype=np.float64)
        tsens_num = np.zeros(K, dtype=np.float64)
        tsens_den = np.zeros(K, dtype=np.float64)
        # Per-image, per-class skeleton length ratios (for distribution stats)
        skel_ratios: list[list[float]] = [[] for _ in range(K)]

        per_image_rows: list[dict] = []

        image_ids = list(self.images.keys())
        if cfg.max_images is not None:
            image_ids = image_ids[: int(cfg.max_images)]

        iterator = image_ids
        if show_progress:
            iterator = tqdm(image_ids, desc=f"whole-image eval (ep{epoch})", leave=False)

        for img_id in iterator:
            rec = self.images[img_id]
            gt = self._get_gt(img_id, rec)                  # (K, H, W) uint8
            probs = predict_full_image(
                model,
                rec.path,
                num_classes=K,
                tile_size=cfg.tile_size,
                stride=cfg.stride,
                device=cfg.device,
                mean=cfg.mean,
                std=cfg.std,
                batch_size=cfg.batch_size,
                show_progress=False,
            )                                               # (K, H, W) float32
            pred = (probs >= cfg.threshold)                 # (K, H, W) bool
            gt_b = (gt >= 128)                              # (K, H, W) bool

            per_image_class: list[dict] = []
            for k in range(K):
                p, t = pred[k], gt_b[k]
                pi = float((p & t).sum())
                sp = float(p.sum())
                st = float(t.sum())
                inter[k] += pi
                sum_pred[k] += sp
                sum_targ[k] += st
                # clDice
                if st > 0:
                    skel_t = skeletonize(t)
                    skel_p = skeletonize(p)
                    tprec_num[k] += float((skel_p & t).sum())
                    tprec_den[k] += float(skel_p.sum())
                    tsens_num[k] += float((skel_t & p).sum())
                    tsens_den[k] += float(skel_t.sum())
                    gt_skel = float(skel_t.sum())
                    if gt_skel >= 1.0:
                        skel_ratios[k].append(float(skel_p.sum()) / max(gt_skel, 1e-6))
                # Per-image Dice for the breakdown
                dice_k = (2.0 * pi + 1e-6) / (sp + st + 1e-6) if (sp + st) > 0 else float("nan")
                per_image_class.append({
                    "class": self.class_names[k],
                    "dice": float(dice_k),
                    "gt_pixels": int(st),
                    "pred_pixels": int(sp),
                })
            per_image_rows.append({
                "image_id": int(img_id),
                "file_name": rec.file_name,
                "per_class": per_image_class,
            })

        # Aggregate
        eps = 1e-6
        dice_per_class = (2.0 * inter + eps) / (sum_pred + sum_targ + eps)
        union = sum_pred + sum_targ - inter
        iou_per_class = (inter + eps) / (union + eps)
        tprec = tprec_num / (tprec_den + eps)
        tsens = tsens_num / (tsens_den + eps)
        cldice_per_class = 2.0 * tprec * tsens / (tprec + tsens + eps)

        def _ratio_stats(ratios: list[float]) -> dict:
            if not ratios:
                return {"n": 0, "mean": float("nan"), "median": float("nan"),
                        "p25": float("nan"), "p75": float("nan")}
            arr = np.asarray(ratios, dtype=np.float64)
            return {
                "n": int(arr.size),
                "mean": float(arr.mean()),
                "median": float(np.median(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
            }

        skel_stats = [_ratio_stats(s) for s in skel_ratios]
        valid_means = [s["mean"] for s in skel_stats if s["n"] > 0]
        macro_skel_mean = float(np.mean(valid_means)) if valid_means else float("nan")
        valid_medians = [s["median"] for s in skel_stats if s["n"] > 0]
        macro_skel_median = float(np.mean(valid_medians)) if valid_medians else float("nan")

        return {
            "epoch": epoch,
            "n_images": len(image_ids),
            "dice_per_class": dice_per_class.tolist(),
            "dice_macro": float(dice_per_class.mean()),
            "iou_per_class": iou_per_class.tolist(),
            "iou_macro": float(iou_per_class.mean()),
            "cldice_per_class": cldice_per_class.tolist(),
            "cldice_macro": float(cldice_per_class.mean()),
            "skel_length_ratio_per_class": skel_stats,
            "skel_length_ratio_macro_mean": macro_skel_mean,
            "skel_length_ratio_macro_median": macro_skel_median,
            "class_names": list(self.class_names),
            "per_image": per_image_rows,
        }
