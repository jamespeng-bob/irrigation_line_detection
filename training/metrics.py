"""Validation-time metric accumulators for multi-class irrigation lines.

Every accumulator tracks per-class statistics over ``K`` channels and
exposes both **per-class** values and a **macro-averaged** scalar. Macro
averaging matches our loss + best-checkpoint metric (rare classes get
equal vote), and the per-class breakdown is what tells us *which* class
is dragging the macro down.

All ``compute()`` calls are DDP-safe — under ``torch.distributed`` we
``all_reduce`` the raw counts (or ``all_gather`` the per-sample lists) so
the reported number is the *global* metric over the whole val set, not
just the local rank's shard.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import torch
import torch.distributed as dist
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_sum_tensor(t: torch.Tensor) -> torch.Tensor:
    """In-place SUM all-reduce of ``t`` across ranks (no-op single-GPU)."""
    if not _ddp_active():
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _all_gather_list(local_list: list) -> list:
    if not _ddp_active():
        return list(local_list)
    world = dist.get_world_size()
    gathered: List[Any] = [None] * world
    dist.all_gather_object(gathered, list(local_list))
    out: list = []
    for sub in gathered:
        if sub:
            out.extend(sub)
    return out


# ---------------------------------------------------------------------------
# Per-class binary Dice
# ---------------------------------------------------------------------------


class PerClassDiceAccumulator:
    """``Dice_k = 2 * inter_k / (sum_pred_k + sum_target_k)`` per class.

    Predictions thresholded at ``threshold`` (default 0.5). Counts are
    accumulated as length-``K`` float64 tensors and SUM-all-reduced under DDP.
    """

    higher_is_better = True

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.K = int(num_classes)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        # CPU float64 buffers — moved to device only at all_reduce time.
        self.inter = torch.zeros(self.K, dtype=torch.float64)
        self.denom = torch.zeros(self.K, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).float()  # (B, K, H, W)
        tgt = (target >= 0.5).float()
        dims = (0, 2, 3)
        self.inter += (prob * tgt).sum(dim=dims).detach().double().cpu()
        self.denom += (prob.sum(dim=dims) + tgt.sum(dim=dims)).detach().double().cpu()

    def compute(self) -> dict:
        if _ddp_active():
            device = torch.device("cuda", torch.cuda.current_device())
            inter = _all_reduce_sum_tensor(self.inter.to(device).clone()).cpu()
            denom = _all_reduce_sum_tensor(self.denom.to(device).clone()).cpu()
        else:
            inter, denom = self.inter, self.denom
        per_class = ((2.0 * inter + self.eps) / (denom + self.eps)).numpy()
        return {
            "per_class": per_class.tolist(),
            "macro": float(per_class.mean()),
        }


# ---------------------------------------------------------------------------
# Per-class binary IoU
# ---------------------------------------------------------------------------


class PerClassIoUAccumulator:
    higher_is_better = True

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.K = int(num_classes)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.inter = torch.zeros(self.K, dtype=torch.float64)
        self.sum_pred = torch.zeros(self.K, dtype=torch.float64)
        self.sum_targ = torch.zeros(self.K, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).float()
        tgt = (target >= 0.5).float()
        dims = (0, 2, 3)
        self.inter += (prob * tgt).sum(dim=dims).detach().double().cpu()
        self.sum_pred += prob.sum(dim=dims).detach().double().cpu()
        self.sum_targ += tgt.sum(dim=dims).detach().double().cpu()

    def compute(self) -> dict:
        if _ddp_active():
            device = torch.device("cuda", torch.cuda.current_device())
            inter = _all_reduce_sum_tensor(self.inter.to(device).clone()).cpu()
            sp = _all_reduce_sum_tensor(self.sum_pred.to(device).clone()).cpu()
            st = _all_reduce_sum_tensor(self.sum_targ.to(device).clone()).cpu()
        else:
            inter, sp, st = self.inter, self.sum_pred, self.sum_targ
        union = sp + st - inter
        per_class = ((inter + self.eps) / (union + self.eps)).numpy()
        return {
            "per_class": per_class.tolist(),
            "macro": float(per_class.mean()),
        }


# ---------------------------------------------------------------------------
# Per-class clDice (skimage CPU skeletonize, per sample)
# ---------------------------------------------------------------------------


class PerClassClDiceAccumulator:
    """``clDice_k = 2 * tprec_k * tsens_k / (tprec_k + tsens_k)`` per class.

    Skeletonization is done on CPU with ``skimage.morphology.skeletonize``,
    one (sample, class) at a time. ~50ms per (1024², single-channel) at
    train time; acceptable at validation cadence.
    """

    higher_is_better = True

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.K = int(num_classes)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.tprec_num = torch.zeros(self.K, dtype=torch.float64)
        self.tprec_den = torch.zeros(self.K, dtype=torch.float64)
        self.tsens_num = torch.zeros(self.K, dtype=torch.float64)
        self.tsens_den = torch.zeros(self.K, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).detach().cpu().numpy()
        targ = (target >= 0.5).detach().cpu().numpy()
        # Expect (B, K, H, W).
        B, K, H, W = prob.shape
        for b in range(B):
            for k in range(K):
                p = prob[b, k].astype(bool)
                t = targ[b, k].astype(bool)
                sp = skeletonize(p)
                st = skeletonize(t)
                self.tprec_num[k] += float((sp & t).sum())
                self.tprec_den[k] += float(sp.sum())
                self.tsens_num[k] += float((st & p).sum())
                self.tsens_den[k] += float(st.sum())

    def compute(self) -> dict:
        if _ddp_active():
            device = torch.device("cuda", torch.cuda.current_device())
            tpn = _all_reduce_sum_tensor(self.tprec_num.to(device).clone()).cpu()
            tpd = _all_reduce_sum_tensor(self.tprec_den.to(device).clone()).cpu()
            tsn = _all_reduce_sum_tensor(self.tsens_num.to(device).clone()).cpu()
            tsd = _all_reduce_sum_tensor(self.tsens_den.to(device).clone()).cpu()
        else:
            tpn, tpd = self.tprec_num, self.tprec_den
            tsn, tsd = self.tsens_num, self.tsens_den
        tprec = (tpn / (tpd + self.eps)).numpy()
        tsens = (tsn / (tsd + self.eps)).numpy()
        per_class = 2.0 * tprec * tsens / (tprec + tsens + self.eps)
        return {
            "per_class": per_class.tolist(),
            "macro": float(per_class.mean()),
        }


# ---------------------------------------------------------------------------
# Per-class length ratio (pred_fg / gt_fg), both raw and skeleton-pixel
# ---------------------------------------------------------------------------


class PerClassLengthRatioAccumulator:
    """Per-sample, per-class ratios collected as Python lists (one per class).

    Tracks two ratios in parallel for every (sample, class) where the class
    has GT pixels:

    * ``pixel``    = #pred_fg / #gt_fg            (width-sensitive)
    * ``skeleton`` = #skel(pred) / #skel(gt)      (width-invariant)

    Aggregates report mean / median / p25 / p75 of each distribution per class
    plus a macro-average (mean of per-class means). A median far from 1.0
    indicates systematic over- or under-prediction for that class.
    """

    higher_is_better = None  # not unidirectional — "closer to 1.0" is good

    def __init__(
        self,
        num_classes: int,
        threshold: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        self.K = int(num_classes)
        self.threshold = float(threshold)
        self.eps = float(eps)
        self.reset()

    def reset(self) -> None:
        self.pixel: list[list[float]] = [[] for _ in range(self.K)]
        self.skel: list[list[float]] = [[] for _ in range(self.K)]

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        prob = (torch.sigmoid(logits) >= self.threshold).detach().cpu().numpy()
        targ = (target >= 0.5).detach().cpu().numpy()
        B, K, _, _ = prob.shape
        for b in range(B):
            for k in range(K):
                p = prob[b, k].astype(bool)
                t = targ[b, k].astype(bool)
                gt_pix = float(t.sum())
                if gt_pix < 1.0:
                    continue
                self.pixel[k].append(float(p.sum()) / (gt_pix + self.eps))
                gt_skel = float(skeletonize(t).sum())
                if gt_skel < 1.0:
                    continue
                pred_skel = float(skeletonize(p).sum())
                self.skel[k].append(pred_skel / (gt_skel + self.eps))

    def compute(self) -> dict:
        # Gather per-class lists across ranks.
        pixel_gathered = [_all_gather_list(lst) for lst in self.pixel]
        skel_gathered = [_all_gather_list(lst) for lst in self.skel]

        def _dist_stats(lists: list[list[float]]) -> dict:
            per_class = []
            for ratios in lists:
                if not ratios:
                    per_class.append({
                        "n": 0, "mean": float("nan"), "median": float("nan"),
                        "p25": float("nan"), "p75": float("nan"),
                    })
                    continue
                arr = np.asarray(ratios, dtype=np.float64)
                per_class.append({
                    "n": int(arr.size),
                    "mean": float(arr.mean()),
                    "median": float(np.median(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                })
            valid_means = [d["mean"] for d in per_class if d["n"] > 0]
            macro_mean = float(np.mean(valid_means)) if valid_means else float("nan")
            valid_medians = [d["median"] for d in per_class if d["n"] > 0]
            macro_median = float(np.mean(valid_medians)) if valid_medians else float("nan")
            return {
                "per_class": per_class,
                "macro_mean": macro_mean,
                "macro_median": macro_median,
            }

        return {
            "pixel": _dist_stats(pixel_gathered),
            "skel": _dist_stats(skel_gathered),
        }
