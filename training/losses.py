"""Multi-class binary segmentation losses (one sigmoid channel per class).

The headline metric is **macro-averaged Dice** across ``K`` foreground classes:
each class contributes equally regardless of how dominant its pixel count is.
We mirror that in the loss so optimization and evaluation are aligned.

Recipes (selected via ``loss.name`` in the training config):

- ``bce_dice``   →  :class:`BCEDiceLoss`  — ``BCE(pos_weight) + w * macro-Dice``
  (the Phase-1 baseline; chosen because the v3 lateral_detection ladder
  showed clDice/Lovász add ≤ +0.0018 Dice on this task)
- ``composite`` →  :class:`CompositeLoss` — ``BCEDice [+ w_cl * clDice]
  [+ w_lv * Lovász]`` per channel, macro-averaged, with optional per-aux
  linear warmup. Aux weights default to ``0`` so this is functionally
  equivalent to ``bce_dice`` unless explicitly enabled.

All ``forward`` calls return a ``dict`` containing at least
``{"total": ..., "dice": ...}`` so the trainer's logging keeps working
across recipes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Per-channel Dice (macro-averaged across classes)
# ---------------------------------------------------------------------------


def per_class_soft_dice(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return shape-``(K,)`` soft-Dice ``in [0, 1]`` per class.

    Inputs: ``logits``, ``target`` both ``(B, K, H, W)``; target ∈ [0, 1].
    Each class's Dice is computed by summing intersection/union across the
    full batch ``B * H * W``, which matches our val accumulator's behavior
    and keeps small-batch variance low.
    """
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    dims = (0, 2, 3)  # sum over batch + spatial → keep class dim
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def macro_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """``1 - mean_k(per_class_soft_dice_k)``."""
    return 1.0 - per_class_soft_dice(logits, target, eps).mean()


# ---------------------------------------------------------------------------
# BCEDice (Phase-1 baseline)
# ---------------------------------------------------------------------------


class BCEDiceLoss(nn.Module):
    """Per-channel BCE + ``dice_weight`` × macro-Dice.

    BCE per-channel `pos_weight` is shared across classes by default. The
    rare-class problem is handled by **macro-Dice** averaging (which gives
    rare classes equal vote), not by manual BCE re-weighting — we can swap
    to per-channel ``pos_weight`` in Phase 2 if needed.
    """

    def __init__(self, pos_weight: float = 1.0, dice_weight: float = 1.0) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([float(pos_weight)]))
        self.dice_weight = float(dice_weight)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        # BCE pos_weight broadcasts over (K, H, W) when we pass a length-1 tensor.
        bce = F.binary_cross_entropy_with_logits(
            logits, target, pos_weight=self.pos_weight
        )
        dl = macro_dice_loss(logits, target)
        total = bce + self.dice_weight * dl
        return {"total": total, "bce": bce, "dice": dl}


# ---------------------------------------------------------------------------
# clDice (per-channel, macro-averaged) — kept for Phase 3 experimentation.
# Reference: Shit et al., "clDice", CVPR 2021. https://github.com/jocpae/clDice
# ---------------------------------------------------------------------------


def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Min-pool 3x3 (3x1 then 1x3 = approximate erosion). Differentiable."""
    p1 = -F.max_pool2d(-img, (3, 1), stride=(1, 1), padding=(1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), stride=(1, 1), padding=(0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(img, kernel_size=3, stride=1, padding=1)


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(img))


def soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Differentiable approximation of the morphological skeleton."""
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def cl_dice_loss_per_class(
    logits: torch.Tensor,
    target: torch.Tensor,
    iter_: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Return shape-``(K,)`` ``1 - clDice_k`` per class.

    Inputs are ``(B, K, H, W)``. Skeletonization is done per-channel; the
    soft_skel routines operate on the spatial dims, so we can fold ``K``
    into the batch dim for the skel pass.
    """
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    B, K, H, W = prob.shape
    prob_flat = prob.reshape(B * K, 1, H, W)
    targ_flat = target.reshape(B * K, 1, H, W)
    skel_pred = soft_skel(prob_flat, iter_=iter_).reshape(B, K, H, W)
    skel_targ = soft_skel(targ_flat, iter_=iter_).reshape(B, K, H, W)
    dims = (0, 2, 3)
    tprec = (
        (skel_pred * target).sum(dim=dims) + smooth
    ) / (skel_pred.sum(dim=dims) + smooth)
    tsens = (
        (skel_targ * prob).sum(dim=dims) + smooth
    ) / (skel_targ.sum(dim=dims) + smooth)
    cl_dice = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl_dice


def macro_cl_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iter_: int = 3,
    smooth: float = 1.0,
) -> torch.Tensor:
    return cl_dice_loss_per_class(logits, target, iter_, smooth).mean()


# ---------------------------------------------------------------------------
# Lovász hinge per channel (macro-averaged)
# Reference: Berman et al., "The Lovász-Softmax loss", CVPR 2018.
# ---------------------------------------------------------------------------


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted.float()).cumsum(0)
    jaccard = 1.0 - intersection / union
    p = len(gt_sorted)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[:-1].clone()
    return jaccard


def _lovasz_hinge_flat(logits_flat: torch.Tensor, labels_flat: torch.Tensor) -> torch.Tensor:
    if labels_flat.numel() == 0:
        return logits_flat.sum() * 0.0
    signs = 2.0 * labels_flat.float() - 1.0
    errors = 1.0 - logits_flat * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
    gt_sorted = labels_flat[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)


def lovasz_hinge_per_class(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return shape-``(K,)`` per-image-averaged Lovász hinge per class.

    Inputs: ``(B, K, H, W)``. For each class we compute one Lovász hinge per
    sample (over its ``H*W`` pixels), then mean over samples.
    """
    B, K, _, _ = logits.shape
    losses_per_class = []
    for k in range(K):
        per_sample = [
            _lovasz_hinge_flat(
                logits[b, k].reshape(-1),
                target[b, k].reshape(-1).long(),
            )
            for b in range(B)
        ]
        losses_per_class.append(torch.stack(per_sample).mean())
    return torch.stack(losses_per_class)


def macro_lovasz_hinge_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return lovasz_hinge_per_class(logits, target).mean()


# ---------------------------------------------------------------------------
# Composite (BCEDice + optional clDice + optional Lovász, per-aux warmup)
# ---------------------------------------------------------------------------


class CompositeLoss(nn.Module):
    """Sum of BCEDice + optionally weighted macro-clDice + optionally weighted
    macro-Lovász. All terms operate per-channel and are macro-averaged.

    Auxiliary losses can be linearly ramped from 0 to their target weight over
    ``warmup_epochs`` to avoid early-training instability. Set ``cldice_weight=0``
    or ``lovasz_weight=0`` to disable a component. A loss with all aux weights
    = 0 is functionally equivalent to BCEDice (which is also our default config).

    Expects the trainer to call ``set_epoch(epoch)`` once per epoch before
    train_epoch — otherwise weights stay at full strength.
    """

    def __init__(
        self,
        bce_pos_weight: float = 1.0,
        dice_weight: float = 1.0,
        cldice_weight: float = 0.0,
        cldice_warmup: int = 0,
        cldice_iter: int = 3,
        lovasz_weight: float = 0.0,
        lovasz_warmup: int = 0,
    ) -> None:
        super().__init__()
        self.bce_dice = BCEDiceLoss(
            pos_weight=bce_pos_weight, dice_weight=dice_weight
        )
        self.cldice_weight = float(cldice_weight)
        self.cldice_warmup = int(cldice_warmup)
        self.cldice_iter = int(cldice_iter)
        self.lovasz_weight = float(lovasz_weight)
        self.lovasz_warmup = int(lovasz_warmup)
        self.current_epoch = 1

    def set_epoch(self, epoch: int) -> None:
        self.current_epoch = int(epoch)

    def _eff(self, weight: float, warmup: int) -> float:
        if warmup <= 0 or self.current_epoch >= warmup:
            return weight
        return weight * (self.current_epoch / warmup)

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        bd = self.bce_dice(logits, target)
        out: dict[str, torch.Tensor] = {
            "bce": bd["bce"],
            "dice": bd["dice"],
            "total": bd["total"],
        }
        if self.cldice_weight > 0:
            cd = macro_cl_dice_loss(logits, target, iter_=self.cldice_iter)
            out["cldice"] = cd
            out["total"] = out["total"] + self._eff(self.cldice_weight, self.cldice_warmup) * cd
        if self.lovasz_weight > 0:
            lv = macro_lovasz_hinge_loss(logits, target)
            out["lovasz"] = lv
            out["total"] = out["total"] + self._eff(self.lovasz_weight, self.lovasz_warmup) * lv
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_loss(cfg: dict) -> nn.Module:
    name = cfg.get("name", "bce_dice")
    if name == "bce_dice":
        return BCEDiceLoss(
            pos_weight=cfg.get("bce_pos_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
        )
    if name == "composite":
        return CompositeLoss(
            bce_pos_weight=cfg.get("bce_pos_weight", 1.0),
            dice_weight=cfg.get("dice_weight", 1.0),
            cldice_weight=cfg.get("cldice_weight", 0.0),
            cldice_warmup=cfg.get("cldice_warmup", 0),
            cldice_iter=cfg.get("cldice_iter", 3),
            lovasz_weight=cfg.get("lovasz_weight", 0.0),
            lovasz_warmup=cfg.get("lovasz_warmup", 0),
        )
    raise ValueError(f"Unknown loss name: {name!r}")
