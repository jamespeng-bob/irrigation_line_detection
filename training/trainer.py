"""Train/val loop for multi-class binary irrigation-line segmentation.

Single-GPU and DDP both supported through the same class. When DDP is
active:

- All ranks run forward/backward on their data shard.
- All ranks run validation on their val shard.
- Metric accumulators all-reduce inside ``compute()`` so the reported
  numbers are the *global* val metrics over the whole val set.
- Only rank 0 prints, writes ``history.{json,png}``, writes checkpoints,
  saves val viz, and runs whole-image evaluation.
- Whole-image eval runs every ``whole_image_eval_every`` epochs and at
  the final epoch. It uses the full ``inference.tiled_predict`` stack and
  is the trustworthy "production" metric.

Single-GPU runs work unchanged — every DDP check short-circuits.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inference.whole_image_eval import WholeImageEvaluator

from .metrics import (
    PerClassClDiceAccumulator,
    PerClassDiceAccumulator,
    PerClassIoUAccumulator,
    PerClassLengthRatioAccumulator,
)


@dataclass
class TrainerConfig:
    save_dir: str
    num_classes: int
    class_names: list[str]
    # Model identity, copied through to every saved checkpoint so any
    # downstream loader (fusion eval, deployment, etc.) can rebuild the
    # network without consulting the training config separately.
    model_name: str = "smp_unet"
    encoder_name: str = "mit_b2"
    epochs: int = 80
    lr: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    log_interval: int = 20
    val_viz_count: int = 4
    device: str = "cuda"
    best_metric: str = "dice_macro"  # 'dice_macro' or 'loss'
    lr_schedule: str = "constant"     # 'constant' | 'cosine'
    warmup_epochs: int = 0
    cosine_min_lr_ratio: float = 0.01
    # Whole-image eval cadence. The whole-image evaluator runs every N epochs
    # AND on the final epoch. Set to 0 to disable entirely.
    whole_image_eval_every: int = 10


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------


def _ddp_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _is_rank_zero() -> bool:
    return (not _ddp_active()) or dist.get_rank() == 0


def _all_reduce_mean(value: float) -> float:
    if not _ddp_active():
        return float(value)
    device = torch.device("cuda", torch.cuda.current_device())
    t = torch.tensor([float(value)], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float((t / dist.get_world_size()).item())


def _build_lr_scheduler(
    optimizer: optim.Optimizer,
    schedule: str,
    total_epochs: int,
    warmup_epochs: int,
    cosine_min_lr_ratio: float,
):
    if schedule == "constant":
        return None
    if schedule != "cosine":
        raise ValueError(f"Unknown lr_schedule: {schedule!r}")

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine_min_lr_ratio + (1.0 - cosine_min_lr_ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SegTrainer:
    """Train a multi-class segmentation model end-to-end (single GPU or DDP)."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        normalization: tuple[list[float], list[float]],
        whole_image_evaluator: Optional[WholeImageEvaluator] = None,
    ) -> None:
        self.model = model
        self.criterion = criterion.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        self.class_names = list(config.class_names)
        self.K = int(config.num_classes)
        self.whole_image_evaluator = whole_image_evaluator

        params = (
            self.model.module.parameters() if hasattr(self.model, "module")
            else self.model.parameters()
        )
        self.optimizer = optim.AdamW(
            [p for p in params if p.requires_grad],
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
        )

        self.lr_scheduler = _build_lr_scheduler(
            self.optimizer,
            schedule=config.lr_schedule,
            total_epochs=config.epochs,
            warmup_epochs=config.warmup_epochs,
            cosine_min_lr_ratio=config.cosine_min_lr_ratio,
        )

        self.save_dir = Path(config.save_dir)
        if _is_rank_zero():
            self.save_dir.mkdir(parents=True, exist_ok=True)

        self.mean = np.array(normalization[0], dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(normalization[1], dtype=np.float32).reshape(3, 1, 1)

        # Scalar histories (always logged each epoch)
        self.history: dict = {
            "train_total": [],
            "train_dice_loss": [],
            "val_total": [],
            "val_dice_loss": [],
            "val_dice_macro": [],
            "val_iou_macro": [],
            "val_cldice_macro": [],
            "val_length_ratio_skel_macro_mean": [],
            "val_length_ratio_skel_macro_median": [],
            "lr": [],
            # Per-class arrays (parallel to the macro arrays)
            "val_dice_per_class": [],
            "val_iou_per_class": [],
            "val_cldice_per_class": [],
            # Whole-image eval results (sparse: indexed by epoch number)
            "whole_image_eval": [],
        }
        self.best_val_dice_macro = -1.0
        self.best_val_loss = float("inf")
        # Separately track the best whole-image macro Dice so we save a
        # production-ready checkpoint distinct from the tile-best one.
        # v1's tile-best (ep 78) diverged from whole-image-best (ep 40):
        # ep-78 tile Dice was higher but whole-image Dice was lower —
        # the model had drifted to over-predict on background-only tiles.
        self.best_whole_image_dice_macro = -1.0

        self._dice_acc = PerClassDiceAccumulator(self.K, threshold=0.5)
        self._iou_acc = PerClassIoUAccumulator(self.K, threshold=0.5)
        self._cldice_acc = PerClassClDiceAccumulator(self.K, threshold=0.5)
        self._length_acc = PerClassLengthRatioAccumulator(self.K, threshold=0.5)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------

    def _to_device(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            batch["image"].to(self.device, non_blocking=True),
            batch["mask"].to(self.device, non_blocking=True),
        )

    def _set_epoch_on_sampler(self, loader: DataLoader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    # ------------------------------------------------------------------
    # Train / val epochs
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self._set_epoch_on_sampler(self.train_loader, epoch)
        totals = {"total": 0.0, "dice_loss": 0.0}
        n = 0
        pbar = tqdm(self.train_loader, desc="train", leave=False, disable=not _is_rank_zero())
        for batch in pbar:
            images, masks = self._to_device(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(images)
            losses = self.criterion(logits, masks)
            losses["total"].backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            totals["total"] += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            n += 1
            if n % self.config.log_interval == 0 and _is_rank_zero():
                pbar.set_postfix(
                    total=f"{totals['total']/n:.4f}",
                    dice=f"{totals['dice_loss']/n:.4f}",
                )
        n = max(n, 1)
        return {
            "total": _all_reduce_mean(totals["total"] / n),
            "dice_loss": _all_reduce_mean(totals["dice_loss"] / n),
        }

    @torch.no_grad()
    def val_epoch(self, epoch: int) -> dict:
        self.model.eval()
        self._set_epoch_on_sampler(self.val_loader, epoch)

        totals = {"total": 0.0, "dice_loss": 0.0}
        n = 0
        viz_saved = 0
        viz_dir = self.save_dir / "val_viz"
        if _is_rank_zero():
            viz_dir.mkdir(parents=True, exist_ok=True)

        for acc in (self._dice_acc, self._iou_acc, self._cldice_acc, self._length_acc):
            acc.reset()

        pbar = tqdm(self.val_loader, desc="val", leave=False, disable=not _is_rank_zero())
        for batch in pbar:
            images, masks = self._to_device(batch)
            logits = self.model(images)
            losses = self.criterion(logits, masks)

            totals["total"] += float(losses["total"].item())
            totals["dice_loss"] += float(losses["dice"].item())
            n += 1

            self._dice_acc.update(logits, masks)
            self._iou_acc.update(logits, masks)
            self._cldice_acc.update(logits, masks)
            self._length_acc.update(logits, masks)

            if _is_rank_zero() and viz_saved < self.config.val_viz_count:
                self._save_viz_panel(
                    images.detach().cpu(),
                    masks.detach().cpu(),
                    logits.detach().cpu(),
                    viz_dir=viz_dir,
                    epoch=epoch,
                    start_idx=viz_saved,
                    max_save=self.config.val_viz_count - viz_saved,
                )
                viz_saved += images.shape[0]

        n = max(n, 1)
        dice = self._dice_acc.compute()
        iou = self._iou_acc.compute()
        cldice = self._cldice_acc.compute()
        length = self._length_acc.compute()
        return {
            "total": _all_reduce_mean(totals["total"] / n),
            "dice_loss": _all_reduce_mean(totals["dice_loss"] / n),
            "dice": dice,
            "iou": iou,
            "cldice": cldice,
            "length": length,
        }

    # ------------------------------------------------------------------
    # Visualization (rank 0 only)
    # ------------------------------------------------------------------

    def _denormalize(self, img_t: torch.Tensor) -> np.ndarray:
        img = img_t.numpy() * self.std + self.mean
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img.transpose(1, 2, 0)

    def _class_color(self, k: int) -> tuple[int, int, int]:
        """Stable per-class RGB color for overlays."""
        cmap = plt.get_cmap("tab10")
        r, g, b, _ = cmap(k % 10)
        return int(r * 255), int(g * 255), int(b * 255)

    def _composite_mask(self, mask_kHW: np.ndarray) -> np.ndarray:
        """Render a ``(K, H, W)`` binary mask as an RGB overlay."""
        K, H, W = mask_kHW.shape
        out = np.zeros((H, W, 3), dtype=np.uint8)
        for k in range(K):
            r, g, b = self._class_color(k)
            mk = mask_kHW[k].astype(bool)
            out[mk] = (r, g, b)
        return out

    def _save_viz_panel(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        logits: torch.Tensor,
        viz_dir: Path,
        epoch: int,
        start_idx: int,
        max_save: int,
    ) -> None:
        probs = torch.sigmoid(logits).numpy()
        for b in range(min(images.shape[0], max_save)):
            img_rgb = self._denormalize(images[b])
            gt_kHW = (masks[b].numpy() >= 0.5).astype(np.uint8)
            pred_kHW = (probs[b] >= 0.5).astype(np.uint8)
            gt_rgb = self._composite_mask(gt_kHW)
            pred_rgb = self._composite_mask(pred_kHW)
            # Probability composite: max across classes, colored by argmax.
            argmax = probs[b].argmax(axis=0)
            prob_max = probs[b].max(axis=0)
            prob_color = np.zeros_like(img_rgb)
            for k in range(self.K):
                m = argmax == k
                r, g, b_ = self._class_color(k)
                prob_color[m] = (r, g, b_)
            # Modulate by max prob so background stays dark.
            prob_color = (prob_color.astype(np.float32) * prob_max[..., None]).astype(np.uint8)

            row = np.concatenate([img_rgb, gt_rgb, pred_rgb, prob_color], axis=1)
            out_path = viz_dir / f"epoch{epoch:03d}_sample{start_idx + b:02d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))

    # ------------------------------------------------------------------
    # History + plots + checkpoints (rank 0 only)
    # ------------------------------------------------------------------

    def _update_history(
        self,
        train_metrics: dict,
        val_metrics: dict,
        lr: float,
        whole_image_metrics: Optional[dict] = None,
    ) -> None:
        self.history["train_total"].append(train_metrics["total"])
        self.history["train_dice_loss"].append(train_metrics["dice_loss"])
        self.history["val_total"].append(val_metrics["total"])
        self.history["val_dice_loss"].append(val_metrics["dice_loss"])
        self.history["val_dice_macro"].append(val_metrics["dice"]["macro"])
        self.history["val_iou_macro"].append(val_metrics["iou"]["macro"])
        self.history["val_cldice_macro"].append(val_metrics["cldice"]["macro"])
        self.history["val_length_ratio_skel_macro_mean"].append(
            val_metrics["length"]["skel"]["macro_mean"]
        )
        self.history["val_length_ratio_skel_macro_median"].append(
            val_metrics["length"]["skel"]["macro_median"]
        )
        self.history["val_dice_per_class"].append(val_metrics["dice"]["per_class"])
        self.history["val_iou_per_class"].append(val_metrics["iou"]["per_class"])
        self.history["val_cldice_per_class"].append(val_metrics["cldice"]["per_class"])
        self.history["lr"].append(float(lr))
        if whole_image_metrics is not None:
            self.history["whole_image_eval"].append(whole_image_metrics)

    def _save_history(self) -> None:
        (self.save_dir / "history.json").write_text(json.dumps(self.history, indent=2))

    def _save_plots(self) -> None:
        epochs = list(range(1, len(self.history["train_total"]) + 1))
        if not epochs:
            return
        fig, axes = plt.subplots(1, 4, figsize=(24, 4))

        axes[0].plot(epochs, self.history["train_total"], label="train", marker="o", markersize=3)
        axes[0].plot(epochs, self.history["val_total"], label="val", marker="s", markersize=3)
        axes[0].set_title("total loss"); axes[0].set_xlabel("epoch"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["train_dice_loss"], label="train", marker="o", markersize=3)
        axes[1].plot(epochs, self.history["val_dice_loss"], label="val", marker="s", markersize=3)
        axes[1].set_title("dice loss (macro)"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

        # Per-class val Dice (one line per class) + macro overlay
        per_class = np.asarray(self.history["val_dice_per_class"])  # (E, K)
        if per_class.size:
            for k in range(per_class.shape[1]):
                axes[2].plot(epochs, per_class[:, k], label=self.class_names[k], alpha=0.7, linewidth=1)
        axes[2].plot(
            epochs, self.history["val_dice_macro"],
            label="macro", color="black", linewidth=2, marker="o", markersize=3,
        )
        axes[2].set_title("val Dice (per class + macro)")
        axes[2].set_xlabel("epoch"); axes[2].set_ylim(0, 1); axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=7, ncol=2)

        # Macro IoU / clDice + macro skel length-ratio (separate scale)
        ax3 = axes[3]
        ax3.plot(epochs, self.history["val_iou_macro"], label="iou (macro)", color="C0", marker="s", markersize=3)
        ax3.plot(epochs, self.history["val_cldice_macro"], label="clDice (macro)", color="C1", marker="^", markersize=3)
        ax3.set_ylim(0, 1); ax3.set_ylabel("iou / clDice"); ax3.set_xlabel("epoch"); ax3.grid(True, alpha=0.3)
        ax3.legend(loc="lower right", fontsize=8)
        ax3b = ax3.twinx()
        ax3b.plot(epochs, self.history["val_length_ratio_skel_macro_mean"], label="skel len ratio (macro mean)", color="C3", linestyle="--")
        ax3b.plot(epochs, self.history["val_length_ratio_skel_macro_median"], label="skel len ratio (macro median)", color="C3", linestyle=":")
        ax3b.axhline(1.0, color="k", linewidth=0.5, alpha=0.5)
        ax3b.set_ylim(0, 2.0); ax3b.set_ylabel("length ratio")
        ax3b.legend(loc="upper right", fontsize=8)
        ax3.set_title("val macro metrics")

        fig.tight_layout()
        fig.savefig(self.save_dir / "history.png", dpi=120)
        plt.close(fig)

    def _maybe_save_best_whole_image(
        self,
        epoch: int,
        whole_image_metrics: dict,
        val_metrics: dict,
    ) -> bool:
        """Save ``best_whole_image.pth`` when whole-image macro Dice improves.

        Runs only on rank 0, only when whole-image eval ran this epoch.
        This checkpoint is the *production* best — selected against the
        true full-image evaluation, not the optimistic tile-level metric.
        """
        wi_dice = float(whole_image_metrics["dice_macro"])
        if wi_dice <= self.best_whole_image_dice_macro:
            return False
        self.best_whole_image_dice_macro = wi_dice
        state_dict = (
            self.model.module.state_dict() if hasattr(self.model, "module")
            else self.model.state_dict()
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": state_dict,
                "whole_image_dice_macro": wi_dice,
                "whole_image_iou_macro": float(whole_image_metrics["iou_macro"]),
                "whole_image_cldice_macro": float(whole_image_metrics["cldice_macro"]),
                "tile_val_dice_macro": float(val_metrics["dice"]["macro"]),
                "class_names": self.class_names,
                "model_name": self.config.model_name,
                "encoder_name": self.config.encoder_name,
            },
            self.save_dir / "best_whole_image.pth",
        )
        return True

    def _maybe_save_best(self, epoch: int, val_metrics: dict) -> bool:
        improved = False
        if self.config.best_metric == "dice_macro":
            if val_metrics["dice"]["macro"] > self.best_val_dice_macro:
                self.best_val_dice_macro = val_metrics["dice"]["macro"]
                improved = True
        elif self.config.best_metric == "loss":
            if val_metrics["total"] < self.best_val_loss:
                self.best_val_loss = val_metrics["total"]
                improved = True
        else:
            raise ValueError(f"Unknown best_metric: {self.config.best_metric!r}")
        if improved:
            state_dict = (
                self.model.module.state_dict() if hasattr(self.model, "module")
                else self.model.state_dict()
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": state_dict,
                    "val_dice_macro": val_metrics["dice"]["macro"],
                    "val_iou_macro": val_metrics["iou"]["macro"],
                    "val_total": val_metrics["total"],
                    "class_names": self.class_names,
                    "model_name": self.config.model_name,
                    "encoder_name": self.config.encoder_name,
                },
                self.save_dir / "best.pth",
            )
        return improved

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _format_per_class(self, label: str, per_class: list[float]) -> str:
        parts = [f"{self.class_names[k][:14]}={per_class[k]:.3f}" for k in range(len(per_class))]
        return f"  {label} per-class:  " + "  ".join(parts)

    def _should_run_whole_image_eval(self, epoch: int) -> bool:
        if self.whole_image_evaluator is None:
            return False
        every = max(0, int(self.config.whole_image_eval_every))
        if every <= 0:
            return False
        return (epoch % every == 0) or (epoch == self.config.epochs)

    def run(self) -> None:
        for epoch in range(1, self.config.epochs + 1):
            if hasattr(self.criterion, "set_epoch"):
                self.criterion.set_epoch(epoch)
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            if _is_rank_zero():
                print(f"\n=== epoch {epoch}/{self.config.epochs}   lr={current_lr:.2e} ===")
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.val_epoch(epoch)

            # Optional whole-image evaluation (rank 0 only, every N epochs).
            whole_image_metrics: Optional[dict] = None
            if _is_rank_zero() and self._should_run_whole_image_eval(epoch):
                model_for_eval = self.model.module if hasattr(self.model, "module") else self.model
                whole_image_metrics = self.whole_image_evaluator.evaluate(
                    model_for_eval, epoch=epoch, show_progress=True,
                )

            if _is_rank_zero():
                print(
                    f"  train: total={train_metrics['total']:.4f}  "
                    f"dice_loss={train_metrics['dice_loss']:.4f}"
                )
                print(
                    f"  val:   total={val_metrics['total']:.4f}  "
                    f"dice_loss={val_metrics['dice_loss']:.4f}  "
                    f"dice_macro={val_metrics['dice']['macro']:.4f}  "
                    f"iou_macro={val_metrics['iou']['macro']:.4f}  "
                    f"clDice_macro={val_metrics['cldice']['macro']:.4f}"
                )
                print(self._format_per_class("dice", val_metrics["dice"]["per_class"]))
                skel = val_metrics["length"]["skel"]
                print(
                    f"  len:   skel macro mean={skel['macro_mean']:.3f}  "
                    f"median={skel['macro_median']:.3f}"
                )
                if whole_image_metrics is not None:
                    print(
                        f"  whole-image: dice_macro={whole_image_metrics['dice_macro']:.4f}  "
                        f"iou_macro={whole_image_metrics['iou_macro']:.4f}  "
                        f"clDice_macro={whole_image_metrics['cldice_macro']:.4f}  "
                        f"skel_len_macro_mean={whole_image_metrics['skel_length_ratio_macro_mean']:.3f}"
                    )
                    print(self._format_per_class(
                        "whole-image dice",
                        whole_image_metrics["dice_per_class"],
                    ))

            if _is_rank_zero():
                improved = self._maybe_save_best(epoch, val_metrics)
                if improved:
                    print(f"  saved best → {self.save_dir / 'best.pth'}")
                if whole_image_metrics is not None:
                    wi_improved = self._maybe_save_best_whole_image(
                        epoch, whole_image_metrics, val_metrics
                    )
                    if wi_improved:
                        print(
                            f"  saved best (whole-image) → "
                            f"{self.save_dir / 'best_whole_image.pth'}  "
                            f"wi_dice_macro={whole_image_metrics['dice_macro']:.4f}"
                        )
                state_dict = (
                    self.model.module.state_dict() if hasattr(self.model, "module")
                    else self.model.state_dict()
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": state_dict,
                        "class_names": self.class_names,
                        "model_name": self.config.model_name,
                        "encoder_name": self.config.encoder_name,
                    },
                    self.save_dir / "last.pth",
                )
                self._update_history(train_metrics, val_metrics, lr=current_lr,
                                     whole_image_metrics=whole_image_metrics)
                self._save_history()
                self._save_plots()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            if _ddp_active():
                dist.barrier()
