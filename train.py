"""train.py — multi-class irrigation-line segmentation trainer entrypoint.

Single-GPU and DDP both supported through the same script.

Single-GPU
----------
    python train.py
    python train.py --config configs/train.yaml --device cuda:0
    python train.py --overlay configs/train_v1.yaml --device cuda:0

DDP (across N GPUs on one host)
-------------------------------
    torchrun --nproc-per-node=2 --master-port=29500 train.py \
        --overlay configs/train_v1.yaml

``training.batch_size`` is PER-GPU under DDP; effective global batch
= ``batch_size * world_size``.

Config layering
---------------
``--base-config`` → ``--config`` → ``--overlay``. Later layers override
earlier ones; each layer can omit any field it doesn't change.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow the cached allocator to grow segments on demand. Worth 1–3 GB of
# usable VRAM at dense U-Net activations and 1024² tiles. Must be set
# BEFORE the first ``import torch`` allocator interaction.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentation import TileAugmenter
from data.coco_loader import load_split
from data.dataset import TileDataset, collate_tile_samples, worker_init_fn
from inference.whole_image_eval import WholeImageEvaluator, WholeImageEvalConfig
from models.unet import build_model
from training.losses import build_loss
from training.trainer import SegTrainer, TrainerConfig


# ---------------------------------------------------------------------------
# Config layering
# ---------------------------------------------------------------------------


def merge_configs(base: dict, override: dict) -> dict:
    """Shallow recursive merge: nested dicts are merged one level deep."""
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
# Device + DDP setup
# ---------------------------------------------------------------------------


def _torchrun_launched() -> bool:
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def setup_ddp() -> tuple[int, int, int, str]:
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
        device = f"cuda:{local_rank}"
    else:
        backend = "gloo"
        device = "cpu"
    dist.init_process_group(backend=backend)
    return local_rank, dist.get_rank(), dist.get_world_size(), device


def _resolve_device(name: str) -> str:
    if name.startswith("cuda") and not torch.cuda.is_available():
        print(f"[train] CUDA not available; falling back from {name!r} to 'cpu'.")
        return "cpu"
    if name == "mps":
        mps_ok = (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        )
        if not mps_ok:
            print("[train] MPS not available; falling back to 'cpu'.")
            return "cpu"
    return name


# ---------------------------------------------------------------------------
# Dataset / loader construction
# ---------------------------------------------------------------------------


def _resolve_path(p: str) -> Path:
    """Resolve relative paths against CWD, leave absolute paths intact."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (Path.cwd() / pp).resolve()


def build_datasets(cfg: dict) -> tuple[TileDataset, TileDataset, list[str]]:
    dataset_root = _resolve_path(cfg["data"]["dataset_root"])
    train_dir = dataset_root / cfg["data"]["train_dir"]
    val_dir = dataset_root / cfg["data"]["valid_dir"]
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Train dir not found: {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Val dir not found: {val_dir}")

    # Resolve class list once from the training split's COCO file so the
    # config does not have to duplicate it (and cannot drift from disk).
    _, _, class_names, _, _stats = load_split(train_dir)
    num_classes = len(class_names)

    aug_cfg = cfg.get("augmentation", {})
    aug = (
        TileAugmenter(
            hflip_prob=float(aug_cfg.get("hflip_prob", 0.5)),
            vflip_prob=float(aug_cfg.get("vflip_prob", 0.5)),
            rotate_90_prob=float(aug_cfg.get("rotate_90_prob", 0.5)),
        )
        if aug_cfg.get("enabled", True)
        else None
    )

    norm = cfg["normalization"]
    common_kwargs = dict(
        num_classes=num_classes,
        tile_size=int(cfg["data"]["tile_size"]),
        stride=int(cfg["data"]["stride"]),
        thickness=int(cfg["rasterize"]["thickness"]),
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
    )

    train_ds = TileDataset(
        split_dir=train_dir,
        mode=cfg["data"]["train_mode"],
        augmenter=aug,
        samples_per_epoch_per_image=int(
            cfg["data"].get("samples_per_epoch_per_image", 8)
        ),
        **common_kwargs,
    )
    val_ds = TileDataset(
        split_dir=val_dir,
        mode=cfg["data"]["val_mode"],
        augmenter=None,
        **common_kwargs,
    )
    return train_ds, val_ds, class_names


def build_loaders(
    train_ds: TileDataset,
    val_ds: TileDataset,
    cfg: dict,
    device: str,
    world_size: int,
    global_rank: int,
) -> tuple[DataLoader, DataLoader]:
    bs = int(cfg["training"]["batch_size"])
    nw = int(cfg["training"]["num_workers"])
    persistent = nw > 0
    pin_memory = device != "cpu"

    if world_size > 1:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=global_rank, shuffle=True,
            drop_last=False,
        )
        val_sampler = DistributedSampler(
            val_ds, num_replicas=world_size, rank=global_rank, shuffle=False,
            drop_last=False,
        )
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=bs, shuffle=train_shuffle, sampler=train_sampler,
        num_workers=nw, pin_memory=pin_memory,
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False, sampler=val_sampler,
        num_workers=nw, pin_memory=pin_memory,
        collate_fn=collate_tile_samples,
        worker_init_fn=worker_init_fn,
        persistent_workers=persistent,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Irrigation-line segmentation trainer.")
    parser.add_argument("--base-config", default="configs/base.yaml")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--overlay",
        default=None,
        help="Optional extra config layer applied on top of --config.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override training.device (cuda:0, cpu, mps). Ignored under DDP.",
    )
    parser.add_argument("--save-dir", default=None, help="Override training.save_dir.")
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.base_config))
    train_cfg = yaml.safe_load(open(args.config))
    cfg = merge_configs(base_cfg, train_cfg)
    if args.overlay is not None:
        overlay_cfg = yaml.safe_load(open(args.overlay))
        cfg = merge_configs(cfg, overlay_cfg)
    if args.save_dir is not None:
        cfg["training"]["save_dir"] = args.save_dir

    # ── Device / DDP setup ───────────────────────────────────────────────
    if _torchrun_launched():
        local_rank, global_rank, world_size, device = setup_ddp()
    else:
        device = _resolve_device(args.device or cfg["training"].get("device", "cuda"))
        local_rank, global_rank, world_size = 0, 0, 1

    is_rank_zero = global_rank == 0

    # ── Datasets + loaders ───────────────────────────────────────────────
    train_ds, val_ds, class_names = build_datasets(cfg)
    train_loader, val_loader = build_loaders(
        train_ds, val_ds, cfg, device=device,
        world_size=world_size, global_rank=global_rank,
    )
    if is_rank_zero:
        per_gpu = int(cfg["training"]["batch_size"])
        eff_bs = per_gpu * world_size
        print(f"[train] world_size={world_size}  per-GPU bs={per_gpu}  effective bs={eff_bs}")
        print(f"[train] num_classes={len(class_names)}  classes={class_names}")
        print(f"[train] train tiles: {len(train_ds)},  val tiles: {len(val_ds)}")

    # ── Model + loss ─────────────────────────────────────────────────────
    model = build_model(cfg["model"], num_classes=len(class_names)).to(device)

    if world_size > 1:
        use_sync_bn = bool(cfg["training"].get("sync_batch_norm", True))
        if use_sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if is_rank_zero:
            print(f"[train] DDP: sync_batch_norm={use_sync_bn}")
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=False,
        )

    criterion = build_loss(cfg["loss"])

    # ── Trainer config ──────────────────────────────────────────────────
    trainer_cfg = TrainerConfig(
        save_dir=cfg["training"]["save_dir"],
        num_classes=len(class_names),
        class_names=class_names,
        epochs=int(cfg["training"]["epochs"]),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
        grad_clip=float(cfg["training"]["grad_clip"]),
        log_interval=int(cfg["training"]["log_interval"]),
        val_viz_count=int(cfg["training"]["val_viz_count"]),
        device=device,
        best_metric=str(cfg["training"]["best_metric"]),
        lr_schedule=str(cfg["training"].get("lr_schedule", "constant")),
        warmup_epochs=int(cfg["training"].get("warmup_epochs", 0)),
        cosine_min_lr_ratio=float(cfg["training"].get("cosine_min_lr_ratio", 0.01)),
        whole_image_eval_every=int(cfg["training"].get("whole_image_eval_every", 10)),
    )

    # ── Whole-image evaluator (rank 0 only) ──────────────────────────────
    whole_image_evaluator = None
    if is_rank_zero and trainer_cfg.whole_image_eval_every > 0:
        eval_cfg = WholeImageEvalConfig(
            valid_dir=_resolve_path(cfg["data"]["dataset_root"]) / cfg["data"]["valid_dir"],
            num_classes=len(class_names),
            class_names=class_names,
            tile_size=int(cfg["data"]["tile_size"]),
            stride=int(cfg["data"]["stride"]),
            thickness=int(cfg["rasterize"]["thickness"]),
            device=device,
            mean=tuple(cfg["normalization"]["mean"]),
            std=tuple(cfg["normalization"]["std"]),
            batch_size=int(cfg["training"].get("whole_image_eval_batch_size", 4)),
        )
        whole_image_evaluator = WholeImageEvaluator(eval_cfg)

    if is_rank_zero:
        print(
            f"[train] device={device}  model={cfg['model']['name']}/{cfg['model']['encoder']}  "
            f"loss={cfg['loss']['name']}  save_dir={trainer_cfg.save_dir}  "
            f"whole_image_eval_every={trainer_cfg.whole_image_eval_every}"
        )

    trainer = SegTrainer(
        model=model,
        criterion=criterion,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_cfg,
        normalization=(cfg["normalization"]["mean"], cfg["normalization"]["std"]),
        whole_image_evaluator=whole_image_evaluator,
    )
    trainer.run()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
