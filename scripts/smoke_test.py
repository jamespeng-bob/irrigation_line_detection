"""End-to-end CPU smoke test for the multi-class training pipeline.

Walks one tile through every layer without launching a real training run:

1. ``data.coco_loader.load_split`` — parses the merged COCO and reports the
   class mapping.
2. ``data.dataset.TileDataset`` — emits a ``(3, T, T)`` image and
   ``(K, T, T)`` mask tile pair.
3. ``data.augmentation.TileAugmenter`` — applied on the raw numpy arrays.
4. ``models.unet.SMPUnet`` — forward pass producing ``(B, K, T, T)`` logits.
5. ``training.losses.BCEDiceLoss`` — backward pass.
6. ``training.metrics.PerClass*Accumulator`` — one update + compute.
7. ``inference.whole_image_eval.WholeImageEvaluator`` — initialized but
   *not* run (a real eval needs minutes per image).

Run on Mac before pushing. CPU-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.augmentation import TileAugmenter
from data.coco_loader import load_split
from data.dataset import TileDataset, collate_tile_samples
from inference.whole_image_eval import WholeImageEvalConfig, WholeImageEvaluator
from models.unet import build_model
from training.losses import build_loss
from training.metrics import (
    PerClassClDiceAccumulator,
    PerClassDiceAccumulator,
    PerClassIoUAccumulator,
    PerClassLengthRatioAccumulator,
)


def main() -> int:
    base_cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    train_cfg = yaml.safe_load(open(ROOT / "configs/train.yaml"))
    cfg = {**base_cfg, **train_cfg}
    for k, v in train_cfg.items():
        if isinstance(v, dict) and isinstance(base_cfg.get(k), dict):
            cfg[k] = {**base_cfg[k], **v}

    dataset_root = (Path.cwd() / cfg["data"]["dataset_root"]).resolve()
    train_dir = dataset_root / cfg["data"]["train_dir"]
    valid_dir = dataset_root / cfg["data"]["valid_dir"]

    print(f"[smoke] dataset_root = {dataset_root}")
    images, polylines_by_image, class_names, _cat_id_to_channel, stats = load_split(train_dir)
    print(f"[smoke] coco_loader stats: {stats}")
    print(f"[smoke] class_names = {class_names}")
    K = len(class_names)

    aug = TileAugmenter(
        hflip_prob=cfg["augmentation"]["hflip_prob"],
        vflip_prob=cfg["augmentation"]["vflip_prob"],
        rotate_90_prob=cfg["augmentation"]["rotate_90_prob"],
        seed=0,
    )

    norm = cfg["normalization"]
    ds = TileDataset(
        split_dir=train_dir,
        num_classes=K,
        tile_size=cfg["data"]["tile_size"],
        stride=cfg["data"]["stride"],
        mode="random",
        thickness=cfg["rasterize"]["thickness"],
        augmenter=aug,
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
        samples_per_epoch_per_image=1,
        seed=0,
    )
    print(f"[smoke] train tiles = {len(ds)}")

    sample = ds[0]
    print(f"[smoke] tile: image {tuple(sample.image.shape)} dtype={sample.image.dtype}  "
          f"mask {tuple(sample.mask.shape)} dtype={sample.mask.dtype}")
    print(f"[smoke] mask per-class fg pixels = "
          f"{[int(sample.mask[k].sum()) for k in range(K)]}")

    # Batch of 2 tiles through collate
    batch = collate_tile_samples([ds[0], ds[1]])
    print(f"[smoke] batch image {tuple(batch['image'].shape)}  "
          f"mask {tuple(batch['mask'].shape)}")

    # Build model + loss. Use encoder_weights=None for the smoke test so we
    # don't try to download ImageNet weights (real runs override this and
    # rely on the on-disk torch hub cache).
    model_cfg = dict(cfg["model"])
    model_cfg["encoder_weights"] = None
    model = build_model(model_cfg, num_classes=K)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[smoke] model {cfg['model']['name']}/{cfg['model']['encoder']}  "
          f"trainable params = {n_params/1e6:.1f}M")

    # Tiny tile (256²) to keep the smoke test under a few seconds on CPU.
    small_image = batch["image"][:, :, :256, :256]
    small_mask = batch["mask"][:, :, :256, :256]
    logits = model(small_image)
    print(f"[smoke] logits {tuple(logits.shape)}")
    assert logits.shape[1] == K, f"output channels {logits.shape[1]} != K={K}"

    criterion = build_loss(cfg["loss"])
    losses = criterion(logits, small_mask)
    print(f"[smoke] loss keys = {list(losses.keys())}  "
          f"total = {float(losses['total'].detach()):.4f}  "
          f"dice = {float(losses['dice'].detach()):.4f}")
    losses["total"].backward()
    print(f"[smoke] backward OK")

    # Metric accumulators
    for cls_acc in (PerClassDiceAccumulator(K), PerClassIoUAccumulator(K)):
        cls_acc.update(logits.detach(), small_mask)
        out = cls_acc.compute()
        print(f"[smoke] {type(cls_acc).__name__}: macro={out['macro']:.4f} "
              f"per_class[:3]={[round(v, 4) for v in out['per_class'][:3]]}")
    cl = PerClassClDiceAccumulator(K)
    cl.update(logits.detach(), small_mask)
    print(f"[smoke] PerClassClDiceAccumulator: macro={cl.compute()['macro']:.4f}")
    lr = PerClassLengthRatioAccumulator(K)
    lr.update(logits.detach(), small_mask)
    lr_out = lr.compute()
    print(f"[smoke] PerClassLengthRatioAccumulator: skel macro_mean={lr_out['skel']['macro_mean']}")

    # Whole-image evaluator: instantiate only (running it would take ~30 min).
    eval_cfg = WholeImageEvalConfig(
        valid_dir=valid_dir,
        num_classes=K,
        class_names=class_names,
        tile_size=cfg["data"]["tile_size"],
        stride=cfg["data"]["stride"],
        thickness=cfg["rasterize"]["thickness"],
        device="cpu",
        mean=tuple(norm["mean"]),
        std=tuple(norm["std"]),
        batch_size=2,
        max_images=1,
    )
    evaluator = WholeImageEvaluator(eval_cfg)
    print(f"[smoke] WholeImageEvaluator built on {len(evaluator.images)} valid images")

    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
