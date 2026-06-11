# AGENTS.md — Working Rules for AI Agents on `irrigation_line_detection`

This repository is developed under a strict **edit-local / run-remote** workflow
because the GPU server is shared by multiple colleagues under a single Linux
account. These rules apply to **every** AI agent (Cursor, Claude, GPT, etc.)
operating on this project. Follow them exactly.

---

## 1. Where code is edited vs. where code is run

| Action                    | Where                                                                 |
| ------------------------- | --------------------------------------------------------------------- |
| Read / edit code          | **Local MacBook only** (this workspace)                               |
| Commit & push to GitHub   | **Local MacBook only**                                                |
| Pull latest code          | **Remote server**, via `git pull`                                     |
| Run training / evaluation | **Remote server**, on its GPUs                                        |
| Inspect logs / runs       | **Remote server** (read-only is fine; do not edit source there)       |

### Hard rules

- **NEVER edit source files on the remote server.** The server's Linux account
  is shared with colleagues. Any local change there will be lost on the next
  `git pull` or, worse, will corrupt someone else's state.
- **NEVER `git commit` from the server.** Pushing from the server would mix
  identities and bypass code review on the developer's machine.
- **NEVER `git push --force`** unless explicitly approved by the user, and
  never against `main`.
- All code changes flow in **one direction only**:
  `MacBook edit → git commit → git push → server git pull → server run`.

---

## 2. Remote server details

- **Hostname / SSH alias**: `bobyard-server-6000`
  - Connect with: `ssh bobyard-server-6000`
  - Assume the SSH config on the MacBook is already set up; do not modify it.
- **GPUs**: 2 × NVIDIA RTX 6000 Ada
  - Use `CUDA_VISIBLE_DEVICES=0` or `=1` to pin to a single GPU when others
    are using the box.
  - Check availability with `nvidia-smi` before launching big jobs.
- **Dataset path on server**:
  `/home/rtx6000/james/datasets/poly-irrigation.v6-v2_w_boboflow.yolo26`
- **Code repository path on server**:
  `/home/rtx6000/james/irrigation_line_detection`
- **Local dataset path (this Mac, read-only reference for analysis):**
  `/Users/james.peng/Desktop/Irrigation/datasets/poly-irrigation.v6-v2_w_boboflow.yolo26`

Path differences between Mac and server mean any code that touches the
dataset **must** read paths from a config file, env var, or CLI argument —
never hard-code an absolute path.

---

## 3. Standard development loop

For every change an agent makes, the workflow is:

1. **Edit locally** in this workspace (MacBook).
2. **Test locally** for anything that does not require a GPU
   (data loading, config parsing, lint, unit tests, small CPU smoke tests).
3. **Commit + push** from the MacBook:
   ```bash
   git add -A
   git commit -m "<concise message>"
   git push
   ```
4. **Pull on the server** (the user, or the agent if explicitly running an SSH
   step, will do this):
   ```bash
   ssh bobyard-server-6000
   cd /home/rtx6000/james/irrigation_line_detection
   git pull --ff-only
   ```
5. **Run on the server** (training / eval / long jobs).
6. **Inspect outputs / logs on the server**, copy artifacts back to the
   MacBook only if needed.

If an agent finds itself wanting to "just fix this one thing on the server,"
that is a signal to stop and instead go back to step 1 on the MacBook.

---

## 4. Things agents must not check into git

Datasets, model weights, run outputs, virtual environments, and editor caches
are all excluded by `.gitignore`. Specifically, do not commit:

- `datasets/`, `data/`, raw images, raw labels.
- Trained weights: `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.engine`,
  `*.safetensors`, …
- Run artifacts: `runs/`, `wandb/`, `mlruns/`, `lightning_logs/`,
  `tensorboard/`, `checkpoints/`, …
- Environments: `.venv/`, `venv/`, `conda-env/`, `.env`, …
- OS / editor cruft: `.DS_Store`, `.idea/`, `.vscode/`, `__pycache__/`, …
- Anything > ~10 MB unless explicitly approved.

If a config or small artifact really needs to be tracked, add an explicit
allow-rule (e.g. `!configs/**/*.yaml`) rather than relaxing the global ignore.

---

## 5. Reproducibility expectations

- Pin Python dependency versions in `requirements.txt` (or `pyproject.toml`).
- Make randomness controllable: expose `seed` in training configs.
- Log the exact dataset version (e.g. `poly-irrigation.v6-v2_w_boboflow.yolo26`)
  and the git commit SHA in every training run.
- Prefer config files (YAML) over hard-coded constants for paths, classes,
  hyperparameters.

---

## 6. Conventions agents should follow

- **Branches**: small, focused branches off `main`; descriptive names like
  `feat/yolo-baseline`, `fix/dataloader-empty-labels`.
- **Commits**: imperative, concise; explain *why* if non-obvious.
- **Scripts**: any new training / eval script must accept a `--config` (or
  equivalent) and must not hard-code Mac- or server-specific paths.
- **Tests / smoke checks**: include a CPU-only smoke test where feasible so
  the MacBook can validate basic correctness before pushing.

---

## 7. Quick reference

```bash
# Local (MacBook)
cd /Users/james.peng/Desktop/Irrigation/irrigation_line_detection
# ... edit ...
git add -A && git commit -m "..." && git push

# Remote (RTX 6000 server)
ssh bobyard-server-6000
cd /home/rtx6000/james/irrigation_line_detection
git pull --ff-only
# ... run training / eval ...
```

**Golden rule:** if you're about to type `vim`, `nano`, or any editor command
on `bobyard-server-6000`, stop. Go back to the MacBook, edit there, commit,
push, then `git pull` on the server.

---

## 8. Code conventions specific to the training stack

- **Multi-label, not multi-class softmax.** The model emits ``(B, K, H, W)``
  logits with one independent sigmoid per class. Pixels at line crossings
  legitimately belong to two classes; softmax would force a choice.
- **All paths use `pathlib.Path`.** No `os.path.join`, no hard-coded absolute
  paths in any committed code — read paths from `configs/*.yaml` or CLI.
- **`from __future__ import annotations`** at the top of every `.py` file so
  3.10-style union types don't break Python 3.9 imports locally (system
  Python on macOS is 3.9; the 6000 server is 3.10; the 5090 server is 3.13).
- **Augmentation policy:** thin lines (~4 px) are sensitive — only h-flip,
  v-flip, and 90° rotations. **Do not** introduce elastic, arbitrary
  rotation, perspective, or heavy color jitter without a Dice benchmark.
- **Loss recipe:** per-channel BCE + macro-Dice is the Phase-1 default.
  clDice / Lovász code lives in `training/losses.py` but is disabled by
  default — the v3 ladder in `lateral_detection` showed they add ≤ 0.0018
  Dice on this task. Re-enable only behind a config overlay + benchmark.
- **Best-checkpoint metric:** **macro-Dice** across the 6 classes (forces the
  rare classes — `drip`, `main_1` — to actually be learned). Whole-image
  Dice is logged in parallel but does not select the checkpoint (it only
  runs every N epochs).
- **SyncBN under DDP:** opt-in; on by default. Override to `false` for any
  encoder with many small BN layers (EfficientNet's depthwise + SE blocks,
  MobileNet, etc.) — `lateral_detection` lost ~20 Dice points to SyncBN+EffB3.

## 9. Dataset pipeline reproducibility

The training pipeline reads the *merged* COCO produced by
`scripts/remap_classes.py`, NOT the raw Roboflow export. The class set is
the 6-class merge defined in `configs/class_remap.yaml`:

```
1: lateral_solid_0   2: sleeves           3: lateral_other_0
4: main_0            5: drip              6: main_1
```

The trainer reads class names directly from the merged COCO's
`categories:` block — there is **no** parallel class list in `configs/`
that could drift out of sync.

Local + server pipeline (run after any change to `class_remap.yaml`):

```bash
python scripts/remap_classes.py \
    --src ../datasets/poly-irrigation.v6-v2_w_boboflow.coco \
    --dst ../datasets/poly-irrigation.v6-v2_w_boboflow.coco.merged \
    --rules configs/class_remap.yaml --overwrite
```
