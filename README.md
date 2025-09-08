Simpsons Transfer Learning (EfficientNet-B0)
===========================================

This project trains and evaluates an image classifier for Simpsons characters using PyTorch and transfer learning on EfficientNet‑B0. The entrypoint `main.py` orchestrates training and local-folder inference across multiple fine‑tuning modes. Training implementation lives in `train_simpsons_optimized.py`.

### Quick start (inference-only, 1–2 minutes)
- Ensure Python 3.12 and uv are installed.
- Install deps: `uv sync`
- Add a few test images to `data/inference/input/` (flat folder of images).
- Run: `uv run python main.py`
  - By default, it runs inference for each mode (`linear_probe`, `last2_blocks`, `full`) using the corresponding best checkpoints under `models/` and writes results to `data/inference/output/<mode>/`.
  - An HTML gallery is auto-generated at `data/inference/output/<mode>/index.html`.

### Train the model
- Optional but compute‑intensive. Enable training and optionally inference in the same run:
  - `RUN_TRAINING=1 RUN_INFERENCE=1 uv run python main.py`
- The first run attempts to download the Kaggle Simpsons dataset via `kagglehub` and prepares a 90/10 stratified split into `data/processed/train` and `data/processed/val`.
- If `kagglehub` is not available, place the dataset manually (see Data layout).

### Requirements
- Python 3.12
- uv (for environment and reproducible installs)
- Optional: GPU (CUDA) or Apple Silicon (MPS) for faster training/inference

### Data layout (for training)
- Place the dataset under `data/processed/`:
  - `data/processed/train/<class_name>/*.jpg`
  - `data/processed/val/<class_name>/*.jpg`
  Each class is a folder containing its images. The `tools/prepare_data.py` helper will create this split from the Kaggle dataset when possible and also seed `data/inference/input/` with Kaggle test images for quick local inference.

### What gets produced
- Under `models/`:
  - Best model per mode: `simpsons_effb0_best_v2_<mode>.pt`
  - Confusion matrix: `confusion_matrix_<mode>.npy`
  - Class index mapping: `class_to_idx.json`
  - Logs: `training_<mode>.log`
- Under `data/inference/output/<mode>/`:
  - `predictions.csv` and per‑image `*.json` files
  - `index.html` gallery (Tailwind, generated automatically)

### Environment variables in main.py
Use these to customize behavior without editing code. Types and defaults in parentheses.

- Paths
  - `DATA_DIR` (path, `data/processed`): root for `train/` and `val/`.
  - `MODELS_DIR` (path, `models`): where models and logs are written.
  - `INFER_INPUT_DIR` (path, `data/inference/input`): folder of images to infer.
  - `INFER_OUTPUT_DIR` (path, `data/inference/output`): where inference results are written.
- General
  - `SEED` (int, `42`): RNG seed.
  - `BATCH_SIZE` (int, `32`): batch size for train/val.
  - `NUM_WORKERS` (int, `4`): dataloader workers.
  - `RESIZE` (int, `256`): resize shorter side before crop.
  - `IMG_SIZE` (int, `224`): center/random crop size.
  - `USE_COLOR_JITTER` (bool, `true`): enable color jitter in training transforms.
- Training schedule
  - `EPOCHS` (int, `2`): max epochs.
  - `EARLY_STOP` (int, `5`): early stopping patience (epochs without val acc improvement).
  - `FINETUNE_MODES` (csv, `linear_probe,last2_blocks,full`): which modes to run.
- Execution toggles
  - `RUN_TRAINING` (bool, `false`): enable training.
  - `RUN_INFERENCE` (bool, `true`): enable inference.
- Optimization
  - `LR_BACKBONE` (float, `3e-4`): learning rate for unfrozen backbone params.
  - `LR_HEAD` (float, `1e-3`): learning rate for classifier head.
  - `WEIGHT_DECAY` (float, `1e-4`): AdamW weight decay.
  - `ROP_FACTOR` (float, `0.5`): ReduceLROnPlateau factor.
  - `ROP_PATIENCE` (int, `2`): ReduceLROnPlateau patience.
- Filenames (base names; per‑mode suffixes are added by `main.py`)
  - `BEST_MODEL` (str, `simpsons_effb0_best_v2.pt`)
  - `CM_FILE` (str, `confusion_matrix.npy`)

Notes on boolean envs: use `1/true/yes/on` to enable.

### Model details
- Backbone: torchvision EfficientNet‑B0 with `EfficientNet_B0_Weights.IMAGENET1K_V1` pre‑training.
- Head: final Linear layer replaced with `num_classes` outputs (matches dataset classes).
- Fine‑tuning modes (controlled by `FINETUNE_MODES`):
  - `linear_probe`: freeze all except the classifier head.
  - `last2_blocks`: unfreeze the last two EfficientNet feature blocks and classifier.
  - `full`: unfreeze the entire backbone and classifier.
- Optimizer & schedule: AdamW with parameter groups (`LR_BACKBONE`, `LR_HEAD`), ReduceLROnPlateau on val loss, early stopping via `EARLY_STOP`.
- Transforms: train uses Resize(RESIZE) → RandomCrop(IMG_SIZE) → RandomHorizontalFlip → optional ColorJitter → Normalize(ImageNet mean/std). Val uses Resize → CenterCrop → Normalize.
- Mixed precision & memory format: AMP on CUDA/MPS, channels‑last on CUDA/MPS, device auto‑selects CUDA → MPS → CPU.
- Metrics & artifacts: best model by val acc, classification report, confusion matrix saved per mode.

### HTML Overview (optional CLI)
`main.py` auto‑generates `index.html` per mode after inference. You can also regenerate manually:
- `python tools/generate_overview.py --subfolder linear_probe`
- Optional: `--title "Inference Overview — linear_probe"` and `--open`

### Tips
- Training on CPU is slow; prefer GPU (CUDA) or Apple Silicon (MPS).
- `data/**` and `models/**` are ignored by Git to avoid committing large files.
