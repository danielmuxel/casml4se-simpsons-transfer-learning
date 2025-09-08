Simpsons Transfer Learning (EfficientNet-B0)
===========================================

This repository trains an image classifier for Simpsons characters using PyTorch and transfer learning on EfficientNet-B0. The entrypoint `main.py` orchestrates multiple fine-tuning modes and writes trained models and metrics to the `models/` directory. The training implementation lives in `train_simpsons_optimized.py`.

Requirements
- Python 3.12
- uv (for environment and reproducible installs)
- Optional: GPU (CUDA) or Apple Silicon (MPS) for faster training

Data Layout
- Place your dataset under `data/processed/` with the following structure:
  - `data/processed/train/<class_name>/*.jpg`
  - `data/processed/val/<class_name>/*.jpg`
  Each class is a folder containing its images.

Setup and Run (uv)
1) Install dependencies:
   - `uv sync`
2) Run training:
   - `uv run python main.py`

Outputs
- Trained model checkpoints and artifacts are saved in `models/`, including:
  - Best model per mode: `simpsons_effb0_best_v2_<mode>.pt`
  - Confusion matrix: `confusion_matrix_<mode>.npy`
  - Class index mapping: `class_to_idx.json`
  - Logs: `training_<mode>.log`

Notes
- Training is compute-intensive and may take a while on CPU.
- `data/**` and `models/**` are ignored by Git to avoid committing large files.

HTML Overview Generator
-----------------------

Generate a simple Tailwind-based HTML gallery for inference outputs.

Prerequisites:
- Inference results exist under `data/inference/output/<subfolder>/` as pairs of images and JSON files (same basename). Example: `linear_probe/foo.jpg` and `linear_probe/foo.jpg.json`.

Usage:
- Generate an overview for a subfolder (e.g., `linear_probe`):
  - `python tools/generate_overview.py --subfolder linear_probe`
- Optional custom page title:
  - `python tools/generate_overview.py --subfolder linear_probe --title "Inference Overview — linear_probe"`
- Auto-open in default browser after generation:
  - `python tools/generate_overview.py --subfolder linear_probe --open`

What it does:
- Scans `data/inference/output/<subfolder>` for `*.json` and their matching images.
- Creates `index.html` and `manifest.json` in that folder.
- `index.html` uses Tailwind CDN and shows each image with `prediction_index`, `prediction_label`, and `prediction_probability`.
- You can click “Reload” to refresh the page, or “Refresh” to re-render the grid from `manifest.json` (re-run the generator to update the manifest when files change).
