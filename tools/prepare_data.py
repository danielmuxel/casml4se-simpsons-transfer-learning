"""
Dataset preparation utility.

Behavior:
- If data/processed/train and data/processed/val already contain images, does nothing.
- Otherwise:
  - Downloads the Kaggle Simpsons dataset via kagglehub (if available)
  - Copies the Kaggle testset into data/inference/input for local inference
  - Creates a stratified 90/10 split from simpsons_dataset into
    data/processed/train and data/processed/val
  - Writes data/processed/classes.json

This avoids scikit-learn dependency by using a simple random per-class split.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
import json
import os
import random
import shutil


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
INFER_INPUT_DIR = PROJECT_ROOT / "data" / "inference" / "input"


def _dir_has_images(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    try:
        for sub in root.iterdir():
            if sub.is_dir():
                for p in sub.glob("*.jpg"):
                    return True
                for p in sub.glob("*.jpeg"):
                    return True
                for p in sub.glob("*.png"):
                    return True
    except Exception:
        pass
    return False


def _safe_copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        # If destination has content, skip copy
        has_any = any(dst.iterdir())
        if has_any:
            print(f"Skip copying: destination not empty: {dst}")
            return
        # If empty, remove to allow copytree
        try:
            dst.rmdir()
        except Exception:
            pass
    print(f"Copying folder: {src} -> {dst} ...")
    shutil.copytree(src, dst)
    print(f"Finished copying folder: {dst}")


def _download_kaggle_dataset() -> Path | None:
    try:
        import kagglehub  # type: ignore
    except Exception:
        print("kagglehub not available; skipping dataset download.")
        return None
    try:
        base = Path(kagglehub.dataset_download("alexattia/the-simpsons-characters-dataset"))
        return base
    except Exception as e:
        print(f"Failed to download Kaggle dataset: {e}")
        return None


def _resolve_kaggle_paths(base: Path) -> Tuple[Path | None, Path | None]:
    data_candidates = [
        base / "simpsons_dataset",
    ]
    test_candidates = [
        base / "kaggle_simpson_testset" / "kaggle_simpson_testset",
        base / "kaggle_simpson_testset",
    ]
    data_dir = next((p for p in data_candidates if p.exists()), None)
    test_dir = next((p for p in test_candidates if p.exists()), None)
    return data_dir, test_dir


def _prepare_split(data_dir: Path, train_dir: Path, val_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> List[str]:
    random.seed(seed)
    classes: List[str] = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    print(f"Starting stratified split copy into train/val for {len(classes)} classes ...")
    total_train = 0
    total_val = 0
    for cls in classes:
        images = sorted((data_dir / cls).glob("*.jpg"))
        if not images:
            continue
        n = len(images)
        n_val = max(1, int(n * val_ratio))
        random.shuffle(images)
        val_images = images[:n_val]
        train_images = images[n_val:]

        (train_dir / cls).mkdir(parents=True, exist_ok=True)
        (val_dir / cls).mkdir(parents=True, exist_ok=True)
        for p in train_images:
            shutil.copy2(p, train_dir / cls / p.name)
            total_train += 1
        for p in val_images:
            shutil.copy2(p, val_dir / cls / p.name)
            total_val += 1

    print(f"Finished split copy. Train images: {total_train}, Val images: {total_val}")
    return classes


def prepare_data() -> None:
    # If already prepared, skip
    if _dir_has_images(TRAIN_DIR) and _dir_has_images(VAL_DIR):
        print("Data already prepared; skipping download/split.")
        return

    # Ensure directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    INFER_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Download dataset
    base = _download_kaggle_dataset()
    if base is None:
        # If we cannot download, leave as-is; user can place data manually
        print("No dataset available to prepare. Please install kagglehub or place data manually.")
        return

    data_dir, test_dir = _resolve_kaggle_paths(base)
    if data_dir is None:
        print("Could not locate simpsons_dataset in Kaggle download; aborting preparation.")
        return

    # Copy test set into inference input for convenience
    if test_dir and test_dir.exists():
        print(f"Preparing inference input from Kaggle testset ...")
        _safe_copytree(test_dir, INFER_INPUT_DIR)

    # Perform split
    classes = _prepare_split(data_dir, TRAIN_DIR, VAL_DIR, val_ratio=0.1, seed=42)

    # Write classes.json
    classes_path = PROCESSED_DIR / "classes.json"
    with classes_path.open("w") as f:
        json.dump(classes, f, indent=2)
    print(f"Prepared data. Train: {TRAIN_DIR}, Val: {VAL_DIR}, Classes: {len(classes)}")


if __name__ == "__main__":
    prepare_data()


