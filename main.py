from pathlib import Path
from datetime import datetime

from train_simpsons_optimized import TrainingConfig, train_from_config

# -----------------------------
# Simple training variables
# -----------------------------
DATA_DIR = Path("data/processed")
MODELS_DIR = Path("models")

SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
RESIZE = 256
IMG_SIZE = 224
USE_COLOR_JITTER = True

EPOCHS = 10
EARLY_STOP = 5
FINETUNE_MODES = ["linear_probe", "last2_blocks", "full"]  # three configurations

LR_BACKBONE = 3e-4
LR_HEAD = 1e-3
WEIGHT_DECAY = 1e-4
ROP_FACTOR = 0.5
ROP_PATIENCE = 2

BEST_MODEL = "simpsons_effb0_best_v2.pt"  # base name; per-mode suffixes will be added
CM_FILE = "confusion_matrix.npy"         # base name; per-mode suffixes will be added


def main() -> None:
    total_start = datetime.now()
    print(f"\n=== Overall run start: {total_start.isoformat()} ===")

    for mode in FINETUNE_MODES:
        mode_start = datetime.now()
        print(f"\n=== Training start ({mode}): {mode_start.isoformat()} ===")
        cfg = TrainingConfig(
            data_dir=DATA_DIR,
            models_dir=MODELS_DIR,
            seed=SEED,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            resize_size=RESIZE,
            image_size=IMG_SIZE,
            use_color_jitter=USE_COLOR_JITTER,
            epochs=EPOCHS,
            early_stop_patience=EARLY_STOP,
            finetune_mode=mode,
            lr_backbone=LR_BACKBONE,
            lr_head=LR_HEAD,
            weight_decay=WEIGHT_DECAY,
            reduce_on_plateau={"factor": ROP_FACTOR, "patience": ROP_PATIENCE},
            best_model_filename=f"simpsons_effb0_best_v2_{mode}.pt",
            cm_filename=f"confusion_matrix_{mode}.npy",
            log_filename=f"training_{mode}.log",
        )

        print(f"\n=== Training configuration: finetune_mode={mode} ===")
        metrics = train_from_config(cfg)
        mode_end = datetime.now()
        mode_delta = mode_end - mode_start
        mode_secs = mode_delta.total_seconds()
        print({
            "mode": mode,
            "best_val_acc": metrics["best_val_acc"],
            "num_classes": metrics["num_classes"],
            "start": mode_start.isoformat(),
            "end": mode_end.isoformat(),
            "duration": str(mode_delta),
            "duration_seconds": round(mode_secs, 3),
        })

    total_end = datetime.now()
    total_delta = total_end - total_start
    total_secs = total_delta.total_seconds()
    print(f"\n=== Overall run end:   {total_end.isoformat()} ===")
    print(f"=== Overall duration: {total_delta} ({total_secs:.3f}s) ===")


if __name__ == "__main__":
    main()
