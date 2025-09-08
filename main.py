from pathlib import Path
from datetime import datetime
import os

from train_simpsons_optimized import (
    TrainingConfig,
    train_from_config,
    run_inference_on_kaggle_testset,
    run_inference_on_folder,
)
from tools.prepare_data import prepare_data
from tools.generate_overview import find_items as gen_find_items, render_html as gen_render_html, write_manifest as gen_write_manifest, OUTPUT_ROOT as GEN_OUTPUT_ROOT

# -----------------------------
# Simple training variables (env-overridable)
# -----------------------------

def _get_env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "t", "yes", "y", "on"}


def _get_env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_path(name: str, default: Path) -> Path:
    val = os.getenv(name)
    return Path(val) if val else default


def _get_env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if val is None:
        return default
    items = [item.strip() for item in val.split(",")]
    return [i for i in items if i]


DATA_DIR = _get_env_path("DATA_DIR", Path("data/processed"))
MODELS_DIR = _get_env_path("MODELS_DIR", Path("models"))
INFER_INPUT_DIR = _get_env_path("INFER_INPUT_DIR", Path("data/inference/input"))
INFER_OUTPUT_DIR = _get_env_path("INFER_OUTPUT_DIR", Path("data/inference/output"))

SEED = _get_env_int("SEED", 42)
BATCH_SIZE = _get_env_int("BATCH_SIZE", 32)
NUM_WORKERS = _get_env_int("NUM_WORKERS", 4)
RESIZE = _get_env_int("RESIZE", 256)
IMG_SIZE = _get_env_int("IMG_SIZE", 224)
USE_COLOR_JITTER = _get_env_bool("USE_COLOR_JITTER", True)

EPOCHS = _get_env_int("EPOCHS", 2)
EARLY_STOP = _get_env_int("EARLY_STOP", 5)
FINETUNE_MODES = _get_env_list("FINETUNE_MODES", ["linear_probe", "last2_blocks", "full"])  # three configurations

# Execution controls
RUN_TRAINING = _get_env_bool("RUN_TRAINING", False)
RUN_INFERENCE = _get_env_bool("RUN_INFERENCE", True)

LR_BACKBONE = _get_env_float("LR_BACKBONE", 3e-4)
LR_HEAD = _get_env_float("LR_HEAD", 1e-3)
WEIGHT_DECAY = _get_env_float("WEIGHT_DECAY", 1e-4)
ROP_FACTOR = _get_env_float("ROP_FACTOR", 0.5)
ROP_PATIENCE = _get_env_int("ROP_PATIENCE", 2)

BEST_MODEL = os.getenv("BEST_MODEL", "simpsons_effb0_best_v2.pt")  # base name; per-mode suffixes will be added
CM_FILE = os.getenv("CM_FILE", "confusion_matrix.npy")         # base name; per-mode suffixes will be added


def main() -> None:
    total_start = datetime.now()
    print(f"\n=== Overall run start: {total_start.isoformat()} ===")

    # Ensure data is prepared (download + split) if missing
    try:
        prepare_data()
    except Exception as e:
        print(f"Data preparation warning: {e}")

    for mode in FINETUNE_MODES:
        mode_start = datetime.now()
        if RUN_TRAINING and RUN_INFERENCE:
            print(f"\n=== Training + Inference start ({mode}): {mode_start.isoformat()} ===")
        elif RUN_TRAINING and not RUN_INFERENCE:
            print(f"\n=== Training only start ({mode}): {mode_start.isoformat()} ===")
        elif RUN_INFERENCE and not RUN_TRAINING:
            print(f"\n=== Inference only start ({mode}): {mode_start.isoformat()} ===")
        else:
            print(f"\n=== Both training and inference disabled for mode '{mode}'. Skipping. ===")
            continue
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
        print(f"\n=== Configuration: finetune_mode={mode} ===")
        if RUN_TRAINING:
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
        else:
            print(f"Skipping training for mode '{mode}'.")

        # Inference on Kaggle test set for this mode (best model)
        if RUN_INFERENCE:
            try:
                # Prefer local folder inference as requested
                out_csv = run_inference_on_folder(
                    cfg,
                    cfg.models_dir / cfg.best_model_filename,
                    INFER_INPUT_DIR,
                    INFER_OUTPUT_DIR / mode,
                    batch_size=max(32, BATCH_SIZE),
                )
                if out_csv is not None:
                    print(f"Inference complete ({mode}). CSV: {out_csv}")
                    print(f"Copied images and JSONs under: {INFER_OUTPUT_DIR / mode}")
                    # Generate HTML overview automatically
                    try:
                        items = gen_find_items(mode)
                        html_text = gen_render_html(mode, items, f"Inference Overview — {mode}")
                        out_dir = GEN_OUTPUT_ROOT / mode
                        out_file = out_dir / "index.html"
                        out_file.write_text(html_text)
                        gen_write_manifest(mode, items)
                        print(f"Overview generated: {out_file.resolve()}")
                    except Exception as e:
                        print(f"Overview generation warning for mode '{mode}': {e}")
                else:
                    print("No images found in data/inference/input. Skipping inference.")
            except Exception as e:
                print(f"Inference error for mode '{mode}': {e}")
        else:
            print(f"Skipping inference for mode '{mode}'.")

    total_end = datetime.now()
    total_delta = total_end - total_start
    total_secs = total_delta.total_seconds()
    print(f"\n=== Overall run end:   {total_end.isoformat()} ===")
    print(f"=== Overall duration: {total_delta} ({total_secs:.3f}s) ===")


if __name__ == "__main__":
    main()
