"""Trainable EfficientNet-B0 module with configurable parameters.

This module exposes a TrainingConfig dataclass and a train_from_config entrypoint
so that you can prepare parameters in another file and invoke training.
"""

from dataclasses import dataclass, field
import logging
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.amp import autocast, GradScaler
from sklearn.metrics import classification_report, confusion_matrix
from tqdm.auto import tqdm


IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


@dataclass
class TrainingConfig:
    """Configuration for training run.

    Directories are relative to project root by default.
    """

    data_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")
    seed: int = 42

    # Data loading
    batch_size: int = 32
    num_workers: int = 4
    resize_size: int = 256
    image_size: int = 224
    use_color_jitter: bool = True
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1

    # Training
    epochs: int = 10
    early_stop_patience: int = 5
    finetune_mode: str = "linear_probe"  # one of: linear_probe, last2_blocks, full

    # Optimization
    lr_backbone: float = 3e-4
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    reduce_on_plateau: Dict[str, float] = field(default_factory=lambda: {"factor": 0.5, "patience": 2})

    # Output filenames
    best_model_filename: str = "simpsons_effb0_best_v2.pt"
    cm_filename: str = "confusion_matrix.npy"
    log_filename: Optional[str] = None

    @property
    def train_dir(self) -> Path:
        return self.data_dir / "train"

    @property
    def val_dir(self) -> Path:
        return self.data_dir / "val"


def set_trainable_layers(model: nn.Module, mode: str = "full") -> None:
    """Freeze/unfreeze parameters based on finetuning mode."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if mode == "linear_probe":
        for parameter in model.classifier[1].parameters():
            parameter.requires_grad = True
        return

    if mode == "last2_blocks":
        for parameter in model.features[-2:].parameters():
            parameter.requires_grad = True
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if mode == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    raise ValueError(f"Unknown FINETUNE_MODE: {mode}")


def build_transforms(cfg: TrainingConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    train_list: List[transforms.Transform] = [
        transforms.Resize(cfg.resize_size),
        transforms.RandomCrop(cfg.image_size),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if cfg.use_color_jitter:
        train_list.append(
            transforms.ColorJitter(
                cfg.color_jitter_brightness,
                cfg.color_jitter_contrast,
                cfg.color_jitter_saturation,
                cfg.color_jitter_hue,
            )
        )
    train_list += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    val_list: List[transforms.Transform] = [
        transforms.Resize(cfg.resize_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]

    return transforms.Compose(train_list), transforms.Compose(val_list)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: AdamW,
    train: bool,
    device: torch.device,
    use_channels_last: bool,
    use_amp: bool,
    amp_device: str,
    amp_dtype: torch.dtype,
    scaler: GradScaler,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total = 0
    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    iterator = tqdm(loader, total=len(loader), desc="train" if train else "val", leave=False)
    outer_ctx = torch.enable_grad() if train else torch.inference_mode()

    with outer_ctx:
        for images, targets in iterator:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=amp_device, dtype=amp_dtype, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if train:
                if torch.cuda.is_available():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total += images.size(0)
            all_preds.append(preds.detach().cpu().numpy())
            all_targets.append(targets.detach().cpu().numpy())

            iterator.set_postfix(loss=total_loss / max(total, 1), acc=total_correct / max(total, 1))

    avg_loss = total_loss / max(total, 1)
    accuracy = total_correct / max(total, 1)
    return avg_loss, accuracy, np.concatenate(all_targets), np.concatenate(all_preds)


def train_from_config(cfg: TrainingConfig) -> Dict[str, float]:
    """Run training using the provided configuration.

    Returns minimal metrics summary. Saves best model and confusion matrix.
    """

    # Reproducibility
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Device and AMP
    is_cuda = torch.cuda.is_available()
    is_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if is_mps else ("cuda" if is_cuda else "cpu"))

    use_amp = is_cuda or is_mps
    amp_device = "cuda" if is_cuda else ("mps" if is_mps else "cpu")
    amp_dtype = torch.float16
    scaler = GradScaler(enabled=is_cuda)

    use_channels_last = is_cuda or is_mps
    pin_memory = is_cuda

    # Logging
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    log_path = cfg.models_dir / (cfg.log_filename or f"training_{cfg.finetune_mode}.log")
    logger = logging.getLogger("simpsons_trainer")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Reset handlers to avoid duplicates across multiple runs in one process
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    start_dt = datetime.now()
    logger.info("Starting training run")
    logger.info(
        "Config: mode=%s, epochs=%d, batch_size=%d, lr_head=%.1e, lr_backbone=%.1e, wd=%.1e",
        cfg.finetune_mode,
        cfg.epochs,
        cfg.batch_size,
        cfg.lr_head,
        cfg.lr_backbone,
        cfg.weight_decay,
    )
    logger.info("Data: train_dir=%s val_dir=%s", str(cfg.train_dir), str(cfg.val_dir))
    logger.info("Run start timestamp: %s", start_dt.isoformat())

    # Prepare data
    train_tfms, val_tfms = build_transforms(cfg)
    train_ds = datasets.ImageFolder(cfg.train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(cfg.val_dir, transform=val_tfms)

    with open(cfg.models_dir / "class_to_idx.json", "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)

    prefetch = 2 if cfg.num_workers > 0 else None
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        persistent_workers=(cfg.num_workers > 0),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
    )

    # Model
    num_classes = len(train_ds.classes)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    backbone = models.efficientnet_b0(weights=weights)
    in_feats = backbone.classifier[1].in_features
    backbone.classifier[1] = nn.Linear(in_feats, num_classes)

    model = backbone.to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    set_trainable_layers(model, cfg.finetune_mode)

    # Optimizer and scheduler
    head_params = list(model.classifier[1].parameters())
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("classifier.1")]

    opt_groups = []
    if backbone_params:
        opt_groups.append({"params": backbone_params, "lr": cfg.lr_backbone, "weight_decay": cfg.weight_decay})
    if head_params:
        opt_groups.append({"params": head_params, "lr": cfg.lr_head, "weight_decay": cfg.weight_decay})

    optimizer = AdamW(opt_groups)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.reduce_on_plateau)
    criterion = nn.CrossEntropyLoss()

    # Train loop
    best_val_acc = 0.0
    patience = 0
    y_true_last: np.ndarray = np.array([])
    y_pred_last: np.ndarray = np.array([])
    best_model_path = cfg.models_dir / cfg.best_model_filename

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = datetime.now()

        train_start = datetime.now()
        train_loss, train_acc, _, _ = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            train=True,
            device=device,
            use_channels_last=use_channels_last,
            use_amp=use_amp,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        train_end = datetime.now()

        val_start = datetime.now()
        val_loss, val_acc, y_true_last, y_pred_last = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            train=False,
            device=device,
            use_channels_last=use_channels_last,
            use_amp=use_amp,
            amp_device=amp_device,
            amp_dtype=amp_dtype,
            scaler=scaler,
        )
        val_end = datetime.now()

        # Timings and throughput
        train_secs = max((train_end - train_start).total_seconds(), 1e-6)
        val_secs = max((val_end - val_start).total_seconds(), 1e-6)
        epoch_end = datetime.now()
        epoch_secs = max((epoch_end - epoch_start).total_seconds(), 1e-6)
        train_imgs = len(train_ds)
        val_imgs = len(val_ds)
        train_ips = train_imgs / train_secs
        val_ips = val_imgs / val_secs

        scheduler.step(val_loss)
        logger.info(
            "Epoch %02d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )
        logger.info(
            (
                "Epoch %02d time | "
                "train: %s -> %s (%.2fs, %.1f img/s) | "
                "val: %s -> %s (%.2fs, %.1f img/s) | "
                "total: %.2fs"
            ),
            epoch,
            train_start.isoformat(),
            train_end.isoformat(),
            train_secs,
            train_ips,
            val_start.isoformat(),
            val_end.isoformat(),
            val_secs,
            val_ips,
            epoch_secs,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(
                {"model": model.state_dict(), "classes": train_ds.classes},
                best_model_path,
            )
            logger.info("Saved new best model: %s (val_acc=%.4f)", str(best_model_path), best_val_acc)
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                logger.info("Early stopping.")
                break

    # Final report
    report = classification_report(y_true_last, y_pred_last, target_names=train_ds.classes, digits=3)
    logger.info("\n%s", report)
    cm_path = cfg.models_dir / cfg.cm_filename
    np.save(cm_path, confusion_matrix(y_true_last, y_pred_last))
    logger.info("Saved confusion matrix: %s", str(cm_path))
    logger.info(
        "Summary | mode=%s | best_val_acc=%.4f | best_model=%s | confusion_matrix=%s",
        cfg.finetune_mode,
        best_val_acc,
        str(best_model_path),
        str(cm_path),
    )

    end_dt = datetime.now()
    duration = end_dt - start_dt
    logger.info("Run end timestamp: %s", end_dt.isoformat())
    logger.info("Run duration: %s (%.3fs)", str(duration), duration.total_seconds())

    return {
        "best_val_acc": float(best_val_acc),
        "num_classes": int(num_classes),
    }


__all__ = ["TrainingConfig", "train_from_config"]


if __name__ == "__main__":
    # Default run when executing this module directly
    default_cfg = TrainingConfig()
    train_from_config(default_cfg)
