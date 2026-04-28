"""
Training script for Document Denoising Autoencoder.
Supports checkpoint-based resume for interrupted Colab sessions.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .model import DenoisingUNet
from .utils import CombinedLoss, EpochTimer, MetricTracker, psnr, ssim


# ─────────────────────────────────────────────
#  CHECKPOINT MANAGEMENT
# ─────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    history: Dict,
    best_psnr: float,
    checkpoint_dir: str,
    filename: str = "latest.pth",
):
    """Save a full training checkpoint with timestamp."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "history": history,
        "best_psnr": best_psnr,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(checkpoint_dir, filename)
    torch.save(state, path)
    # Also save epoch-specific copy
    epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch:04d}.pth")
    torch.save(state, epoch_path)
    return path


def load_checkpoint(
    checkpoint_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: str = "cpu",
    filename: str = "latest.pth",
) -> Dict:
    """Load checkpoint and restore training state. Returns metadata dict."""
    path = os.path.join(checkpoint_dir, filename)
    if not os.path.exists(path):
        print(f"No checkpoint found at {path}. Starting from scratch.")
        return {"epoch": 0, "history": {}, "best_psnr": 0.0}

    print(f"Loading checkpoint from {path}")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])

    info = {
        "epoch": ckpt["epoch"],
        "history": ckpt.get("history", {}),
        "best_psnr": ckpt.get("best_psnr", 0.0),
        "saved_at": ckpt.get("saved_at", "unknown"),
    }
    print(f"  Resumed from epoch {info['epoch']}  |  Best PSNR: {info['best_psnr']:.2f} dB")
    print(f"  Checkpoint saved at: {info['saved_at']}")
    return info


def save_best_model(model: nn.Module, output_dir: str, model_name: str = "best_model.pth"):
    """Save only the model weights (no optimizer state) for deployment."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, model_name)
    torch.save(model.state_dict(), path)
    print(f"Best model saved → {path}")
    return path


# ─────────────────────────────────────────────
#  TRAIN / VALIDATE EPOCHS
# ─────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: str,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Run one training epoch. Returns averaged metrics."""
    model.train()
    tracker = MetricTracker()

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy)
        loss = loss_fn(pred, clean)
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        with torch.no_grad():
            p = psnr(pred, clean)
            s = ssim(pred, clean)
        tracker.update(loss=loss.item(), psnr=p, ssim=s)

    return tracker.averages()


@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader,
    loss_fn: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Run one validation epoch. Returns averaged metrics."""
    model.eval()
    tracker = MetricTracker()

    for noisy, clean in loader:
        noisy = noisy.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        pred = model(noisy)

        # For validation, we may get full images — process in patches if needed
        if noisy.shape[-1] > 512 or noisy.shape[-2] > 512:
            pred = torch.clamp(pred, 0, 1)

        loss = loss_fn(pred, clean)
        p = psnr(pred, clean)
        s = ssim(pred, clean)
        tracker.update(loss=loss.item(), psnr=p, ssim=s)

    return tracker.averages()


# ─────────────────────────────────────────────
#  MAIN TRAINING LOOP
# ─────────────────────────────────────────────

def train(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    checkpoint_dir: str = "checkpoints",
    output_dir: str = "outputs",
    resume: bool = True,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Full training loop with automatic checkpoint-based resume.

    Returns:
        history dict with keys: train_loss, val_loss, train_psnr, val_psnr,
                                 train_ssim, val_ssim, epoch_times, lr_history.
    """
    # Setup
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    loss_fn = CombinedLoss(mse_weight=0.5, ssim_weight=0.5)
    timer = EpochTimer(num_epochs)

    # Initialize history
    history = {
        "train_loss": [], "val_loss": [],
        "train_psnr": [], "val_psnr": [],
        "train_ssim": [], "val_ssim": [],
        "epoch_times": [], "lr_history": [],
    }
    best_psnr = 0.0
    start_epoch = 0

    # Resume if checkpoint exists
    if resume:
        info = load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device)
        start_epoch = info["epoch"]
        best_psnr = info["best_psnr"]
        if info["history"]:
            history = info["history"]

    print(f"\n{'='*60}")
    print(f"  Training: {start_epoch} → {num_epochs} epochs")
    print(f"  Device  : {device.upper()}")
    print(f"  Params  : {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch + 1, num_epochs + 1):
        timer.start()

        train_metrics = train_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip)
        val_metrics = validate_epoch(model, val_loader, loss_fn, device)
        scheduler.step()

        epoch_time = timer.stop()
        current_lr = optimizer.param_groups[0]["lr"]

        # Store history
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_psnr"].append(train_metrics["psnr"])
        history["val_psnr"].append(val_metrics["psnr"])
        history["train_ssim"].append(train_metrics["ssim"])
        history["val_ssim"].append(val_metrics["ssim"])
        history["epoch_times"].append(epoch_time)
        history["lr_history"].append(current_lr)

        is_best = val_metrics["psnr"] > best_psnr
        if is_best:
            best_psnr = val_metrics["psnr"]
            save_best_model(model, output_dir)

        # Save checkpoint every epoch
        save_checkpoint(
            epoch, model, optimizer, scheduler, history, best_psnr, checkpoint_dir
        )

        if verbose:
            eta = timer.eta(epoch)
            elapsed = timer.total_elapsed()
            star = "★" if is_best else " "
            print(
                f"Ep {epoch:3d}/{num_epochs} "
                f"| loss {train_metrics['loss']:.4f}/{val_metrics['loss']:.4f} "
                f"| PSNR {train_metrics['psnr']:.2f}/{val_metrics['psnr']:.2f} dB "
                f"| SSIM {val_metrics['ssim']:.4f} "
                f"| {epoch_time:.1f}s "
                f"| ETA {eta} "
                f"| Total {elapsed} {star}"
            )

    print(f"\n✓ Training complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"  Total time: {timer.total_elapsed()}")
    return history
