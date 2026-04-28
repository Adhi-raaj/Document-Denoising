"""
Utility functions: metrics (PSNR, SSIM), combined loss, visualization.
"""

import math
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (dB). Higher is better. >30 dB is good."""
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 20 * math.log10(max_val) - 10 * math.log10(mse)


def _gaussian_kernel(kernel_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Create a 2D Gaussian kernel."""
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel = g.unsqueeze(0) * g.unsqueeze(1)
    return kernel / kernel.sum()


_SSIM_KERNEL: torch.Tensor = None


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 11,
    sigma: float = 1.5,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> float:
    """
    Structural Similarity Index (SSIM). Range [0, 1]. Higher is better.
    1.0 = perfect reconstruction.
    """
    global _SSIM_KERNEL
    if _SSIM_KERNEL is None or _SSIM_KERNEL.device != pred.device:
        kernel = _gaussian_kernel(kernel_size, sigma)
        _SSIM_KERNEL = kernel.view(1, 1, kernel_size, kernel_size).to(pred.device)

    mu_x = F.conv2d(pred, _SSIM_KERNEL, padding=kernel_size // 2)
    mu_y = F.conv2d(target, _SSIM_KERNEL, padding=kernel_size // 2)

    mu_x_sq = mu_x**2
    mu_y_sq = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred**2, _SSIM_KERNEL, padding=kernel_size // 2) - mu_x_sq
    sigma_y = F.conv2d(target**2, _SSIM_KERNEL, padding=kernel_size // 2) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, _SSIM_KERNEL, padding=kernel_size // 2) - mu_xy

    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

    return numerator.div(denominator).mean().item()


# ─────────────────────────────────────────────
#  LOSS FUNCTIONS
# ─────────────────────────────────────────────

class SSIMLoss(nn.Module):
    """SSIM-based loss (1 - SSIM). Lower is better during training."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        s = ssim(pred, target)
        return 1.0 - torch.tensor(s, requires_grad=False)


class CombinedLoss(nn.Module):
    """
    MSE + (1 - SSIM) combined loss.

    MSE ensures pixel-level accuracy.
    SSIM preserves structural details critical for document legibility.
    """

    def __init__(self, mse_weight: float = 0.5, ssim_weight: float = 0.5):
        super().__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = F.mse_loss(pred, target)

        # SSIM loss (differentiable via autograd through SSIM kernel conv)
        ssim_val = self._ssim_loss(pred, target)

        return self.mse_weight * mse_loss + self.ssim_weight * ssim_val

    def _ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        C1, C2 = 0.01**2, 0.03**2
        kernel = _gaussian_kernel(11, 1.5).view(1, 1, 11, 11).to(pred.device)

        mu_x = F.conv2d(pred, kernel, padding=5)
        mu_y = F.conv2d(target, kernel, padding=5)
        mu_x_sq, mu_y_sq, mu_xy = mu_x**2, mu_y**2, mu_x * mu_y

        sigma_x = F.conv2d(pred**2, kernel, padding=5) - mu_x_sq
        sigma_y = F.conv2d(target**2, kernel, padding=5) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, kernel, padding=5) - mu_xy

        num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        den = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)

        return 1.0 - (num / den).mean()


# ─────────────────────────────────────────────
#  TIMING UTILITIES
# ─────────────────────────────────────────────

class EpochTimer:
    """Track per-epoch times and estimate remaining training duration."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.times: List[float] = []
        self._start: float = 0.0

    def start(self):
        self._start = time.time()

    def stop(self) -> float:
        elapsed = time.time() - self._start
        self.times.append(elapsed)
        return elapsed

    def eta(self, current_epoch: int) -> str:
        if not self.times:
            return "N/A"
        avg = sum(self.times) / len(self.times)
        remaining = (self.total_epochs - current_epoch) * avg
        m, s = divmod(int(remaining), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"

    def total_elapsed(self) -> str:
        t = sum(self.times)
        m, s = divmod(int(t), 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h}h {m}m {s}s"
        return f"{m}m {s}s"


# ─────────────────────────────────────────────
#  INFERENCE UTILITIES
# ─────────────────────────────────────────────

def denoise_image(
    model: nn.Module,
    image: np.ndarray,
    patch_size: int = 256,
    overlap: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Denoise a full-resolution image using overlapping patch inference.

    This avoids boundary artifacts and handles any image size.
    Uses weighted blending in overlap regions.

    Args:
        model:      Trained DenoisingUNet model.
        image:      Grayscale image as np.ndarray in [0, 1], shape H×W.
        patch_size: Size of patches for inference.
        overlap:    Overlap between adjacent patches.
        device:     'cpu' or 'cuda'.

    Returns:
        Denoised image as np.ndarray in [0, 1], shape H×W.
    """
    model.eval()
    h, w = image.shape
    step = patch_size - overlap

    # Pad image so it fits evenly
    pad_h = math.ceil((h - patch_size) / step) * step + patch_size - h if h > patch_size else patch_size - h
    pad_w = math.ceil((w - patch_size) / step) * step + patch_size - w if w > patch_size else patch_size - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    output = np.zeros((ph, pw), dtype=np.float64)
    weight = np.zeros((ph, pw), dtype=np.float64)

    # Gaussian weight map for smooth blending
    gy = np.hanning(patch_size)
    gx = np.hanning(patch_size)
    blend = np.outer(gy, gx).astype(np.float64)
    blend = np.clip(blend, 1e-6, None)

    with torch.no_grad():
        for top in range(0, ph - patch_size + 1, step):
            for left in range(0, pw - patch_size + 1, step):
                patch = padded[top : top + patch_size, left : left + patch_size]
                t = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
                pred = model(t).squeeze().cpu().numpy()
                output[top : top + patch_size, left : left + patch_size] += pred * blend
                weight[top : top + patch_size, left : left + patch_size] += blend

    result = output / (weight + 1e-8)
    return np.clip(result[:h, :w], 0.0, 1.0).astype(np.float32)


# ─────────────────────────────────────────────
#  HISTORY TRACKING
# ─────────────────────────────────────────────

class MetricTracker:
    """Accumulate and average metrics over an epoch."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums = {}
        self._counts = {}

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self._sums[k] = self._sums.get(k, 0.0) + float(v)
            self._counts[k] = self._counts.get(k, 0) + 1

    def averages(self) -> dict:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}


if __name__ == "__main__":
    # Quick sanity check
    pred = torch.rand(2, 1, 128, 128)
    tgt = torch.rand(2, 1, 128, 128)
    print(f"PSNR: {psnr(pred, tgt):.2f} dB")
    print(f"SSIM: {ssim(pred, tgt):.4f}")

    loss_fn = CombinedLoss()
    loss = loss_fn(pred, tgt)
    print(f"Combined loss: {loss.item():.4f}")
