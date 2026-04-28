"""
Dataset for Denoising Dirty Documents (Kaggle dataset).
Uses patch-based sampling to augment limited training images.
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class DirtyDocumentDataset(Dataset):
    """
    Patch-based dataset for document denoising.

    Samples random patches from noisy/clean image pairs for training.
    For validation/inference, returns full images.

    Args:
        noisy_dir:        Path to directory with noisy images.
        clean_dir:        Path to directory with clean images (None for test).
        patch_size:       Size of square patches to extract (for training).
        patches_per_img:  Number of random patches per image per epoch.
        mode:             'train', 'val', or 'test'.
        augment:          Apply random augmentations (flip, rotate).
        val_split:        Fraction of data to use for validation (0.1 = 10%).
        seed:             Random seed for reproducibility.
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: Optional[str] = None,
        patch_size: int = 128,
        patches_per_img: int = 30,
        mode: str = "train",
        augment: bool = True,
        val_split: float = 0.1,
        seed: int = 42,
    ):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.patch_size = patch_size
        self.patches_per_img = patches_per_img
        self.mode = mode
        self.augment = augment and (mode == "train")

        # Collect all image files
        all_noisy = sorted(
            [f for f in os.listdir(noisy_dir) if f.lower().endswith((".png", ".jpg", ".tif"))]
        )

        if clean_dir:
            all_clean = sorted(
                [f for f in os.listdir(clean_dir) if f.lower().endswith((".png", ".jpg", ".tif"))]
            )
        else:
            all_clean = all_noisy  # Test mode: no clean reference

        # Train/val split
        rng = random.Random(seed)
        indices = list(range(len(all_noisy)))
        rng.shuffle(indices)
        n_val = max(1, int(len(indices) * val_split))

        if mode == "train":
            indices = indices[n_val:]
        elif mode == "val":
            indices = indices[:n_val]
        # test mode: use all indices

        self.noisy_files = [all_noisy[i] for i in indices]
        self.clean_files = [all_clean[i] for i in indices] if clean_dir else [None] * len(indices)

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.noisy_files) * self.patches_per_img
        return len(self.noisy_files)

    def _load_gray(self, path: str) -> np.ndarray:
        """Load image as float32 numpy array in [0, 1]."""
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32) / 255.0

    def _random_patch(
        self, noisy: np.ndarray, clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch of patch_size × patch_size."""
        h, w = noisy.shape
        ps = self.patch_size

        # Pad if image is smaller than patch
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            noisy = np.pad(noisy, ((0, pad_h), (0, pad_w)), mode="reflect")
            clean = np.pad(clean, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = noisy.shape

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        noisy_patch = noisy[top : top + ps, left : left + ps]
        clean_patch = clean[top : top + ps, left : left + ps]
        return noisy_patch, clean_patch

    def _augment(
        self, noisy: np.ndarray, clean: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply consistent random augmentations to both images."""
        # Horizontal flip
        if random.random() > 0.5:
            noisy = np.fliplr(noisy).copy()
            clean = np.fliplr(clean).copy()
        # Vertical flip
        if random.random() > 0.5:
            noisy = np.flipud(noisy).copy()
            clean = np.flipud(clean).copy()
        # 90° rotation
        k = random.randint(0, 3)
        noisy = np.rot90(noisy, k).copy()
        clean = np.rot90(clean, k).copy()
        return noisy, clean

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        """Convert H×W numpy array to 1×H×W tensor."""
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            img_idx = idx // self.patches_per_img
        else:
            img_idx = idx

        noisy_path = os.path.join(self.noisy_dir, self.noisy_files[img_idx])
        noisy = self._load_gray(noisy_path)

        if self.clean_files[img_idx] is not None:
            clean_path = os.path.join(self.clean_dir, self.clean_files[img_idx])
            clean = self._load_gray(clean_path)
        else:
            clean = noisy.copy()  # Test mode placeholder

        if self.mode == "train":
            noisy, clean = self._random_patch(noisy, clean)
            if self.augment:
                noisy, clean = self._augment(noisy, clean)

        noisy_tensor = self._to_tensor(noisy)
        clean_tensor = self._to_tensor(clean)

        if self.mode == "test":
            return noisy_tensor, self.noisy_files[img_idx]

        return noisy_tensor, clean_tensor


def build_dataloaders(
    data_root: str,
    patch_size: int = 128,
    patches_per_img: int = 30,
    batch_size: int = 16,
    num_workers: int = 2,
    val_split: float = 0.1,
    seed: int = 42,
):
    """Build train and validation DataLoaders."""
    from torch.utils.data import DataLoader

    noisy_dir = os.path.join(data_root, "train")
    clean_dir = os.path.join(data_root, "train_cleaned")

    train_ds = DirtyDocumentDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        patch_size=patch_size,
        patches_per_img=patches_per_img,
        mode="train",
        augment=True,
        val_split=val_split,
        seed=seed,
    )

    val_ds = DirtyDocumentDataset(
        noisy_dir=noisy_dir,
        clean_dir=clean_dir,
        patch_size=patch_size,
        patches_per_img=1,
        mode="val",
        augment=False,
        val_split=val_split,
        seed=seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print(f"Train samples : {len(train_ds):,}  ({len(train_loader)} batches)")
    print(f"Val   samples : {len(val_ds):,}  ({len(val_loader)} batches)")
    return train_loader, val_loader
