"""
Cached Dataset — Loads pre-extracted .npy features instead of raw audio.
10-50x faster than loading audio files every epoch.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict

from ml.config import NUM_CLASSES, PROJECT_ROOT


class CachedAudioDataset(Dataset):
    """
    Loads pre-computed mel spectrogram features (.npy) directly.
    Must run extract_features.py first to populate the cache.
    """

    def __init__(self, manifest_path: str, split: str = "train", augment: bool = False):
        self.split   = split
        self.augment = augment and split == "train"

        # Load label mapping
        labels_path = PROJECT_ROOT / "data" / "labels.json"
        with open(labels_path) as f:
            ld = json.load(f)
        self.label_to_idx = ld["label_to_idx"]
        self.idx_to_label = {int(k): v for k, v in ld["idx_to_label"].items()}
        self.num_classes  = len(self.label_to_idx)

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.samples: List[Dict] = manifest.get(split, [])
        print(f"[CachedDataset] {split}: {len(self.samples)} samples, {self.num_classes} classes")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Load cached feature
        feat = np.load(sample["cache_path"])          # (1, n_mels, time)
        feat = torch.FloatTensor(feat)

        # Light augmentation in-memory (fast, no disk I/O)
        if self.augment:
            feat = self._augment(feat)

        # Label
        label_vec = torch.zeros(self.num_classes, dtype=torch.float32)
        for li in sample["labels"]:
            if li < self.num_classes:
                label_vec[li] = 1.0

        return feat, label_vec

    def _augment(self, feat: torch.Tensor) -> torch.Tensor:
        """Fast spectrogram-domain augmentation."""
        # SpecAugment: time masking
        if torch.rand(1) < 0.5:
            t = feat.shape[-1]
            mask_len = torch.randint(1, max(2, t // 8), (1,)).item()
            start    = torch.randint(0, t - mask_len, (1,)).item()
            feat = feat.clone()
            feat[..., start:start + mask_len] = 0.0

        # SpecAugment: frequency masking
        if torch.rand(1) < 0.5:
            f = feat.shape[-2]
            mask_len = torch.randint(1, max(2, f // 8), (1,)).item()
            start    = torch.randint(0, f - mask_len, (1,)).item()
            feat = feat.clone()
            feat[:, start:start + mask_len, :] = 0.0

        # Random gain
        if torch.rand(1) < 0.4:
            gain = torch.empty(1).uniform_(0.8, 1.2)
            feat = feat * gain

        # Small additive noise in spectrogram domain
        if torch.rand(1) < 0.3:
            feat = feat + torch.randn_like(feat) * 0.01

        return feat

    def get_class_weights(self) -> torch.Tensor:
        counts = np.zeros(self.num_classes)
        for s in self.samples:
            for li in s["labels"]:
                if li < self.num_classes:
                    counts[li] += 1
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (self.num_classes * counts)
        return torch.FloatTensor(weights)
