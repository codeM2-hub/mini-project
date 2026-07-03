"""
Audio Action Dataset — PyTorch Dataset for loading and preprocessing audio samples.

Reads audio files from the organized dataset directory, extracts features,
applies augmentations during training, and returns (feature_tensor, label_tensor) pairs.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, List, Dict

from ml.config import (
    PROCESSED_DIR,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    NUM_CLASSES,
    AUGMENTATION_ENABLED,
    PROJECT_ROOT,
)
from ml.preprocessing.audio_processor import AudioProcessor
from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.preprocessing.augmentation import AudioAugmentor


class AudioActionDataset(Dataset):
    """
    PyTorch Dataset for audio action recognition.

    Loads audio samples, extracts mel spectrogram features, and applies
    augmentations during training.

    Args:
        manifest_path: Path to the JSON manifest file listing samples.
        split: One of "train", "val", "test".
        augment: Whether to apply data augmentation.
        feature_type: Feature to extract ("mel_spectrogram" or "mfcc").
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        augment: bool = True,
        feature_type: str = "mel_spectrogram",
    ):
        self.split = split
        self.augment = augment and split == "train" and AUGMENTATION_ENABLED
        self.feature_type = feature_type

        # Initialize processors
        self.audio_processor = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.augmentor = AudioAugmentor() if self.augment else None

        # Load manifest
        self.samples = self._load_manifest(manifest_path, split)

        # Load label mapping
        labels_path = PROJECT_ROOT / "data" / "labels.json"
        if labels_path.exists():
            with open(labels_path, "r") as f:
                label_data = json.load(f)
            self.label_to_idx = label_data.get("label_to_idx", {})
            self.idx_to_label = {int(v): k for k, v in self.label_to_idx.items()}
            self.num_classes = len(self.label_to_idx)
        else:
            self.num_classes = NUM_CLASSES
            self.label_to_idx = {}
            self.idx_to_label = {}

    def _load_manifest(
        self, manifest_path: str, split: str
    ) -> List[Dict]:
        """Load the dataset manifest JSON file."""
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run 'python scripts/prepare_dataset.py' first."
            )

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        samples = manifest.get(split, [])
        print(f"[Dataset] Loaded {len(samples)} samples for '{split}' split.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            (features, label): Feature tensor and multi-hot label tensor.
        """
        sample = self.samples[idx]
        audio_path = sample["path"]
        label_indices = sample["labels"]  # List of integer class indices

        # Load audio
        try:
            audio = self.audio_processor.load_audio(audio_path)
        except Exception as e:
            print(f"[Dataset] Warning: Failed to load {audio_path}: {e}")
            # Return a zero tensor as fallback
            audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

        # Pad or trim to fixed length
        audio = self.audio_processor.pad_or_trim(audio, WINDOW_SAMPLES)

        # Apply augmentation
        if self.augment and self.augmentor is not None:
            audio = self.augmentor.augment(audio)

        # Extract features
        features = self.feature_extractor.audio_to_tensor(audio, self.feature_type)

        # Apply spectrogram augmentation
        if self.augment and self.augmentor is not None:
            feat_np = features.squeeze(0).numpy()
            feat_np = self.augmentor.augment_spectrogram(feat_np)
            features = torch.FloatTensor(feat_np).unsqueeze(0)

        # Create multi-hot label vector
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        for li in label_indices:
            if li < self.num_classes:
                label[li] = 1.0

        return features, label

    def get_class_weights(self) -> torch.Tensor:
        """
        Compute class weights inversely proportional to class frequency.
        Useful for handling imbalanced datasets.
        """
        class_counts = np.zeros(self.num_classes)
        for sample in self.samples:
            for li in sample["labels"]:
                if li < self.num_classes:
                    class_counts[li] += 1

        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        total = np.sum(class_counts)
        weights = total / (self.num_classes * class_counts)

        return torch.FloatTensor(weights)
