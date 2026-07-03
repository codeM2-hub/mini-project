"""
Audio Augmentor — Data augmentation techniques for robust training.

Applies various transformations to audio samples to increase dataset diversity
and improve model generalization. Supports:
  - Gaussian noise injection
  - Time stretching
  - Pitch shifting
  - Volume perturbation
  - Random cropping & padding
  - Frequency masking (SpecAugment-style)
  - Time masking (SpecAugment-style)
"""

import numpy as np
import librosa
from typing import Optional

from ml.config import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    AUG_NOISE_FACTOR,
    AUG_PITCH_SHIFT_RANGE,
    AUG_TIME_STRETCH_RANGE,
    AUG_VOLUME_RANGE,
    AUG_PROBABILITY,
)


class AudioAugmentor:
    """
    Applies random audio augmentations to training samples.

    Usage:
        augmentor = AudioAugmentor()
        augmented = augmentor.augment(audio_array)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        probability: float = AUG_PROBABILITY,
        seed: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.probability = probability
        self.rng = np.random.RandomState(seed)

    def _should_apply(self) -> bool:
        """Randomly decide whether to apply an augmentation."""
        return self.rng.random() < self.probability

    # ------------------------------------------------------------------
    # Individual Augmentations
    # ------------------------------------------------------------------
    def add_noise(
        self, audio: np.ndarray, noise_factor: float = AUG_NOISE_FACTOR
    ) -> np.ndarray:
        """Add Gaussian white noise to the audio signal."""
        noise = self.rng.randn(len(audio)) * noise_factor
        return audio + noise

    def time_stretch(
        self,
        audio: np.ndarray,
        rate_range: tuple = AUG_TIME_STRETCH_RANGE,
    ) -> np.ndarray:
        """Speed up or slow down the audio without changing pitch."""
        rate = self.rng.uniform(*rate_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        # Pad or trim to original length
        if len(stretched) > len(audio):
            stretched = stretched[: len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        return stretched

    def pitch_shift(
        self,
        audio: np.ndarray,
        shift_range: tuple = AUG_PITCH_SHIFT_RANGE,
    ) -> np.ndarray:
        """Shift pitch up or down by a random number of semitones."""
        n_steps = self.rng.uniform(*shift_range)
        shifted = librosa.effects.pitch_shift(
            audio, sr=self.sample_rate, n_steps=n_steps
        )
        return shifted

    def change_volume(
        self,
        audio: np.ndarray,
        volume_range: tuple = AUG_VOLUME_RANGE,
    ) -> np.ndarray:
        """Randomly scale the amplitude of the audio."""
        factor = self.rng.uniform(*volume_range)
        return np.clip(audio * factor, -1.0, 1.0)

    def random_crop_pad(
        self, audio: np.ndarray, target_length: int = WINDOW_SAMPLES
    ) -> np.ndarray:
        """Randomly crop or pad the audio to a fixed length."""
        if len(audio) > target_length:
            start = self.rng.randint(0, len(audio) - target_length)
            return audio[start : start + target_length]
        elif len(audio) < target_length:
            pad_left = self.rng.randint(0, target_length - len(audio))
            pad_right = target_length - len(audio) - pad_left
            return np.pad(audio, (pad_left, pad_right))
        return audio

    def frequency_mask(
        self,
        spectrogram: np.ndarray,
        max_mask_size: int = 20,
    ) -> np.ndarray:
        """Apply frequency masking (SpecAugment) to a spectrogram."""
        spec = spectrogram.copy()
        n_freq = spec.shape[0]
        mask_size = self.rng.randint(1, min(max_mask_size, n_freq))
        start = self.rng.randint(0, n_freq - mask_size)
        spec[start : start + mask_size, :] = 0
        return spec

    def time_mask(
        self,
        spectrogram: np.ndarray,
        max_mask_size: int = 10,
    ) -> np.ndarray:
        """Apply time masking (SpecAugment) to a spectrogram."""
        spec = spectrogram.copy()
        n_time = spec.shape[1]
        mask_size = self.rng.randint(1, min(max_mask_size, n_time))
        start = self.rng.randint(0, n_time - mask_size)
        spec[:, start : start + mask_size] = 0
        return spec

    # ------------------------------------------------------------------
    # Combined Augmentation Pipeline
    # ------------------------------------------------------------------
    def augment(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply a random subset of augmentations to the audio.

        Each augmentation is applied independently with `self.probability`.
        """
        augmented = audio.copy()

        if self._should_apply():
            augmented = self.add_noise(augmented)

        if self._should_apply():
            augmented = self.time_stretch(augmented)

        if self._should_apply():
            augmented = self.pitch_shift(augmented)

        if self._should_apply():
            augmented = self.change_volume(augmented)

        return augmented

    def augment_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """
        Apply SpecAugment-style augmentations to a spectrogram.

        Applied after feature extraction for additional regularization.
        """
        aug_spec = spectrogram.copy()

        if self._should_apply():
            aug_spec = self.frequency_mask(aug_spec)

        if self._should_apply():
            aug_spec = self.time_mask(aug_spec)

        return aug_spec
