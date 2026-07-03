"""
Feature Extractor — Converts raw audio windows into ML-ready feature tensors.

Extracts multiple audio features and combines them into a rich representation:
  - Mel Spectrograms (primary feature for CNN input)
  - MFCCs (compact spectral envelope)
  - Chroma Features (pitch class profiles)
  - Spectral Contrast (valley-to-peak energy ratio per sub-band)
  - Zero Crossing Rate (noisiness indicator)

The primary output is a mel spectrogram tensor shaped for the CRNN model.
Additional features can be concatenated or used independently.
"""

import numpy as np
import librosa
import torch
from typing import Dict, Optional

from ml.config import (
    SAMPLE_RATE,
    N_FFT,
    HOP_LENGTH,
    N_MELS,
    N_MFCC,
    FMIN,
    FMAX,
    WINDOW_SAMPLES,
)


class FeatureExtractor:
    """
    Extracts audio features from raw waveform arrays.

    Usage:
        extractor = FeatureExtractor()
        mel_spec = extractor.extract_mel_spectrogram(audio_window)
        all_features = extractor.extract_all_features(audio_window)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
        n_mfcc: int = N_MFCC,
        fmin: float = FMIN,
        fmax: float = FMAX,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = fmax

    # ------------------------------------------------------------------
    # Primary Feature: Mel Spectrogram
    # ------------------------------------------------------------------
    def extract_mel_spectrogram(
        self,
        audio: np.ndarray,
        to_db: bool = True,
    ) -> np.ndarray:
        """
        Compute the mel spectrogram of an audio window.

        Args:
            audio: 1-D audio array.
            to_db: If True, convert power spectrogram to decibel scale.

        Returns:
            2-D array of shape (n_mels, time_frames).
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        if to_db:
            mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec

    # ------------------------------------------------------------------
    # MFCC Features
    # ------------------------------------------------------------------
    def extract_mfcc(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute MFCCs (Mel-Frequency Cepstral Coefficients).

        Returns:
            2-D array of shape (n_mfcc, time_frames).
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        # Add delta and delta-delta for richer temporal information
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

        return np.concatenate([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    # ------------------------------------------------------------------
    # Chroma Features
    # ------------------------------------------------------------------
    def extract_chroma(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute chroma features (pitch class energy distribution).

        Returns:
            2-D array of shape (12, time_frames).
        """
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return chroma

    # ------------------------------------------------------------------
    # Spectral Contrast
    # ------------------------------------------------------------------
    def extract_spectral_contrast(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral contrast across frequency sub-bands.

        Returns:
            2-D array of shape (7, time_frames).
        """
        contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return contrast

    # ------------------------------------------------------------------
    # Zero Crossing Rate
    # ------------------------------------------------------------------
    def extract_zcr(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute zero crossing rate per frame.

        Returns:
            2-D array of shape (1, time_frames).
        """
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            frame_length=self.n_fft,
            hop_length=self.hop_length,
        )
        return zcr

    # ------------------------------------------------------------------
    # Combined Feature Extraction
    # ------------------------------------------------------------------
    def extract_all_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all supported features from an audio window.

        Returns:
            Dictionary mapping feature names to numpy arrays.
        """
        return {
            "mel_spectrogram": self.extract_mel_spectrogram(audio),
            "mfcc": self.extract_mfcc(audio),
            "chroma": self.extract_chroma(audio),
            "spectral_contrast": self.extract_spectral_contrast(audio),
            "zcr": self.extract_zcr(audio),
        }

    # ------------------------------------------------------------------
    # Tensor Conversion for Model Input
    # ------------------------------------------------------------------
    def audio_to_tensor(
        self,
        audio: np.ndarray,
        feature_type: str = "mel_spectrogram",
    ) -> torch.Tensor:
        """
        Convert audio to a PyTorch tensor ready for the CRNN model.

        Args:
            audio: 1-D audio array.
            feature_type: Which feature to extract ("mel_spectrogram" or "mfcc").

        Returns:
            Tensor of shape (1, n_features, time_frames) — batch dim NOT included.
        """
        if feature_type == "mel_spectrogram":
            features = self.extract_mel_spectrogram(audio)
        elif feature_type == "mfcc":
            features = self.extract_mfcc(audio)
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        # Add channel dimension: (1, freq, time)
        tensor = torch.FloatTensor(features).unsqueeze(0)
        return tensor

    def batch_extract(
        self,
        audio_windows: list,
        feature_type: str = "mel_spectrogram",
    ) -> torch.Tensor:
        """
        Extract features from a batch of audio windows.

        Args:
            audio_windows: List of 1-D audio arrays.
            feature_type: Feature type to extract.

        Returns:
            Tensor of shape (batch, 1, n_features, time_frames).
        """
        tensors = [self.audio_to_tensor(w, feature_type) for w in audio_windows]
        return torch.stack(tensors, dim=0)
