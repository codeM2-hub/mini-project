"""
Audio Processor — Handles loading, normalization, resampling, and windowing.

This module is the first stage of the pipeline. It takes raw audio files or
real-time microphone streams and produces clean, normalized, fixed-length
audio segments ready for feature extraction.

Key responsibilities:
  - Load audio from WAV/M4A files
  - Resample to target sample rate
  - Normalize amplitude
  - Remove silence / apply voice activity detection
  - Segment into overlapping windows for sliding-window inference
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, List, Optional, Generator

from ml.config import (
    SAMPLE_RATE,
    WINDOW_DURATION,
    HOP_DURATION,
    WINDOW_SAMPLES,
    HOP_SAMPLES,
    MAX_AUDIO_DURATION,
)


class AudioProcessor:
    """
    Processes raw audio into clean, windowed segments.

    Usage:
        processor = AudioProcessor()
        audio = processor.load_audio("path/to/file.wav")
        windows = processor.segment_into_windows(audio)
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_duration: float = WINDOW_DURATION,
        hop_duration: float = HOP_DURATION,
    ):
        self.sample_rate = sample_rate
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.window_samples = int(sample_rate * window_duration)
        self.hop_samples = int(sample_rate * hop_duration)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load_audio(
        self,
        file_path: str,
        mono: bool = True,
        max_duration: Optional[float] = MAX_AUDIO_DURATION,
    ) -> np.ndarray:
        """
        Load an audio file and return a 1-D numpy array at the target sample rate.

        Args:
            file_path: Path to the audio file (WAV, M4A, etc.)
            mono: If True, convert stereo to mono.
            max_duration: Maximum duration in seconds to load. None = full file.

        Returns:
            1-D numpy array of audio samples, normalized to [-1, 1].
        """
        file_path = str(file_path)
        duration = max_duration

        try:
            audio, sr = librosa.load(
                file_path,
                sr=self.sample_rate,
                mono=mono,
                duration=duration,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {file_path}: {e}")

        # Normalize
        audio = self.normalize(audio)
        return audio

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------
    @staticmethod
    def normalize(audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range using peak normalization."""
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        return audio

    @staticmethod
    def rms_normalize(audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        """Normalize audio to a target RMS level."""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms > 0:
            audio = audio * (target_rms / current_rms)
        return np.clip(audio, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Silence Removal
    # ------------------------------------------------------------------
    def remove_silence(
        self,
        audio: np.ndarray,
        top_db: int = 30,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> np.ndarray:
        """
        Remove leading/trailing silence from audio using librosa's trim.

        Args:
            audio: Input audio array.
            top_db: Threshold in dB below reference to consider as silence.

        Returns:
            Trimmed audio array.
        """
        trimmed, _ = librosa.effects.trim(
            audio, top_db=top_db,
            frame_length=frame_length,
            hop_length=hop_length,
        )
        return trimmed

    # ------------------------------------------------------------------
    # Windowing
    # ------------------------------------------------------------------
    def segment_into_windows(
        self,
        audio: np.ndarray,
        pad_last: bool = True,
    ) -> List[np.ndarray]:
        """
        Split audio into overlapping windows for sliding-window processing.

        Args:
            audio: 1-D audio array.
            pad_last: If True, zero-pad the last window to reach full length.

        Returns:
            List of 1-D numpy arrays, each of length window_samples.
        """
        windows = []
        start = 0

        while start < len(audio):
            end = start + self.window_samples
            window = audio[start:end]

            # Pad if the last window is shorter
            if len(window) < self.window_samples:
                if pad_last:
                    window = np.pad(
                        window,
                        (0, self.window_samples - len(window)),
                        mode="constant",
                    )
                else:
                    break

            windows.append(window)
            start += self.hop_samples

        return windows

    def stream_windows(
        self, audio: np.ndarray
    ) -> Generator[Tuple[np.ndarray, float], None, None]:
        """
        Generator that yields (window, timestamp) pairs for streaming inference.

        Args:
            audio: Full audio array.

        Yields:
            (window_array, start_time_seconds) tuples.
        """
        start = 0
        while start + self.window_samples <= len(audio):
            window = audio[start : start + self.window_samples]
            timestamp = start / self.sample_rate
            yield window, timestamp
            start += self.hop_samples

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def pad_or_trim(self, audio: np.ndarray, target_length: int = None) -> np.ndarray:
        """Pad with zeros or trim audio to a fixed target length."""
        if target_length is None:
            target_length = self.window_samples

        if len(audio) >= target_length:
            return audio[:target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode="constant")

    @staticmethod
    def to_mono(audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if audio.ndim > 1:
            return np.mean(audio, axis=0)
        return audio

    def get_duration(self, audio: np.ndarray) -> float:
        """Get the duration of an audio array in seconds."""
        return len(audio) / self.sample_rate
