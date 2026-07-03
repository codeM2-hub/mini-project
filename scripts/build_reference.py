"""
Build Reference — Pre-compute feature centroid from NORMAL (voice) audio ONLY.

IMPORTANT: Only normal/ folder is used as the "known normal" baseline.
Action/sequence audio from dataset/ will then correctly appear as ANOMALY
when the mic hears something different from normal voice.

Usage:
    python scripts/build_reference.py
"""

import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import librosa
from ml.config import SAMPLE_RATE, N_MFCC, N_FFT, HOP_LENGTH

# Use voice_data/ as the normal reference (new dataset structure)
VOICE_DATA_DIR = PROJECT_ROOT / "voice_data"
NORMAL_DIR = VOICE_DATA_DIR if VOICE_DATA_DIR.exists() else PROJECT_ROOT / "normal"


def extract_feature_vector(file_path, sr=SAMPLE_RATE):
    try:
        audio, _ = librosa.load(str(file_path), sr=sr, mono=True, duration=10.0)
        if len(audio) < sr * 0.2:
            return None
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std  = np.std(mfcc, axis=1)
        spec_centroid  = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
        spec_rolloff   = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        zcr  = np.mean(librosa.feature.zero_crossing_rate(audio))
        rms  = np.mean(librosa.feature.rms(y=audio))

        return np.concatenate([mfcc_mean, mfcc_std,
                               [spec_centroid, spec_bandwidth, spec_rolloff, zcr, rms]])
    except Exception as e:
        print(f"  ⚠ Skipping {file_path.name}: {e}")
        return None


def build_reference():
    print("=" * 60)
    print("  Building NORMAL Reference (voice_data/ folder)")
    print("=" * 60)

    features = []
    if not NORMAL_DIR.exists():
        print(f"✗ {NORMAL_DIR} not found!")
        return

    print(f"\nProcessing: {NORMAL_DIR}")
    audio_exts = {".wav", ".m4a", ".mp3", ".flac"}
    all_files = [f for f in sorted(NORMAL_DIR.iterdir()) if f.suffix.lower() in audio_exts]
    print(f"  Found {len(all_files)} voice files...")

    for f in all_files:
        feat = extract_feature_vector(f)
        if feat is not None:
            features.append(feat)

    print(f"  ✓ Extracted {len(features)} feature vectors")

    if not features:
        print("✗ No features extracted!")
        return

    features = np.array(features)
    centroid = np.mean(features, axis=0)

    # Compute cosine similarities of all normal samples to centroid
    from numpy.linalg import norm
    sims = []
    for f in features:
        s = np.dot(f, centroid) / (norm(f) * norm(centroid) + 1e-10)
        sims.append(s)
    sims = np.array(sims)

    mean_sim = float(np.mean(sims))
    std_sim  = float(np.std(sims))
    # Threshold = mean - 2.5 * std (tight boundary)
    threshold = float(mean_sim - 2.5 * std_sim)

    print(f"\n  Similarity stats (normal vs normal):")
    print(f"    Mean:      {mean_sim:.4f}")
    print(f"    Std:       {std_sim:.4f}")
    print(f"    Min:       {float(np.min(sims)):.4f}")
    print(f"    Threshold: {threshold:.4f}  (mean - 2.5*std)")

    output_path = PROJECT_ROOT / "data" / "reference.npz"
    np.savez(
        output_path,
        global_centroid=centroid,
        all_features=features,
        threshold=np.array([threshold]),
        mean_similarity=np.array([mean_sim]),
    )

    print(f"\n  ✓ Reference saved: {output_path}")
    print(f"  ✓ {len(features)} normal voice samples used as baseline")
    print(f"\n  Sounds similar to your voice → NORMAL")
    print(f"  Sounds different (doorbell, etc.) → ANOMALY\n")
    print("=" * 60)


if __name__ == "__main__":
    build_reference()
