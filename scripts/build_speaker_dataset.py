"""
build_speaker_dataset.py
------------------------
Reads raw voice data and long anomaly audio clips.
Slices long clips into overlapping windows, skipping silence.
Applies extreme augmentations to ensure robustness against TV/YouTube noise.
Outputs to: data/speaker_dataset/valid and data/speaker_dataset/anomaly
"""

import os
import glob
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import multiprocessing

# Configuration
SR = 22050
CLIP_DURATION = 3.0  # seconds
CLIP_SAMPLES = int(SR * CLIP_DURATION)
OVERLAP_SAMPLES = int(SR * 1.5)  # 1.5 seconds overlap
SILENCE_THRESH_DB = 30  # dB threshold for silence detection

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_ANOMALY_DIR = PROJECT_ROOT / "anomaly_raw"
RAW_VOICE_DIR = PROJECT_ROOT / "voice_data"
OUT_DIR = PROJECT_ROOT / "data" / "speaker_dataset"
OUT_VALID = OUT_DIR / "valid"
OUT_ANOMALY = OUT_DIR / "anomaly"

OUT_VALID.mkdir(parents=True, exist_ok=True)
OUT_ANOMALY.mkdir(parents=True, exist_ok=True)

# Augmentations pipeline
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.6),
    TimeStretch(min_rate=0.8, max_rate=1.2, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.6)
])

def process_file(args):
    filepath, label, is_long = args
    y, sr = librosa.load(filepath, sr=SR)
    
    filename = Path(filepath).stem
    out_path = OUT_VALID if label == "valid" else OUT_ANOMALY

    # Skip silent parts and get non-silent intervals
    intervals = librosa.effects.split(y, top_db=SILENCE_THRESH_DB)
    
    y_clean = []
    for start, end in intervals:
        y_clean.extend(y[start:end])
    y_clean = np.array(y_clean)

    if len(y_clean) == 0:
        return

    step = CLIP_SAMPLES - OVERLAP_SAMPLES if is_long else CLIP_SAMPLES
    
    chunk_idx = 0
    for start_idx in range(0, len(y_clean) - CLIP_SAMPLES + 1, step):
        chunk = y_clean[start_idx:start_idx + CLIP_SAMPLES]
        
        # Save original clean chunk
        sf.write(out_path / f"{filename}_{chunk_idx}.wav", chunk, SR)
        
        # Create 2 augmented versions for each chunk
        for aug_idx in range(2):
            try:
                chunk_aug = augmenter(samples=chunk, sample_rate=SR)
                # Pad/Trim just in case TimeStretch changed length
                if len(chunk_aug) < CLIP_SAMPLES:
                    chunk_aug = np.pad(chunk_aug, (0, CLIP_SAMPLES - len(chunk_aug)))
                else:
                    chunk_aug = chunk_aug[:CLIP_SAMPLES]
                    
                sf.write(out_path / f"{filename}_{chunk_idx}_aug{aug_idx}.wav", chunk_aug, SR)
            except Exception as e:
                pass
        
        chunk_idx += 1


def main():
    print("="*60)
    print(" Speaker Dataset Generation Pipeline ")
    print("="*60)
    
    tasks = []
    
    # 1. Gather valid voice files (skip already augmented ones from previous runs to keep it clean)
    voice_files = list(RAW_VOICE_DIR.glob("*.wav"))
    orig_voice_files = [f for f in voice_files if "_aug" not in f.name]
    if not orig_voice_files:
        # fallback if all are augmented from previous scripts
        orig_voice_files = voice_files 
        
    for f in orig_voice_files:
        tasks.append((f, "valid", False))
        
    # 2. Gather long anomaly files
    anomaly_files = list(RAW_ANOMALY_DIR.glob("*.wav"))
    for f in anomaly_files:
        tasks.append((f, "anomaly", True))

    print(f"Found {len(orig_voice_files)} valid voice files.")
    print(f"Found {len(anomaly_files)} raw anomaly files.")
    print(f"Processing and augmenting... This will take a few minutes.")

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_file, tasks)

    num_valid = len(list(OUT_VALID.glob("*.wav")))
    num_anomaly = len(list(OUT_ANOMALY.glob("*.wav")))
    
    print("\nDataset Generation Complete!")
    print(f"Valid files generated:   {num_valid}")
    print(f"Anomaly files generated: {num_anomaly}")
    print("="*60)

if __name__ == "__main__":
    main()
