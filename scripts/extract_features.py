"""
Pre-extract Features — runs ONCE, saves all mel spectrograms as .npy files.
Training then loads cached tensors instantly instead of re-processing audio each epoch.

Usage:
    python scripts/extract_features.py
"""

import sys, json
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import PROCESSED_DIR, SAMPLE_RATE, WINDOW_SAMPLES
from ml.preprocessing.audio_processor import AudioProcessor
from ml.preprocessing.feature_extractor import FeatureExtractor

CACHE_DIR = PROJECT_ROOT / "data" / "feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

ap = AudioProcessor()
fe = FeatureExtractor()


def extract_and_cache(entry, idx, total):
    path = entry["path"]
    cache_key = Path(path).stem + "_" + str(abs(hash(path)) % 1000000)
    cache_path = CACHE_DIR / f"{cache_key}.npy"

    if cache_path.exists():
        return cache_path, True  # already cached

    try:
        audio = ap.load_audio(path)
        audio = ap.pad_or_trim(audio, WINDOW_SAMPLES)
        feat  = fe.audio_to_tensor(audio)           # (1, n_mels, time)
        np.save(str(cache_path), feat.numpy())
        return cache_path, False
    except Exception as e:
        print(f"  ⚠ Failed {Path(path).name}: {e}")
        return None, False


def main():
    print("=" * 60)
    print("  Pre-extracting Features → Feature Cache")
    print("=" * 60)

    with open(PROCESSED_DIR / "manifest.json") as f:
        manifest = json.load(f)

    all_entries = []
    for split in ["train", "val", "test"]:
        all_entries.extend(manifest.get(split, []))

    total    = len(all_entries)
    cached   = 0
    fresh    = 0
    failed   = 0

    print(f"\n  Files to process: {total}")
    print(f"  Cache dir: {CACHE_DIR}\n")

    # Build updated manifest with cache paths
    new_manifest = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        entries = manifest.get(split, [])
        for i, entry in enumerate(entries):
            cache_path, was_cached = extract_and_cache(entry, i, total)
            if cache_path:
                new_manifest[split].append({
                    "path":       entry["path"],
                    "cache_path": str(cache_path),
                    "labels":     entry["labels"],
                })
                if was_cached: cached += 1
                else:          fresh  += 1
            else:
                failed += 1
            if (fresh + cached) % 50 == 0:
                print(f"  Progress: {fresh+cached}/{total}  "
                      f"(new={fresh}, reused={cached}, failed={failed})")

    print(f"\n  ✓ Done: {fresh} extracted, {cached} reused, {failed} failed")

    # Save updated manifest with cache paths
    cache_manifest = PROCESSED_DIR / "manifest_cached.json"
    with open(cache_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)
    print(f"  ✓ Cached manifest: {cache_manifest}")
    print(f"\n{'='*60}\n  Run next: python scripts/train.py --cached\n{'='*60}\n")


if __name__ == "__main__":
    main()
