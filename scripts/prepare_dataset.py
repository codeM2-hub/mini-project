"""
Prepare Dataset from voice_data/ and sequence_action_data/
===========================================================
Reads:
  voice_data/            → class: voice_normal (920 files, already augmented)
  sequence_action_data/  → classes: sequence_63..67, oyo_hotel (1000 files)

Splits into train(70%) / val(15%) / test(15%) with stratification.
Saves manifest.json and labels.json.
"""

import sys, json, random
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

VOICE_DIR  = PROJECT_ROOT / "voice_data"
ACTION_DIR = PROJECT_ROOT / "sequence_action_data"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SEED = 42
random.seed(SEED)


def get_action_class(fname):
    n = fname.lower()
    if '4th cross road 63' in n: return 'sequence_63'
    if '4th cross road 64' in n: return 'sequence_64'
    if '4th cross road 65' in n: return 'sequence_65'
    if '4th cross road 66' in n: return 'sequence_66'
    if '4th cross road 67' in n: return 'sequence_67'
    if 'oyo hotel'         in n: return 'oyo_hotel'
    return None


def main():
    print("=" * 60)
    print("  Preparing Dataset")
    print("  voice_data/ + sequence_action_data/ → manifest.json")
    print("=" * 60)

    # ── Collect all samples ────────────────────────────────
    all_samples = []  # list of (file_path, class_name)

    # voice_data
    voice_files = [f for f in sorted(VOICE_DIR.iterdir())
                   if f.suffix.lower() in ('.wav', '.mp3', '.m4a', '.flac')]
    for f in voice_files:
        all_samples.append((str(f), 'voice_normal'))
    print(f"\n  voice_data/:           {len(voice_files)} files  → voice_normal")

    # sequence_action_data
    action_files_by_class = defaultdict(list)
    for f in sorted(ACTION_DIR.iterdir()):
        if f.suffix.lower() not in ('.wav', '.mp3', '.m4a', '.flac'):
            continue
        cls = get_action_class(f.name)
        if cls:
            action_files_by_class[cls].append(f)

    print(f"  sequence_action_data/: breakdown:")
    for cls in sorted(action_files_by_class):
        n = len(action_files_by_class[cls])
        all_samples.extend([(str(f), cls) for f in action_files_by_class[cls]])
        print(f"    {cls:15}: {n} files")

    total = len(all_samples)
    print(f"\n  Total samples: {total}")

    # ── Build label maps ───────────────────────────────────
    all_classes = sorted(set(cls for _, cls in all_samples))
    label_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    idx_to_label = {str(i): cls for cls, i in label_to_idx.items()}
    num_classes  = len(all_classes)
    print(f"  Classes ({num_classes}): {all_classes}")

    # ── Stratified split ───────────────────────────────────
    paths   = [s[0] for s in all_samples]
    classes = [s[1] for s in all_samples]

    train_p, temp_p, train_c, temp_c = train_test_split(
        paths, classes, test_size=0.30, stratify=classes, random_state=SEED
    )
    val_p, test_p, val_c, test_c = train_test_split(
        temp_p, temp_c, test_size=0.50, stratify=temp_c, random_state=SEED
    )

    def make_entries(paths, classes):
        return [{"path": p, "labels": [label_to_idx[c]]} for p, c in zip(paths, classes)]

    manifest = {
        "train": make_entries(train_p, train_c),
        "val":   make_entries(val_p,   val_c),
        "test":  make_entries(test_p,  test_c),
    }

    print(f"\n  Split: train={len(manifest['train'])} | val={len(manifest['val'])} | test={len(manifest['test'])}")

    # Per-class in each split
    for split in ['train', 'val', 'test']:
        c = Counter(idx_to_label[str(e['labels'][0])] for e in manifest[split])
        print(f"  {split:5}: { {k:v for k,v in sorted(c.items())} }")

    # ── Save ───────────────────────────────────────────────
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    labels_path = PROJECT_ROOT / "data" / "labels.json"
    with open(labels_path, "w") as f:
        json.dump({
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
            "num_classes":  num_classes,
        }, f, indent=2)

    print(f"\n  ✓ manifest.json saved → {manifest_path}")
    print(f"  ✓ labels.json saved   → {labels_path}")
    print(f"\n{'='*60}")
    print(f"  Next: python scripts/extract_features.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
