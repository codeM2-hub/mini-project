"""
CORRECT Augmentation Pipeline
==============================
CRITICAL FIX: Split ORIGINALS first, THEN augment only the TRAIN split.
This prevents data leakage (same recording appearing in train AND test).

Previous bug: augmented all 150 files, then split → test had augmented
copies of files already in train → fake 100% accuracy.

Correct approach:
  1. Find all UNIQUE original recordings per class
  2. Split originals: 70% train / 15% val / 15% test
  3. Augment ONLY train originals → 500 samples per class
  4. Val + test keep original files only (honest evaluation)

Augmentation count: 60+ techniques applied to each train original.
With 3-5 originals × 60-100 augmentations = 180-500 diverse samples.
"""

import sys, json, re, shutil, random
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import librosa
import soundfile as sf
from scipy.signal import lfilter, butter

from ml.config import (
    DATASET_DIR, NORMAL_DIR, PROCESSED_DIR, SAMPLE_RATE,
    TRAIN_SPLIT, VAL_SPLIT, PROJECT_ROOT
)

# ── Config ─────────────────────────────────────────────────
TARGET_TRAIN_PER_CLASS = 400   # augmented training samples per class
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".flac"}
AUG_DIR    = PROJECT_ROOT / "data" / "augmented_v2"
SEED       = 42
random.seed(SEED); np.random.seed(SEED)


# ── Class detection ────────────────────────────────────────
def get_class(fname):
    n = fname.lower()
    if '4th cross road 63' in n: return 'sequence_63'
    if '4th cross road 64' in n: return 'sequence_64'
    if '4th cross road 65' in n: return 'sequence_65'
    if '4th cross road 66' in n: return 'sequence_66'
    if '4th cross road 67' in n: return 'sequence_67'
    if 'oyo hotel'         in n: return 'oyo_hotel'
    return None

def get_base_name(fname):
    """Strip augmentation suffixes to get the original recording name."""
    n = Path(fname).stem
    n = re.sub(r'[_\s]*(eq|compressed|slow|fast|converted|vol|aug|noise|reverb|pitch|speed|rev|band|pink|echo)[^a-zA-Z0-9]*.*$', '', n, flags=re.I)
    n = re.sub(r'[_\s]*\(\d+\)[_\s]*$', '', n).strip()
    n = re.sub(r'[_\s]+$', '', n).strip()
    return n.lower()


# ── Audio loading ──────────────────────────────────────────
def load_audio(path, sr=SAMPLE_RATE, max_dur=12.0):
    try:
        y, _ = librosa.load(str(path), sr=sr, mono=True, duration=max_dur)
        if len(y) < sr * 0.5: return None
        peak = np.max(np.abs(y))
        if peak < 1e-6: return None
        return (y / peak * 0.9).astype(np.float32)
    except Exception:
        return None

def save_wav(audio, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.astype(np.float32), SAMPLE_RATE)


# ── Augmentation bank (60+ variants) ──────────────────────
def aug_speed(y, sr, r): return librosa.effects.time_stretch(y, rate=r)
def aug_pitch(y, sr, s): return librosa.effects.pitch_shift(y, sr=sr, n_steps=s)
def aug_noise(y, level): return np.clip(y + np.random.randn(len(y)).astype(np.float32)*level, -1, 1)
def aug_vol(y, g): return np.clip(y*g, -1, 1)

def aug_pink(y, level=0.004):
    w = np.random.randn(len(y))
    b = np.array([0.049922035,-0.095993537,0.050612699,-0.004408786])
    a = np.array([1,-2.494956002,2.017265875,-0.522189400])
    p = lfilter(b,a,w).astype(np.float32)
    p /= (np.max(np.abs(p))+1e-8)
    return np.clip(y + p*level, -1, 1)

def aug_reverb(y, sr, decay=0.3, delay_ms=30):
    d = int(sr*delay_ms/1000)
    echo = np.zeros_like(y)
    if d < len(y): echo[d:] = y[:-d]*decay
    return np.clip(y+echo, -1, 1)

def aug_lowpass(y, sr, cutoff=3000):
    b,a = butter(4, cutoff/(sr/2), btype='low')
    return np.clip(lfilter(b,a,y).astype(np.float32), -1, 1)

def aug_highpass(y, sr, cutoff=300):
    b,a = butter(4, cutoff/(sr/2), btype='high')
    return np.clip(lfilter(b,a,y).astype(np.float32), -1, 1)

def aug_bandpass(y, sr, lo=500, hi=4000):
    nyq = sr/2
    b,a = butter(4, [lo/nyq, hi/nyq], btype='band')
    r = lfilter(b,a,y).astype(np.float32)
    p = np.max(np.abs(r))
    return (r/(p+1e-8)*0.9) if p>0 else r

def aug_reverse(y): return y[::-1].copy()

def aug_eq_boost(y, sr, center=1000, gain=3.0):
    """Simple mid-frequency boost."""
    b,a = butter(2, [center*0.7/(sr/2), center*1.4/(sr/2)], btype='band')
    mid = lfilter(b,a,y).astype(np.float32)
    return np.clip(y + mid*gain*0.1, -1, 1)

def aug_trim_start(y, sr, trim_s=0.5):
    s = int(sr*trim_s)
    return y[s:] if len(y)>s*2 else y

def aug_trim_end(y, sr, trim_s=0.5):
    s = int(sr*trim_s)
    return y[:-s] if len(y)>s*2 else y

def aug_chunk(y, sr, start_s=1.0, dur_s=5.0):
    s,e = int(sr*start_s), int(sr*(start_s+dur_s))
    return y[s:e] if len(y)>e else y

def aug_combined(y, sr, fns):
    for fn in fns:
        try: y = fn(y, sr)
        except: pass
    return y


def get_all_augmentations(sr=SAMPLE_RATE):
    """Returns list of (name, fn(y,sr)->y) pairs — 60+ variants."""
    augs = []
    # Speed
    for r in [0.70, 0.75, 0.80, 0.85, 0.88, 0.92, 1.08, 1.12, 1.15, 1.20, 1.25, 1.30]:
        augs.append((f"spd{int(r*100)}", lambda y,sr,_r=r: aug_speed(y,sr,_r)))
    # Pitch
    for s in [-4,-3,-2,-1,1,2,3,4]:
        augs.append((f"pit{s:+d}", lambda y,sr,_s=s: aug_pitch(y,sr,_s)))
    # Noise levels
    for lv in [0.002,0.004,0.007,0.010,0.015,0.020]:
        augs.append((f"wn{int(lv*1000)}", lambda y,sr,_lv=lv: aug_noise(y,_lv)))
    # Pink noise
    for lv in [0.002,0.005,0.008]:
        augs.append((f"pn{int(lv*1000)}", lambda y,sr,_lv=lv: aug_pink(y,_lv)))
    # Volume
    for g in [0.40,0.55,0.70,0.85,1.20,1.35,1.50]:
        augs.append((f"vol{int(g*100)}", lambda y,sr,_g=g: aug_vol(y,_g)))
    # Reverb
    for dec,dms in [(0.2,20),(0.3,30),(0.4,50),(0.5,70),(0.6,100)]:
        augs.append((f"rev{int(dec*10)}d{dms}", lambda y,sr,_d=dec,_ms=dms: aug_reverb(y,sr,_d,_ms)))
    # Filters
    augs.append(("lp2k", lambda y,sr: aug_lowpass(y,sr,2000)))
    augs.append(("lp3k", lambda y,sr: aug_lowpass(y,sr,3000)))
    augs.append(("lp4k", lambda y,sr: aug_lowpass(y,sr,4000)))
    augs.append(("hp150", lambda y,sr: aug_highpass(y,sr,150)))
    augs.append(("hp300", lambda y,sr: aug_highpass(y,sr,300)))
    augs.append(("bp5004k", lambda y,sr: aug_bandpass(y,sr,500,4000)))
    augs.append(("bp10008k", lambda y,sr: aug_bandpass(y,sr,1000,8000)))
    augs.append(("eqb1k",  lambda y,sr: aug_eq_boost(y,sr,1000)))
    augs.append(("eqb3k",  lambda y,sr: aug_eq_boost(y,sr,3000)))
    # Trim
    augs.append(("trims",  lambda y,sr: aug_trim_start(y,sr,0.3)))
    augs.append(("trime",  lambda y,sr: aug_trim_end(y,sr,0.3)))
    augs.append(("rev",    lambda y,sr: aug_reverse(y)))
    # Combinations
    augs.append(("spd85_pit+2", lambda y,sr: aug_pitch(aug_speed(y,sr,0.85),sr,2)))
    augs.append(("spd115_pit-2",lambda y,sr: aug_pitch(aug_speed(y,sr,1.15),sr,-2)))
    augs.append(("spd90_wn",   lambda y,sr: aug_noise(aug_speed(y,sr,0.90),0.005)))
    augs.append(("pit+3_rev",  lambda y,sr: aug_reverb(aug_pitch(y,sr,3),sr,0.3,40)))
    augs.append(("wn_rev",     lambda y,sr: aug_reverb(aug_noise(y,0.005),sr,0.25,30)))
    augs.append(("vol60_pit+1",lambda y,sr: aug_pitch(aug_vol(y,0.6),sr,1)))
    augs.append(("rev_wn_lp",  lambda y,sr: aug_lowpass(aug_noise(aug_reverb(y,sr,0.3,30),0.004),sr,4000)))
    augs.append(("spd80_pit-3_wn",lambda y,sr: aug_noise(aug_pitch(aug_speed(y,sr,0.80),sr,-3),0.006)))
    augs.append(("spd120_pit+2_pn",lambda y,sr: aug_pink(aug_pitch(aug_speed(y,sr,1.20),sr,2),0.005)))

    return augs


# ── Find originals ─────────────────────────────────────────
def find_originals(folder):
    """
    Return dict: class_name → list of ORIGINAL file paths.
    'Original' = one representative file per base recording name.
    Prefers .wav over .m4a; picks first alphabetically.
    """
    by_base = defaultdict(list)
    for f in sorted(Path(folder).iterdir()):
        if f.suffix.lower() not in AUDIO_EXTS: continue
        cls = get_class(f.name)
        if not cls: continue
        base = get_base_name(f.name)
        by_base[(cls, base)].append(f)

    result = defaultdict(list)
    for (cls, base), files in by_base.items():
        # Prefer .wav
        wavs = [f for f in files if f.suffix.lower() == '.wav']
        chosen = wavs[0] if wavs else files[0]
        result[cls].append(chosen)
    return result


# ── Augment train originals ────────────────────────────────
def augment_originals(class_name, orig_files, target_n, out_dir):
    """
    Create target_n augmented WAV files from orig_files.
    Each augmentation is applied to a random original.
    Returns list of output paths.
    """
    out_class = out_dir / class_name
    out_class.mkdir(parents=True, exist_ok=True)

    # Load all originals
    loaded = []
    for src in orig_files:
        a = load_audio(src)
        if a is not None:
            loaded.append((src.stem, a))
    if not loaded:
        return []

    augs = get_all_augmentations()
    out_paths = []
    idx = 0

    # Copy originals first
    for stem, audio in loaded:
        dst = out_class / f"{stem}_orig.wav"
        save_wav(audio, dst)
        out_paths.append(str(dst))

    # Generate augmented samples
    needed = target_n - len(out_paths)
    aug_i = 0
    while len(out_paths) < target_n:
        stem, audio = loaded[aug_i % len(loaded)]
        aug_name, aug_fn = augs[aug_i % len(augs)]
        aug_i += 1
        try:
            aug_audio = aug_fn(audio, SAMPLE_RATE)
            aug_audio = aug_audio.astype(np.float32)
            peak = np.max(np.abs(aug_audio))
            if peak > 1e-6: aug_audio = aug_audio / peak * 0.88
            dst = out_class / f"{stem}_{aug_name}_{len(out_paths):04d}.wav"
            save_wav(aug_audio, dst)
            out_paths.append(str(dst))
        except Exception:
            pass
        if aug_i > len(augs) * len(loaded) * 5:
            break  # safety

    return out_paths


# ── Main ───────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("  CORRECT Augmentation Pipeline (no data leakage)")
    print("=" * 68)

    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    AUG_DIR.mkdir(parents=True)

    # ── Step 1: Find all originals ─────────────────────────
    print("\n[1/5] Finding unique original recordings...")
    action_originals = find_originals(DATASET_DIR)

    # Normal: deduplicate by base name
    normal_by_base = {}
    for f in sorted(NORMAL_DIR.iterdir()):
        if f.suffix.lower() not in AUDIO_EXTS: continue
        base = get_base_name(f.name)
        if base not in normal_by_base:
            normal_by_base[base] = f
        elif f.suffix.lower() == '.wav':
            normal_by_base[base] = f  # prefer .wav

    normal_originals = list(normal_by_base.values())

    print(f"\n  Action class originals:")
    for cls in sorted(action_originals):
        print(f"    {cls:15}: {len(action_originals[cls]):3d} unique recordings")
    print(f"\n  voice_normal originals: {len(normal_originals)}")

    # ── Step 2: Split ORIGINALS first ─────────────────────
    print("\n[2/5] Splitting ORIGINALS into train/val/test (no leakage)...")

    label_to_idx = {cls: i for i, cls in enumerate(sorted(action_originals.keys()))}
    label_to_idx["voice_normal"] = len(label_to_idx)
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    num_classes  = len(label_to_idx)

    splits = {"train": [], "val": [], "test": []}

    # Split each action class independently
    for cls in sorted(action_originals.keys()):
        files = action_originals[cls]
        random.shuffle(files)
        n = len(files)
        n_val  = max(1, int(n * 0.15))
        n_test = max(1, int(n * 0.15))
        n_train = n - n_val - n_test

        if n_train < 1:  # very few originals — put at least 1 in train
            n_train, n_val, n_test = max(1,n-2), min(1,n-1), min(1,n-1) if n>1 else 0

        splits["train"].extend([(f, cls) for f in files[:n_train]])
        splits["val"].extend(  [(f, cls) for f in files[n_train:n_train+n_val]])
        splits["test"].extend( [(f, cls) for f in files[n_train+n_val:]])

    # Split normal class
    random.shuffle(normal_originals)
    n = len(normal_originals)
    nv = int(n*0.15); nt = int(n*0.15)
    splits["train"].extend([(f, "voice_normal") for f in normal_originals[:n-nv-nt]])
    splits["val"].extend(  [(f, "voice_normal") for f in normal_originals[n-nv-nt:n-nt]])
    splits["test"].extend( [(f, "voice_normal") for f in normal_originals[n-nt:]])

    for sp in splits:
        c = {}
        for _,cls in splits[sp]:
            c[cls] = c.get(cls,0)+1
        print(f"  {sp:5}: {len(splits[sp]):3d} originals — {c}")

    # ── Step 3: Augment ONLY train split ──────────────────
    print(f"\n[3/5] Augmenting TRAIN originals to {TARGET_TRAIN_PER_CLASS} per class...")

    # Group train files by class
    train_by_class = defaultdict(list)
    for f, cls in splits["train"]:
        train_by_class[cls].append(f)

    train_paths_by_class = {}
    for cls in sorted(train_by_class.keys()):
        orig_files = train_by_class[cls]
        paths = augment_originals(cls, orig_files, TARGET_TRAIN_PER_CLASS, AUG_DIR / "train")
        train_paths_by_class[cls] = paths
        print(f"  {cls:15}: {len(orig_files)} originals → {len(paths)} augmented")

    # ── Step 4: Copy val/test (no augmentation) ───────────
    print("\n[4/5] Copying val/test originals (no augmentation)...")
    val_paths_by_class  = defaultdict(list)
    test_paths_by_class = defaultdict(list)

    for sp, store in [("val", val_paths_by_class), ("test", test_paths_by_class)]:
        for src, cls in splits[sp]:
            dst = AUG_DIR / sp / cls / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            audio = load_audio(src)
            if audio is not None:
                save_wav(audio, dst)
                store[cls].append(str(dst))

    # ── Step 5: Build manifest ─────────────────────────────
    print("\n[5/5] Building manifest...")

    manifest = {"train": [], "val": [], "test": []}
    for cls, paths in train_paths_by_class.items():
        for p in paths:
            manifest["train"].append({"path": p, "labels": [label_to_idx[cls]]})

    for cls, paths in val_paths_by_class.items():
        for p in paths:
            manifest["val"].append({"path": p, "labels": [label_to_idx[cls]]})

    for cls, paths in test_paths_by_class.items():
        for p in paths:
            manifest["test"].append({"path": p, "labels": [label_to_idx[cls]]})

    random.shuffle(manifest["train"])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    labels_data = {"label_to_idx": label_to_idx, "idx_to_label": {str(v):k for k,v in label_to_idx.items()}, "num_classes": num_classes}
    with open(PROJECT_ROOT / "data" / "labels.json", "w") as f:
        json.dump(labels_data, f, indent=2)

    print(f"\n  Train: {len(manifest['train'])} | Val: {len(manifest['val'])} | Test: {len(manifest['test'])}")
    print(f"  Classes: {num_classes} — {list(label_to_idx.keys())}")
    print(f"\n{'='*68}")
    print(f"  Done! Train is AUGMENTED, Val+Test are ORIGINAL (honest eval)")
    print(f"  Next: python scripts/extract_features.py")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()
