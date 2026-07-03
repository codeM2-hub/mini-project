"""
Train Script — Trains Audio Spectrogram Transformer (AST).

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 100 --batch-size 64
    python scripts/train.py --model crnn       # use old CRNN instead
    python scripts/train.py --model transformer # use AST (default)
"""

import sys, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.config import (
    NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, PROCESSED_DIR,
    N_MELS, N_FFT, HOP_LENGTH, SAMPLE_RATE, WINDOW_SAMPLES, NUM_CLASSES,
)
from ml.training.cached_dataset import CachedAudioDataset
from ml.training.trainer import Trainer


def get_feature_shape():
    """Compute actual (n_mels, n_frames) from config."""
    import numpy as np
    import librosa
    hop = HOP_LENGTH
    n_fft = N_FFT
    n_mels = N_MELS
    # frames for one window
    n_frames = 1 + (WINDOW_SAMPLES) // hop
    return n_mels, n_frames


def main():
    parser = argparse.ArgumentParser(description="Train Audio Action Recognition model")
    parser.add_argument("--epochs",     type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=LEARNING_RATE)
    parser.add_argument("--model",      type=str,   default="transformer",
                        choices=["transformer", "crnn"],
                        help="Model architecture (default: transformer)")
    args = parser.parse_args()

    cached_manifest = PROCESSED_DIR / "manifest_cached.json"
    if not cached_manifest.exists():
        print("✗ Cached features not found! Run first:")
        print("    python scripts/extract_features.py")
        sys.exit(1)

    print("=" * 64)
    print(f"  Audio Action Recognition — {args.model.upper()} Training")
    print("=" * 64)

    # ── Datasets ───────────────────────────────────────────────
    print("\n[1/4] Loading cached feature datasets...")
    train_ds = CachedAudioDataset(str(cached_manifest), split="train", augment=True)
    val_ds   = CachedAudioDataset(str(cached_manifest), split="val",   augment=False)
    test_ds  = CachedAudioDataset(str(cached_manifest), split="test",  augment=False)

    num_classes = train_ds.num_classes
    print(f"  Classes: {num_classes} | Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ── Model ──────────────────────────────────────────────────
    print(f"\n[2/4] Creating {args.model.upper()} model...")

    if args.model == "transformer":
        from ml.models.audio_transformer import AudioSpectrogramTransformer
        n_mels, n_frames = get_feature_shape()
        # Check actual feature shape from one sample
        sample_feat, _ = train_ds[0]
        _, actual_mels, actual_frames = sample_feat.shape
        print(f"  Feature shape: (1, {actual_mels}, {actual_frames})")

        model = AudioSpectrogramTransformer(
            num_classes=num_classes,
            n_mels=actual_mels,
            n_frames=actual_frames,
            patch_h=8,
            patch_w=8,
            d_model=192,
            nhead=8,
            num_layers=4,
            dim_ff=512,
            dropout=0.15,
        )
        # Attach count helper so Trainer can print it
        model.get_num_parameters = model.count_params
        model_type = "transformer"

    else:
        from ml.models.crnn_model import ActionCRNN
        model = ActionCRNN(num_classes=num_classes)
        model_type = "crnn"

    print(f"  Parameters: {model.get_num_parameters():,}")

    # ── Train ──────────────────────────────────────────────────
    print(f"\n[3/4] Starting training ({args.epochs} epochs)...")
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        model_type=model_type,
    )
    trainer.train(num_epochs=args.epochs)

    # ── Evaluate ───────────────────────────────────────────────
    print(f"\n[4/4] Final evaluation on test set...")
    trainer.evaluate(test_ds)

    print(f"\n✓ Training complete! Model saved to models/best_model.pth")
    print(f"  Architecture: {args.model.upper()}")


if __name__ == "__main__":
    main()
