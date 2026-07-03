"""
Trainer — Handles model training, validation, checkpointing, and evaluation.

Implements a complete training loop with:
  - Mixed precision training (AMP)
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing (saves best model)
  - Training metrics logging
  - Evaluation with precision, recall, F1
"""

import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from ml.config import (
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    MODEL_DIR,
    LOG_DIR,
    CONFIDENCE_THRESHOLD,
    NUM_CLASSES,
)
from ml.models.crnn_model import ActionCRNN
from ml.training.dataset import AudioActionDataset


class Trainer:
    """
    Trains and evaluates the ActionCRNN model.

    Usage:
        trainer = Trainer(model, train_dataset, val_dataset)
        history = trainer.train()
        metrics = trainer.evaluate(test_dataset)
    """

    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        device: Optional[str] = None,
        learning_rate: float = LEARNING_RATE,
        batch_size: int = BATCH_SIZE,
        model_type: str = "transformer",
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.model_type = model_type

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # CrossEntropy for single-label classification (one action per window)
        class_weights = train_dataset.get_class_weights().to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=WEIGHT_DECAY,
        )

        # LR Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=LR_SCHEDULER_PATIENCE,
            factor=LR_SCHEDULER_FACTOR,
        )

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": [], "val_f1": []}

        print(f"[Trainer] Device: {self.device}")
        print(f"[Trainer] Model parameters: {model.get_num_parameters():,}")
        print(f"[Trainer] Train samples: {len(train_dataset)}")
        print(f"[Trainer] Val samples: {len(val_dataset)}")

    def train(self, num_epochs: int = NUM_EPOCHS) -> Dict:
        """
        Full training loop with validation and early stopping.

        Returns:
            Training history dictionary.
        """
        print(f"\n{'='*60}")
        print(f"  Starting Training — {num_epochs} epochs")
        print(f"{'='*60}\n")

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            # Train one epoch
            train_loss = self._train_epoch()

            # Validate
            val_loss, val_metrics = self._validate_epoch()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Log history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_f1"].append(val_metrics["f1_macro"])

            elapsed = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{num_epochs} │ "
                f"Train Loss: {train_loss:.4f} │ "
                f"Val Loss: {val_loss:.4f} │ "
                f"Val Acc: {val_metrics['accuracy']:.1%} │ "
                f"Val F1: {val_metrics['f1_macro']:.3f} │ "
                f"LR: {lr:.2e} │ {elapsed:.1f}s"
            )

            # Checkpointing
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics)
                print(f"  ✓ Best model saved (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n  ✗ Early stopping at epoch {epoch}")
                break

        # Save training history
        self._save_history()
        print(f"\n{'='*60}")
        print(f"  Training Complete — Best Val Loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")

        return self.history

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels   = labels.to(self.device)
            # Convert multi-hot → class index for CrossEntropy
            label_indices = labels.argmax(dim=1)

            self.optimizer.zero_grad()
            logits = self.model(features)
            loss   = self.criterion(logits, label_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def _validate_epoch(self) -> tuple:
        """Run validation. Returns (avg_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for features, labels in self.val_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(features)
            # Convert multi-hot labels to class indices for CrossEntropy
            label_indices = labels.argmax(dim=1)
            loss = self.criterion(logits, label_indices)
            total_loss += loss.item()

            # Single-label prediction via argmax
            preds_idx = logits.argmax(dim=1).cpu().numpy()
            true_idx  = label_indices.cpu().numpy()
            all_preds.append(preds_idx)
            all_labels.append(true_idx)

        avg_loss = total_loss / max(len(self.val_loader), 1)

        all_preds  = np.concatenate(all_preds,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        from sklearn.metrics import accuracy_score
        acc = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        metrics = {
            "accuracy":         float(acc),
            "precision_macro":  float(precision),
            "recall_macro":     float(recall),
            "f1_macro":         float(f1),
        }
        return avg_loss, metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint with architecture info."""
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch":              epoch,
            "model_state_dict":   self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss":           self.best_val_loss,
            "metrics":            metrics,
            "num_classes":        self.model.num_classes,
            "model_type":         self.model_type,
        }
        torch.save(checkpoint, MODEL_DIR / "best_model.pth")

    def _save_history(self):
        """Save training history to JSON."""
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_DIR / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

    @torch.no_grad()
    def evaluate(self, test_dataset: AudioActionDataset) -> Dict:
        """
        Evaluate the model on a test dataset and print detailed metrics.

        Returns:
            Dictionary of evaluation metrics.
        """
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.model.eval()

        all_preds = []
        all_labels = []

        for features, labels in test_loader:
            features    = features.to(self.device)
            logits      = self.model(features)
            preds_idx   = logits.argmax(dim=1).cpu().numpy()
            true_idx    = labels.argmax(dim=1).numpy()
            all_preds.append(preds_idx)
            all_labels.append(true_idx)

        all_preds  = np.concatenate(all_preds,  axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        from sklearn.metrics import accuracy_score
        precision, recall, f1, support = precision_recall_fscore_support(
            all_labels, all_preds, average=None, zero_division=0
        )
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="macro", zero_division=0
        )
        acc = accuracy_score(all_labels, all_preds)

        print(f"\n{'='*60}")
        print(f"  Evaluation Results  (Accuracy: {acc:.1%})")
        print(f"{'='*60}")
        print(f"  {'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'-'*50}")

        idx_to_label = test_dataset.idx_to_label
        for i in range(len(precision)):
            name = idx_to_label.get(i, f"class_{i}")
            print(f"  {name:<20} {precision[i]:>10.4f} {recall[i]:>10.4f} {f1[i]:>10.4f}")

        print(f"  {'-'*50}")
        print(f"  {'MACRO AVG':<20} {p_macro:>10.4f} {r_macro:>10.4f} {f1_macro:>10.4f}")
        print(f"{'='*60}\n")

        return {
            "accuracy":           float(acc),
            "precision_per_class":precision.tolist(),
            "recall_per_class":   recall.tolist(),
            "f1_per_class":       f1.tolist(),
            "precision_macro":    float(p_macro),
            "recall_macro":       float(r_macro),
            "f1_macro":           float(f1_macro),
        }

    @staticmethod
    def load_model(model_path: str, num_classes: int = NUM_CLASSES):
        """Load a trained model — handles both transformer and crnn checkpoints."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        nc         = checkpoint.get("num_classes", num_classes)
        arch       = checkpoint.get("model_type", "crnn")
        if arch == "transformer":
            from ml.models.audio_transformer import AudioSpectrogramTransformer
            model = AudioSpectrogramTransformer(num_classes=nc)
        else:
            model = ActionCRNN(num_classes=nc)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[Trainer] Loaded {arch} model from {model_path} (epoch {checkpoint.get('epoch','?')})")
        return model
