"""
Train Speaker Verification Model
--------------------------------
Extracts Mel Spectrograms from the speaker dataset.
Trains a robust CNN with Batch Normalization and Dropout.
Evaluates using Confusion Matrix, Precision, Recall, F1.
"""

import os
import sys
import glob
import time
import numpy as np
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from pathlib import Path

# Fix python path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.models.speaker_cnn import SpeakerCNN

# Config
DATA_DIR = PROJECT_ROOT / "data" / "speaker_dataset"
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Load Dataset class
class SpeakerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X) # (N, 1, Mels, Time)
        self.y = torch.LongTensor(y)  # (N,)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def extract_features():
    print("Extracting features (this may take a few minutes)...")
    extractor = FeatureExtractor()
    
    X = []
    y = []
    
    valid_files = list((DATA_DIR / "valid").glob("*.wav"))
    anomaly_files = list((DATA_DIR / "anomaly").glob("*.wav"))
    
    # 1 for valid, 0 for anomaly
    all_files = [(f, 1) for f in valid_files] + [(f, 0) for f in anomaly_files]
    
    # To save memory, we pad/trim to exact 3 seconds
    target_samples = int(22050 * 3.0)
    
    for i, (fpath, label) in enumerate(all_files):
        if i % 500 == 0:
            print(f"  Processed {i}/{len(all_files)} files...")
        
        y_audio, sr = librosa.load(fpath, sr=22050)
        if len(y_audio) < target_samples:
            y_audio = np.pad(y_audio, (0, target_samples - len(y_audio)))
        else:
            y_audio = y_audio[:target_samples]
            
        tensor = extractor.audio_to_tensor(y_audio) # (1, Mels, Time)
        X.append(tensor.numpy())
        y.append(label)
        
    X = np.stack(X)
    y = np.array(y)
    
    print(f"Features shape: {X.shape}, Labels shape: {y.shape}")
    return X, y


def main():
    # 1. Extract or Load Features
    cache_path = PROJECT_ROOT / "data" / "speaker_features.npz"
    if cache_path.exists():
        print("Loading cached features...")
        data = np.load(cache_path)
        X, y = data['X'], data['y']
    else:
        X, y = extract_features()
        print("Caching features...")
        np.savez(cache_path, X=X, y=y)
        
    # 2. Split dataset
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    train_dataset = SpeakerDataset(X_train, y_train)
    val_dataset = SpeakerDataset(X_val, y_val)
    test_dataset = SpeakerDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 3. Model setup
    model = SpeakerCNN(num_classes=2).to(DEVICE)
    
    # Weight handling for class imbalance
    num_valid = np.sum(y_train == 1)
    num_anomaly = np.sum(y_train == 0)
    total = len(y_train)
    weights = torch.tensor([total/num_anomaly, total/num_valid], dtype=torch.float32).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 4. Training Loop
    best_val_f1 = 0
    patience_counter = 0
    early_stop_limit = 7
    
    print("\nStarting Training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        all_preds = []
        all_true = []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                out = model(batch_X)
                loss = criterion(out, batch_y)
                val_loss += loss.item()
                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader)
        acc = accuracy_score(all_true, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_preds, average='macro', zero_division=0)
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_val_f1:
            best_val_f1 = f1
            patience_counter = 0
            torch.save(model.state_dict(), PROJECT_ROOT / "models" / "speaker_model.pth")
            print("  -> Saved new best model")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_limit:
            print("Early stopping triggered!")
            break

    # 5. Testing and Evaluation
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "speaker_model.pth"))
    model.eval()
    
    test_preds = []
    test_true = []
    test_probs = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            out = model(batch_X)
            probs = torch.softmax(out, dim=1)
            
            # Using specific confidence threshold > 0.92 for VALID (Class 1)
            # as per user requirement to reduce false positives aggressively
            for i in range(len(probs)):
                if probs[i][1].item() > 0.92:
                    test_preds.append(1)
                else:
                    test_preds.append(0)
                    
            test_true.extend(batch_y.cpu().numpy())

    cm = confusion_matrix(test_true, test_preds)
    p, r, f, _ = precision_recall_fscore_support(test_true, test_preds, average=None)
    
    print("\nConfusion Matrix (Rows=True, Cols=Predicted):")
    print("                Pred ANOMALY | Pred VALID")
    print(f"True ANOMALY  | {cm[0][0]:12d} | {cm[0][1]:10d}")
    print(f"True VALID    | {cm[1][0]:12d} | {cm[1][1]:10d}")
    
    print("\nClass-wise Metrics:")
    print("Class ANOMALY (0): Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(p[0], r[0], f[0]))
    print("Class VALID   (1): Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}".format(p[1], r[1], f[1]))
    
    print("\nOverall Accuracy: {:.4f}".format(accuracy_score(test_true, test_preds)))
    print("Model saved to: models/speaker_model.pth")

if __name__ == "__main__":
    main()
