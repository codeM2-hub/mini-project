import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import librosa

from ml.config import (
    SAMPLE_RATE, N_MELS, N_FFT, HOP_LENGTH,
    NORMAL_SEQUENCE_DIR, ANOMALY_SEQUENCE_DIR, SEQUENCE_MODEL_PATH,
    SEQ_NUM_EPOCHS, SEQ_BATCH_SIZE, SEQ_LEARNING_RATE
)
from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.models.sequence_anomaly_model import SequenceAnomalyModel

# ─── Dataset Class ───────────────────────────────────────

class SequenceAudioDataset(Dataset):
    """
    Efficient Dataset that stores file paths and offsets instead of raw audio.
    """
    def __init__(self, file_paths, labels, window_size_sec=5.0, hop_size_sec=2.5):
        self.extractor = FeatureExtractor()
        self.window_size = int(window_size_sec * SAMPLE_RATE)
        self.hop_size = int(hop_size_sec * SAMPLE_RATE)
        
        self.samples = []
        print(f"🔍 Indexing {len(file_paths)} files...")
        
        for path, label in tqdm(zip(file_paths, labels), total=len(file_paths)):
            try:
                # Just get duration to calculate number of windows
                duration = librosa.get_duration(path=path)
                total_samples = int(duration * SAMPLE_RATE)
                
                for start_sample in range(0, total_samples - self.window_size, self.hop_size):
                    self.samples.append({
                        'path': path,
                        'start': start_sample,
                        'label': label
                    })
            except Exception as e:
                pass # Skip broken files

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            # Load only the required segment
            y, _ = librosa.load(
                sample['path'], 
                sr=SAMPLE_RATE, 
                offset=sample['start']/SAMPLE_RATE, 
                duration=5.0
            )
            
            # Pad if shorter (shouldn't happen with our indexing but to be safe)
            if len(y) < self.window_size:
                y = np.pad(y, (0, self.window_size - len(y)))
            
            # Extract features
            mel = self.extractor.extract_mel_spectrogram(y)
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0)
            return mel_tensor, torch.FloatTensor([sample['label']])
        except:
            # Return silence on error
            return torch.zeros((1, N_MELS, 431)), torch.FloatTensor([sample['label']])

# ─── Training Pipeline ────────────────────────────────────

def train_model():
    normal_files = glob.glob(os.path.join(NORMAL_SEQUENCE_DIR, "*.wav"))
    anomaly_files = glob.glob(os.path.join(ANOMALY_SEQUENCE_DIR, "*.wav"))
    
    # Take a representative subset if too large
    MAX_FILES = 300
    normal_files = normal_files[:MAX_FILES]
    anomaly_files = anomaly_files[:MAX_FILES]
    
    all_files = normal_files + anomaly_files
    all_labels = [0] * len(normal_files) + [1] * len(anomaly_files)
    
    train_f, test_f, train_l, test_l = train_test_split(all_files, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    val_f, test_f, val_l, test_l = train_test_split(test_f, test_l, test_size=0.5, random_state=42, stratify=test_l)
    
    train_ds = SequenceAudioDataset(train_f, train_l)
    val_ds   = SequenceAudioDataset(val_f, val_l)
    test_ds  = SequenceAudioDataset(test_f, test_l)
    
    # Use multiple workers for faster data loading
    train_loader = DataLoader(train_ds, batch_size=SEQ_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=SEQ_BATCH_SIZE)
    test_loader  = DataLoader(test_ds, batch_size=SEQ_BATCH_SIZE)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on {device} | Total Samples: {len(train_ds)}")
    
    model = SequenceAnomalyModel(n_mels=N_MELS).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=SEQ_LEARNING_RATE)
    
    for epoch in range(SEQ_NUM_EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{SEQ_NUM_EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
        
        print(f"✅ Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        torch.save(model.state_dict(), SEQUENCE_MODEL_PATH)

    # Final metrics
    print("\n🏁 Final Evaluation...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs.to(device))
            y_true.extend(targets.numpy().flatten())
            y_pred.extend((torch.sigmoid(outputs) > 0.5).cpu().numpy().flatten())
    
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(SEQUENCE_MODEL_PATH), exist_ok=True)
    train_model()
