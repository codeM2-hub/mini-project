# 🎙️ Real-Time Audio-Based Action Sequence Recognition System

> AI-powered system that continuously listens to audio and identifies specific actions and sequences of actions based on sound patterns — in real time.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Microphone  │────▶│  Audio Processor  │────▶│ Feature Extractor │
│  / File      │     │  (Normalize,      │     │ (Mel Spectrogram, │
│              │     │   Window, Trim)   │     │  MFCC, Chroma)    │
└──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                        │
                     ┌──────────────────┐     ┌────────▼─────────┐
                     │ Sequence Tracker  │◀────│   CRNN Model     │
                     │ (Timeline, Multi- │     │ (CNN + GRU +     │
                     │  label, Overlap)  │     │  Attention)      │
                     └────────┬─────────┘     └──────────────────┘
                              │
                     ┌────────▼─────────┐     ┌──────────────────┐
                     │  FastAPI Backend  │────▶│  React Dashboard  │
                     │  (REST + WS)     │     │ (Waveform, Timeline│
                     │                  │     │  Detections, Stats)│
                     └──────────────────┘     └──────────────────┘
```

## 📁 Project Structure

```
anti_gravity_mini_project/
├── ml/                              # 🧠 Machine Learning Pipeline
│   ├── config.py                    # Central configuration
│   ├── preprocessing/
│   │   ├── audio_processor.py       # Audio loading, normalization, windowing
│   │   ├── feature_extractor.py     # Mel spectrogram, MFCC, chroma extraction
│   │   └── augmentation.py          # Data augmentation (noise, pitch, stretch)
│   ├── models/
│   │   └── crnn_model.py            # CRNN architecture (CNN + GRU + Attention)
│   ├── training/
│   │   ├── dataset.py               # PyTorch Dataset with augmentation
│   │   └── trainer.py               # Training loop, validation, checkpointing
│   └── inference/
│       ├── realtime_detector.py     # Real-time sliding window detector
│       └── sequence_tracker.py      # Multi-action tracking & timeline
├── backend/                         # ⚡ FastAPI Server
│   ├── main.py                      # App entry point
│   ├── routes/
│   │   ├── audio.py                 # REST endpoints (upload, status, labels)
│   │   └── websocket_handler.py     # WebSocket for real-time streaming
│   └── services/
│       └── audio_service.py         # Business logic layer
├── frontend/                        # 🎨 React Dashboard (Vite)
│   └── src/
│       ├── App.jsx                  # Main dashboard with all components
│       └── index.css                # Premium dark theme styles
├── scripts/
│   ├── prepare_dataset.py           # Organize & label audio files
│   └── train.py                     # End-to-end training script
├── dataset/                         # 🔊 Action sequence audio (augmented)
├── normal/                          # 🗣️ Voice/speech audio
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Prepare Dataset

```bash
python scripts/prepare_dataset.py
```

This scans `dataset/` and `normal/` folders, assigns labels, and creates train/val/test splits.

### 3. Train the Model

```bash
python scripts/train.py --epochs 50 --batch-size 32
```

The best model is saved to `models/best_model.pth`.

### 4. Start the Backend

```bash
python -m backend.main
# Or: uvicorn backend.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 5. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

Dashboard available at `http://localhost:5173`

## 🧠 Model Architecture — CRNN

```
Input: Mel Spectrogram (1 × 128 × T)
    │
    ▼
┌─────────────────────────────┐
│  Conv Block 1 (64 filters)  │ → BatchNorm → ReLU → MaxPool → Dropout
│  Conv Block 2 (128 filters) │ → BatchNorm → ReLU → MaxPool → Dropout
│  Conv Block 3 (256 filters) │ → BatchNorm → ReLU → MaxPool → Dropout
└─────────────┬───────────────┘
              │ Reshape (batch, time', features)
              ▼
┌─────────────────────────────┐
│  Bidirectional GRU          │ 2 layers, hidden=128
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  Attention Layer             │ Focus on important time steps
└─────────────┬───────────────┘
              ▼
┌─────────────────────────────┐
│  FC(256→128) → ReLU → Drop  │
│  FC(128→N) → Sigmoid        │ Multi-label output
└─────────────────────────────┘
```

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Real-Time Processing** | Sliding window with 1s windows and 0.5s hop |
| **Multi-Label Detection** | Detects multiple overlapping actions simultaneously |
| **Sequence Tracking** | Identifies ordered sequences of actions |
| **Action Timeline** | Start time, end time, duration, and confidence per action |
| **Data Augmentation** | Noise, pitch shift, time stretch, SpecAugment |
| **Live Dashboard** | Waveform, spectrogram, detections, and timeline |
| **File Upload** | Drag-and-drop audio analysis |
| **WebSocket Streaming** | Real-time results pushed to frontend |

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload audio file for analysis |
| `GET` | `/api/status` | Get detector status |
| `GET` | `/api/labels` | Get available action labels |
| `GET` | `/api/timeline` | Get action timeline |
| `POST` | `/api/start-mic` | Start microphone capture |
| `POST` | `/api/stop-mic` | Stop microphone capture |
| `WS` | `/ws/audio` | Real-time audio streaming |
| `WS` | `/ws/status` | Live status updates |

## 🛠️ Tech Stack

- **ML**: PyTorch, Librosa, Scikit-learn, NumPy
- **Backend**: FastAPI, Uvicorn, WebSockets
- **Frontend**: React 18, Vite, Web Audio API
- **Audio**: sounddevice, soundfile, audioread

## 📝 License

MIT License — Built for learning, hackathons, and portfolio projects.
