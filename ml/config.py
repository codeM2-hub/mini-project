"""
Configuration — Central hyperparameters and settings for the entire ML pipeline.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ============================================
# Path Configuration
# ============================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DATASET_DIR = PROJECT_ROOT / "dataset"
NORMAL_DIR = PROJECT_ROOT / "normal"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"

for d in [PROCESSED_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================
# Audio Configuration
# ============================================
SAMPLE_RATE = 22050
WINDOW_DURATION = 1.0        # 1-second windows
HOP_DURATION = 0.5           # 0.5s hop = 2 predictions per second
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION)
HOP_SAMPLES = int(SAMPLE_RATE * HOP_DURATION)
MAX_AUDIO_DURATION = 10.0

# ============================================
# Feature Extraction
# ============================================
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64                  # Reduced for speed
N_MFCC = 40
FMIN = 20
FMAX = 8000
TIME_FRAMES = int(WINDOW_SAMPLES / HOP_LENGTH) + 1

# ============================================
# Model Architecture
# ============================================
CNN_CHANNELS = [1, 32, 64, 128]   # Lighter model for small dataset
CNN_KERNEL_SIZE = 3
CNN_POOL_SIZE = (2, 2)
CNN_DROPOUT = 0.2

RNN_TYPE = "GRU"
RNN_HIDDEN_SIZE = 64
RNN_NUM_LAYERS = 1             # 1 layer for small dataset
RNN_BIDIRECTIONAL = True
RNN_DROPOUT = 0.3

USE_ATTENTION = True
CLASSIFIER_HIDDEN = 64
CLASSIFIER_DROPOUT = 0.4

# ============================================
# Training Configuration
# ============================================
BATCH_SIZE = 16                # Small batch for small dataset
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 60
EARLY_STOPPING_PATIENCE = 15
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

# ============================================
# Inference Configuration
# ============================================
CONFIDENCE_THRESHOLD = 0.45   # Lower threshold = more sensitive
SMOOTHING_WINDOW = 2
MIN_ACTION_DURATION = 0.3
OVERLAP_IOU_THRESHOLD = 0.3

# ============================================
# Data Augmentation
# ============================================
AUGMENTATION_ENABLED = True
AUG_NOISE_FACTOR = 0.003
AUG_PITCH_SHIFT_RANGE = (-2, 2)
AUG_TIME_STRETCH_RANGE = (0.85, 1.15)
AUG_VOLUME_RANGE = (0.6, 1.4)
AUG_PROBABILITY = 0.5

# ============================================
# Action Labels
# ============================================
ACTION_LABELS = {
    0: "sequence_63",
    1: "sequence_64",
    2: "sequence_65",
    3: "sequence_66",
    4: "sequence_67",
    5: "oyo_hotel",
    6: "voice_normal",
}
NUM_CLASSES = len(ACTION_LABELS)

# ============================================
# Server Configuration
# ============================================
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ============================================
# Email Configuration
# ============================================
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", "sachinsingh454751@gmail.com")

# ============================================
# Sequence Model Configuration (Feature 5)
# ============================================
SEQUENCE_MODEL_PATH = MODEL_DIR / "sequence_anomaly_model.pth"
NORMAL_SEQUENCE_DIR = PROJECT_ROOT / "sequence_action_data"
ANOMALY_SEQUENCE_DIR = PROJECT_ROOT / "anamoly_action_sequence_data_augmentated"

# Training params
SEQ_NUM_EPOCHS = 20
SEQ_BATCH_SIZE = 16
SEQ_LEARNING_RATE = 0.0001
SEQ_HIDDEN_SIZE = 128
SEQ_NUM_LAYERS = 2
SEQ_CONFIDENCE_THRESHOLD = 0.5
