"""
Speaker Verifier Inference Engine
---------------------------------
Real-time interface for validating the user's voice against anomalies.
Applies the aggressive 0.92 confidence threshold.
"""

import numpy as np
import torch
from pathlib import Path
from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.models.speaker_cnn import SpeakerCNN

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
MODEL_PATH = PROJECT_ROOT / "models" / "speaker_model.pth"

# 0.92 Threshold chosen to aggressively reduce false positives 
# (e.g. YouTube, TV, other random human voices)
CONFIDENCE_THRESHOLD = 0.92

class SpeakerVerifier:
    def __init__(self, model_path=str(MODEL_PATH)):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = FeatureExtractor()
        
        # Load the binary CNN model
        self.model = SpeakerCNN(num_classes=2).to(self.device)
        if Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.is_loaded = True
            print(f"[Verifier] ✓ Loaded SpeakerCNN from {model_path}")
        else:
            self.is_loaded = False
            print(f"[Verifier] ✗ Model not found at {model_path}. Train the model first.")

    @torch.no_grad()
    def verify(self, audio_window: np.ndarray, is_padded=False):
        """
        Takes a raw 1D numpy array audio window.
        Returns a dict with 'status' (VALID/ANOMALY), 'confidence', and 'probabilities'.
        """
        if not self.is_loaded:
            return {"status": "ERROR", "message": "Model not loaded"}

        # Expecting exactly 3 seconds for CNN. Pad/trim if needed.
        target_samples = int(self.extractor.sample_rate * 3.0)
        
        if len(audio_window) < target_samples:
            if not is_padded: # Only allow padding if we explicitly want to check short clips
               audio_window = np.pad(audio_window, (0, target_samples - len(audio_window)))
            else:
               pass
        else:
            audio_window = audio_window[:target_samples]

        # Check for silence (prevent model from guessing on dead air)
        rms = np.sqrt(np.mean(audio_window**2))
        if rms < 0.005:  # threshold for silence
            return {
                "status": "ANOMALY", 
                "message": "Silence / Too Quiet",
                "confidence": 0.0,
                "is_valid": False
            }

        # Extract Mel Spectrogram
        tensor = self.extractor.audio_to_tensor(audio_window) # (1, Mels, Time)
        tensor = tensor.unsqueeze(0).to(self.device) # Add batch dimension: (1, 1, Mels, Time)

        # Predict
        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        prob_anomaly = float(probs[0])
        prob_valid = float(probs[1])

        # Apply Threshold Logic
        # It must be > 0.92 confident that it is the user's voice to pass.
        is_valid = prob_valid > CONFIDENCE_THRESHOLD

        if is_valid:
            status = "VALID"
            msg = f"✅ Authorized Voice Detected ({prob_valid*100:.1f}%)"
        else:
            status = "ANOMALY"
            if prob_valid > 0.5:
                 msg = f"🚨 Access Denied (Voice similar but failed threshold: {prob_valid*100:.1f}%)"
            else:
                 msg = f"🚨 Access Denied (Anomaly detected: {prob_anomaly*100:.1f}%)"

        return {
            "status": status,
            "message": msg,
            "confidence": prob_valid if is_valid else prob_anomaly,
            "prob_valid": prob_valid,
            "prob_anomaly": prob_anomaly,
            "is_valid": is_valid
        }

if __name__ == "__main__":
    # Simple test
    verifier = SpeakerVerifier()
    dummy_audio = np.random.randn(22050 * 3) * 0.01
    result = verifier.verify(dummy_audio)
    print("Test Result:", result)
