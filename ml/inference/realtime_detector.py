
import time
import json
import numpy as np
import torch
import threading
import queue
import sounddevice as sd
from pathlib import Path
from typing import Dict, List, Optional

from ml.config import (
    SAMPLE_RATE, WINDOW_SAMPLES, HOP_SAMPLES, CONFIDENCE_THRESHOLD,
    MODEL_DIR, NUM_CLASSES, ACTION_LABELS, PROJECT_ROOT,
    SEQUENCE_MODEL_PATH, SEQ_CONFIDENCE_THRESHOLD, N_MELS
)
from ml.preprocessing.audio_processor import AudioProcessor
from ml.preprocessing.feature_extractor import FeatureExtractor
from ml.models.crnn_model import ActionCRNN
from ml.models.sequence_anomaly_model import SequenceAnomalyModel
from ml.inference.sequence_tracker import SequenceTracker
from backend.services.history_service import history_service
from backend.services.email_service import email_service

# Below this RMS → treat as silence
SILENCE_THRESHOLD = 0.004


class RealtimeDetector:
    """Processes audio windows and returns structured detection results."""

    def __init__(self, model_path=None, confidence_threshold=CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.audio_processor  = AudioProcessor()
        self.feature_extractor = FeatureExtractor()
        self.sequence_tracker  = SequenceTracker()
        self.session_start     = time.time()

        # ── Load label map ──────────────────────────────────
        self.labels: Dict[int, str] = {}
        labels_path = PROJECT_ROOT / "data" / "labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                ld = json.load(f)
            self.labels = {int(k): v for k, v in ld["idx_to_label"].items()}
        else:
            self.labels = ACTION_LABELS.copy()

        # voice_normal label index (for quick check)
        self._voice_normal_idx = next(
            (k for k, v in self.labels.items() if v == "voice_normal"), -1
        )

        # ── Load model ──────────────────────────────────────
        if model_path is None:
            model_path = MODEL_DIR / "best_model.pth"
        self.model_trained = Path(model_path).exists()
        self.device = "cpu"   # Mac CPU is fine for inference
        self.model  = self._load_model(str(model_path))

        # ── Mic streaming state ─────────────────────────────
        self.is_running   = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self._lock        = threading.Lock()
        self._stream      = None
        self.last_result  = None
        self.result_log: List[Dict] = []   # per-window log for UI
        self._log_lock    = threading.Lock()
        self._process_thread = None
        
        # Debounce for email alerts (don't send more than once per 60s for same anomaly)
        self.last_email_time = 0
        self.email_cooldown = 60 

        # ── Load Sequence Model ──────────────────────────────
        self.seq_model = self._load_sequence_model()
        self.seq_buffer = np.array([], dtype=np.float32)
        self.seq_window_size = int(5.0 * SAMPLE_RATE) # 5-second window for sequence model

        print(f"[Detector] ✓ {len(self.labels)} classes: {list(self.labels.values())}")
        print(f"[Detector] ✓ Model trained: {self.model_trained}")

    # ── Model loading ────────────────────────────────────────
    def _load_model(self, model_path: str):
        # We now specifically load the SpeakerCNN model for binary verification
        from ml.models.speaker_cnn import SpeakerCNN
        model_path = str(PROJECT_ROOT / "models" / "speaker_model.pth")
        
        model = SpeakerCNN(num_classes=2)
        if Path(model_path).exists():
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            self.model_trained = True
            print("[Detector] ✓ Loaded SpeakerCNN Binary Model")
        else:
            self.model_trained = False
            print("[Detector] ✗ No speaker_model.pth found — run: python scripts/train_speaker.py")
        
        model.to(self.device).eval()
        self.arch = "speaker_cnn"
        return model

    def _load_sequence_model(self):
        """Loads the sequence-based anomaly detection model."""
        model = SequenceAnomalyModel(n_mels=N_MELS)
        path = str(SEQUENCE_MODEL_PATH)
        if Path(path).exists():
            model.load_state_dict(torch.load(path, map_location=self.device, weights_only=False))
            model.to(self.device).eval()
            print(f"[Detector] ✓ Loaded Sequence Anomaly Model: {path}")
            return model
        print("[Detector] ℹ️ Sequence model not found. Run training first.")
        return None

    # ── Core inference ───────────────────────────────────────
    @torch.no_grad()
    def predict_window(self, audio_window: np.ndarray) -> Dict:
        """
        Classify one audio window for SPEAKER VERIFICATION.
        """
        target_samples = int(SAMPLE_RATE * 3.0)
        
        # We need exactly 3 seconds for the CNN
        if len(audio_window) < target_samples:
            # only pad if strictly necessary, normally the stream buffer handles this
            window = np.pad(audio_window, (0, target_samples - len(audio_window)))
        else:
            window = audio_window[:target_samples]

        timestamp = time.time() - self.session_start
        rms = float(np.sqrt(np.mean(window ** 2)))

        if rms < SILENCE_THRESHOLD:
            result = self._make_result(
                label="SILENCE", conf=0.0, rtype="silence",
                is_anomaly=False, rms=rms, timestamp=timestamp,
                message="🔇 No Voice Detected", all_probs={}, top_label="silence"
            )
            self._log_result(result)
            return result

        # ── Model inference ────────────────────────────────
        feat = self.feature_extractor.audio_to_tensor(window) # (1, Mels, Time)
        feat = feat.unsqueeze(0).to(self.device) # (1, 1, Mels, Time)
        
        logits = self.model(feat)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        prob_anomaly = float(probs[0])
        prob_valid = float(probs[1])

        CONFIDENCE_THRESHOLD = 0.92
        is_valid = prob_valid > CONFIDENCE_THRESHOLD

        # ── Sequence Anomaly Check ────────────────────────
        # Maintain a 5-second buffer for temporal analysis
        self.seq_buffer = np.append(self.seq_buffer, window)
        if len(self.seq_buffer) > self.seq_window_size:
            self.seq_buffer = self.seq_buffer[-self.seq_window_size:]
        
        seq_anomaly_detected = False
        seq_conf = 0.0
        
        if self.seq_model and len(self.seq_buffer) >= self.seq_window_size:
            # Extract features for the 5-second window
            seq_feat = self.feature_extractor.audio_to_tensor(self.seq_buffer)
            seq_feat = seq_feat.unsqueeze(0).to(self.device) # (1, 1, Mels, Time)
            
            with torch.no_grad():
                seq_logits = self.seq_model(seq_feat)
                seq_prob = torch.sigmoid(seq_logits).item()
                seq_conf = seq_prob
                if seq_prob > SEQ_CONFIDENCE_THRESHOLD:
                    seq_anomaly_detected = True
            
            # ── Debug Output (Requirement 4) ───────────
            print(f"\n--- Sequence Detection Debug ---")
            print(f"Window: last 5 seconds")
            print(f"Raw Score: {seq_prob:.4f}")
            print(f"Probabilities: Normal: {1-seq_prob:.4f}, Anomaly: {seq_prob:.4f}")
            print(f"Final Status: {'⚠️ ANOMALY DETECTED' if seq_anomaly_detected else 'System Normal'}")
            print(f"--------------------------------\n")

        if is_valid and not seq_anomaly_detected:
            label, rtype, is_anomaly = "voice_normal", "normal", False
            message = "System Normal"
            top_label = "voice_normal"
            conf = prob_valid
        elif seq_anomaly_detected:
            label, rtype, is_anomaly = "SEQUENCE_ANOMALY", "anomaly", True
            message = "⚠️ ANOMALY DETECTED"
            top_label = "anomaly_sequence"
            conf = seq_conf
        else:
            # Speaker CNN says anomaly
            label, rtype, is_anomaly = "UNAUTHORIZED_VOICE", "anomaly", True
            message = "🚨 Unauthorized Voice / Sound"
            top_label = "unauthorized"
            conf = prob_anomaly

        # Trigger alerts if anomaly
        if is_anomaly:
            self._trigger_anomaly_alerts(message, label, timestamp, conf)

        result = self._make_result(
            label=label, conf=float(conf), rtype=rtype,
            is_anomaly=is_anomaly, rms=rms, timestamp=timestamp,
            message=message, 
            all_probs={"anomaly": float(prob_anomaly), "valid": float(prob_valid), "seq_anomaly": float(seq_conf)},
            top_label=top_label
        )
        
        self._log_result(result)
        return result

    def _trigger_anomaly_alerts(self, message, label, timestamp, confidence):
        """Helper to log history and send emails."""
        # 1. Store in history
        history_service.add_entry({
            "timestamp": timestamp,
            "label": label,
            "confidence": float(confidence),
            "message": message,
            "type": "anomaly"
        })
        
        # 2. Send Email Alert (with debounce)
        current_time = time.time()
        if current_time - self.last_email_time > self.email_cooldown:
            threading.Thread(
                target=email_service.send_anomaly_email,
                args=({
                    "label": label, 
                    "confidence": confidence, 
                    "message": message, 
                    "timestamp": str(round(timestamp, 2))
                },),
                daemon=True
            ).start()
            self.last_email_time = current_time

    def _make_result(self, label, conf, rtype, is_anomaly, rms,
                     timestamp, message, all_probs, top_label) -> Dict:
        return {
            "timestamp":        round(timestamp, 2),
            "label":            label,
            "type":             rtype,
            "confidence":       round(conf, 3),
            "is_anomaly":       is_anomaly,
            "rms":              round(rms, 4),
            "message":          message,
            "all_probs":        all_probs,
            "top_label":        top_label,
            # Legacy keys for frontend compatibility
            "anomaly": {
                "current_label": label,
                "current_type":  rtype,
                "current_conf":  round(conf, 3),
                "is_anomaly":    is_anomaly,
                "top_class":     top_label,
                "top_conf":      round(conf, 3),
                "similarity":    round(conf, 3),
                "rms":           round(rms, 4),
                "message":       message,
                "status":        rtype,
            },
            "all_probabilities": all_probs,
            "predictions": [
                {"action": k, "confidence": v}
                for k, v in all_probs.items() if v >= self.confidence_threshold
            ],
        }

    def _log_result(self, result: Dict):
        with self._log_lock:
            self.result_log.append(result)
            if len(self.result_log) > 200:
                self.result_log = self.result_log[-200:]
            self.last_result = result

    def get_log(self, n: int = 50) -> List[Dict]:
        with self._log_lock:
            return self.result_log[-n:]

    # ── File processing ─────────────────────────────────────
    def process_file(self, file_path: str) -> Dict:
        """Process entire audio file, return per-window results + summary."""
        audio    = self.audio_processor.load_audio(file_path)
        duration = len(audio) / SAMPLE_RATE
        windows  = []

        self.sequence_tracker.reset()
        self.session_start = time.time()

        step = WINDOW_SAMPLES   # 1-second windows, no overlap for file
        t    = 0
        while t + WINDOW_SAMPLES <= len(audio):
            chunk  = audio[t:t + WINDOW_SAMPLES]
            result = self.predict_window(chunk)
            result["timestamp"] = round(t / SAMPLE_RATE, 2)
            result["anomaly"]["timestamp"] = result["timestamp"]
            windows.append(result)
            t += step

        # Also process trailing audio if ≥0.3s
        remaining = len(audio) - t
        if remaining >= int(SAMPLE_RATE * 0.3):
            chunk  = audio[t:]
            result = self.predict_window(chunk)
            result["timestamp"] = round(t / SAMPLE_RATE, 2)
            windows.append(result)

        # Summary
        types      = [w["type"] for w in windows]
        n_action   = sum(1 for tp in types if tp == "action")
        n_normal   = sum(1 for tp in types if tp == "normal")
        n_anomaly  = sum(1 for tp in types if tp in ("anomaly", "wrong_order", "uncertain"))
        n_silence  = sum(1 for tp in types if tp == "silence")
        n_total    = len(windows)

        seq_info   = self.sequence_tracker.get_session_summary()
        alerts     = seq_info.get("alerts", [])

        if n_total == 0:
            overall_status = "empty"
            overall_message = "No audio"
        elif n_action > 0 and alerts:
            overall_status  = "wrong_order"
            overall_message = f"⚠️ WRONG SEQUENCE ORDER — {len(alerts)} alert(s)"
        elif n_action > 0:
            overall_status  = "action_detected"
            overall_message = f"🎯 Action sequence detected ({n_action} windows)"
        elif n_anomaly > n_normal:
            overall_status  = "anomaly"
            overall_message = f"🚨 Anomaly detected ({n_anomaly}/{n_total} windows)"
        elif n_normal > 0:
            overall_status  = "normal"
            overall_message = f"✅ Normal voice ({n_normal}/{n_total} windows)"
        else:
            overall_status  = "silence"
            overall_message = "🔇 Silence"

        return {
            "windows":        windows,
            "duration":       round(duration, 2),
            "total_windows":  n_total,
            "action_count":   n_action,
            "normal_count":   n_normal,
            "anomaly_count":  n_anomaly,
            "silence_count":  n_silence,
            "sequence_info":  seq_info,
            "wrong_order_alerts": alerts,
            "anomaly_summary": {
                "overall_status":  overall_status,
                "overall_message": overall_message,
                "total_windows":   n_total,
                "normal_count":    n_normal,
                "anomaly_count":   n_anomaly,
                "action_count":    n_action,
            },
        }

    # ── Microphone streaming ─────────────────────────────────
    def start_microphone(self):
        """Start real-time microphone capture and processing."""
        if self.is_running:
            return {"status": "already_running"}

        self.is_running    = True
        self.audio_buffer  = np.array([], dtype=np.float32)
        self.session_start = time.time()
        self.sequence_tracker.reset()

        with self._log_lock:
            self.result_log.clear()

        def audio_callback(indata, frames, time_info, status):
            chunk = indata[:, 0].astype(np.float32)
            with self._lock:
                self.audio_buffer = np.concatenate([self.audio_buffer, chunk])

        def process_loop():
            target_samples = int(SAMPLE_RATE * 3.0)
            while self.is_running:
                with self._lock:
                    buf = self.audio_buffer.copy()

                if len(buf) >= target_samples:
                    window = buf[:target_samples]
                    with self._lock:
                        # Advance buffer by 1 second (overlapping 3-second windows)
                        # so that we get predictions every 1 second but using 3s of context
                        advance_samples = int(SAMPLE_RATE * 1.0)
                        self.audio_buffer = self.audio_buffer[advance_samples:]
                    self.predict_window(window)
                else:
                    time.sleep(0.05)

        try:
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=HOP_SAMPLES, callback=audio_callback
            )
            self._stream.start()
            self._process_thread = threading.Thread(target=process_loop, daemon=True)
            self._process_thread.start()
            print("[Detector] ✓ Microphone started")
            return {"status": "started"}
        except Exception as e:
            self.is_running = False
            print(f"[Detector] ✗ Mic failed: {e}")
            return {"status": "error", "error": str(e)}

    def stop_microphone(self):
        """Stop microphone capture."""
        self.is_running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        print("[Detector] ✓ Microphone stopped")
        return {"status": "stopped"}

    def get_mic_result(self) -> Dict:
        """Get the latest microphone detection result."""
        if self.last_result:
            return {**self.last_result, "mic_running": self.is_running}
        return {
            "mic_running": self.is_running,
            "label": "waiting...", "type": "waiting",
            "message": "🎤 Listening...", "confidence": 0.0,
            "anomaly": {"current_label": "waiting", "current_type": "waiting",
                        "current_conf": 0.0, "is_anomaly": False,
                        "top_class": "", "top_conf": 0.0, "message": "🎤 Listening..."},
        }

    def get_status(self) -> Dict:
        return {
            "is_running":    self.is_running,
            "model_trained": self.model_trained,
            "num_classes":   len(self.labels),
            "labels":        list(self.labels.values()),
            "device":        self.device,
        }
