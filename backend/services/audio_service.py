"""
Audio Service — Business logic layer between routes and ML detector.
"""

import json
import traceback
import numpy as np
from pathlib import Path
from typing import Dict

from ml.config import PROJECT_ROOT, ACTION_LABELS
from ml.inference.realtime_detector import RealtimeDetector


class AudioService:
    """Singleton service wrapping the RealtimeDetector."""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if AudioService._initialized:
            return
        AudioService._initialized = True

        print("[AudioService] Initializing...")
        self.detector = RealtimeDetector()

        # Load labels
        self.labels = ACTION_LABELS.copy()
        lp = PROJECT_ROOT / "data" / "labels.json"
        if lp.exists():
            with open(lp) as f:
                ld = json.load(f)
            self.labels = {int(k): v for k, v in ld.get("idx_to_label", {}).items()}
        print("[AudioService] Ready.")

    # ── File upload ──────────────────────────────────────────
    def process_file(self, file_path: str) -> Dict:
        try:
            return self.detector.process_file(file_path)
        except Exception as e:
            traceback.print_exc()
            return {
                "windows": [], "duration": 0, "total_windows": 0,
                "action_count": 0, "normal_count": 0, "anomaly_count": 0,
                "silence_count": 0, "sequence_info": {}, "wrong_order_alerts": [],
                "anomaly_summary": {
                    "overall_status": "error",
                    "overall_message": f"Error: {str(e)}",
                    "total_windows": 0, "normal_count": 0,
                    "anomaly_count": 0, "action_count": 0,
                },
                "status": "error", "error": str(e),
            }

    def process_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """Process a single chunk of audio (called by WebSocket)."""
        return self.detector.predict_window(audio_chunk)

    # ── Microphone ───────────────────────────────────────────
    def start_microphone(self) -> Dict:
        try:
            import sounddevice as sd
            dev = sd.query_devices(kind="input")
            result = self.detector.start_microphone()
            return {**result, "device": dev.get("name", "microphone")}
        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def stop_microphone(self):
        try:
            self.detector.stop_microphone()
        except Exception:
            pass

    # ── Status / data ────────────────────────────────────────
    def get_status(self) -> Dict:
        s = self.detector.get_status()
        return {**s, "mic_active": self.detector.is_running}

    def get_mic_result(self) -> Dict:
        return self.detector.get_mic_result()

    def get_labels(self) -> Dict:
        return {"labels": self.labels, "num_classes": len(self.labels)}

    def get_timeline(self) -> Dict:
        seq = self.detector.sequence_tracker.get_session_summary()
        return {
            "events":          seq.get("events", []),
            "alerts":          seq.get("alerts", []),
            "session_sequence": seq.get("session_sequence", []),
            "expected_sequence": seq.get("expected_sequence", []),
        }

    def get_sequence_log(self) -> list:
        return self.detector.get_log(n=50)
