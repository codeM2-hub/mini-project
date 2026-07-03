import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
HISTORY_FILE = PROJECT_ROOT / "data" / "anomaly_history.json"

class HistoryService:
    def __init__(self):
        self.history_file = HISTORY_FILE
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            with open(self.history_file, "w") as f:
                json.dump([], f)

    def add_entry(self, entry: Dict):
        """
        Add a new anomaly entry to history.
        Expected entry format:
        {
            "timestamp": "ISO format",
            "label": "anomaly",
            "confidence": 0.95,
            "message": "🚨 Unauthorized Voice Detected",
            "details": {...}
        }
        """
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            history = []

        # Add formatted timestamp if not present
        if "time_str" not in entry:
            entry["time_str"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        history.append(entry)
        
        # Keep only last 100 entries
        if len(history) > 100:
            history = history[-100:]

        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)
            
        return entry

    def get_history(self, limit: int = 10) -> List[Dict]:
        try:
            with open(self.history_file, "r") as f:
                history = json.load(f)
            return history[-limit:]
        except Exception:
            return []

    def get_last_anomaly(self) -> Optional[Dict]:
        history = self.get_history(limit=1)
        return history[0] if history else None

history_service = HistoryService()
