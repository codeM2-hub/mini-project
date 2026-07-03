"""
Sequence Tracker — Tracks detected actions over time.
Alerts if action sequence occurs in WRONG ORDER.

Expected normal sequence (example):
  oyo_hotel → sequence_63 → sequence_64 → sequence_65 → sequence_66 → sequence_67

If actions appear out of order → WRONG ORDER ALERT.
"""

import time
from collections import deque
from typing import List, Dict, Optional


# ── Define the EXPECTED order of action sequences ─────────────────────────
# User can change this to match their real-world sequence expectation.
# E.g. doorbell → door-unlock → door-open → door-close
EXPECTED_SEQUENCE = [
    "oyo_hotel",
    "sequence_63",
    "sequence_64",
    "sequence_65",
    "sequence_66",
    "sequence_67",
]

ACTION_NAMES = {
    "oyo_hotel":   "OYO Hotel Sound",
    "sequence_63": "Action Sequence 63",
    "sequence_64": "Action Sequence 64",
    "sequence_65": "Action Sequence 65",
    "sequence_66": "Action Sequence 66",
    "sequence_67": "Action Sequence 67",
    "voice_normal": "Normal Voice",
}


class SequenceTracker:
    """
    Tracks detected action events.
    - Debounces repeated detections of same class
    - Checks if detected order matches EXPECTED_SEQUENCE
    - Raises alert if sequence is out of order
    """

    def __init__(self, debounce_sec: float = 2.0):
        self.debounce_sec  = debounce_sec    # minimum gap between same event
        self.event_log: List[Dict] = []      # all confirmed events
        self.last_seen: Dict[str, float] = {}  # last time each class was detected
        self.sequence_alerts: List[Dict] = [] # order-violation alerts
        self.session_sequence: List[str] = [] # sequence seen so far this session
        self._last_event_time: float = 0.0

    def reset(self):
        self.event_log.clear()
        self.last_seen.clear()
        self.sequence_alerts.clear()
        self.session_sequence.clear()
        self._last_event_time = 0.0

    def register(self, label: str, confidence: float, timestamp: float) -> Optional[Dict]:
        """
        Register a detected action.
        Returns alert dict if sequence is out of order, else None.
        """
        if label == "voice_normal" or label == "SILENCE" or label == "ANOMALY":
            return None

        # Debounce: skip if same label seen recently
        last = self.last_seen.get(label, 0.0)
        if timestamp - last < self.debounce_sec:
            return None

        self.last_seen[label] = timestamp

        # Log the event
        event = {
            "label":      label,
            "name":       ACTION_NAMES.get(label, label),
            "confidence": round(confidence, 3),
            "timestamp":  round(timestamp, 2),
            "order_ok":   True,
            "alert":      None,
        }

        # Check order against expected sequence
        alert = self._check_order(label, event)
        self.event_log.append(event)
        return alert

    def _check_order(self, label: str, event: Dict) -> Optional[Dict]:
        """
        Check if this event matches the expected sequence order.
        Returns alert if it's in wrong position.
        """
        if label not in EXPECTED_SEQUENCE:
            return None

        self.session_sequence.append(label)

        # What's the expected position of this label?
        expected_idx = EXPECTED_SEQUENCE.index(label)

        # What should come at this position in our session?
        our_position = len(self.session_sequence) - 1

        if our_position < len(EXPECTED_SEQUENCE):
            expected_at_pos = EXPECTED_SEQUENCE[our_position]
            if expected_at_pos != label:
                # WRONG ORDER!
                alert = {
                    "type":     "wrong_order",
                    "message":  f"⚠️ WRONG ORDER! Got '{ACTION_NAMES.get(label, label)}' but expected '{ACTION_NAMES.get(expected_at_pos, expected_at_pos)}'",
                    "got":      label,
                    "expected": expected_at_pos,
                    "timestamp": event["timestamp"],
                }
                event["order_ok"] = False
                event["alert"]    = alert["message"]
                self.sequence_alerts.append(alert)
                return alert

        return None

    def get_session_summary(self) -> Dict:
        return {
            "total_events":    len(self.event_log),
            "events":          self.event_log[-20:],  # last 20
            "alerts":          self.sequence_alerts[-5:],
            "session_sequence": self.session_sequence,
            "expected_sequence": EXPECTED_SEQUENCE,
        }
