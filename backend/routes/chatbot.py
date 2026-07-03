from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from backend.services.history_service import history_service
import re

router = APIRouter()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def contains_word_or_phrase(msg: str, queries: List[str]) -> bool:
    for q in queries:
        pattern = r'\b' + re.escape(q) + r'\b'
        if re.search(pattern, msg):
            return True
    return False

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    msg = request.message.lower().strip()
    
    # Comprehensive rule-based keywords matching
    greetings = ["hi", "hello", "hey", "greetings", "yo", "hola"]
    help_queries = ["help", "what can you do", "what do you do", "how to use", "info", "capabilities"]
    status_queries = ["status", "health", "running", "active", "state", "condition", "how is the system"]
    last_anomaly_queries = ["last", "latest", "recent", "show last", "most recent"]
    time_queries = ["when", "time", "date", "hour", "moment"]
    danger_queries = ["danger", "suspicious", "threat", "safe", "secure", "warn", "warning", "incident"]
    event_queries = ["what happened", "what is happening", "anything happen", "did something happen", "detect", "anomaly", "anomalies", "alert", "alerts", "event", "events"]
    thanks_queries = ["thanks", "thank you", "awesome", "great", "ok", "okay"]

    # 1. Greetings
    if contains_word_or_phrase(msg, greetings):
        return {"reply": "Hello! I am your Audio Anomaly Assistant. How can I help you check the system status or anomaly logs today?"}
        
    # 2. Help
    elif contains_word_or_phrase(msg, help_queries):
        return {"reply": "I can help you monitor and query the system status. You can ask me: 'What happened?', 'When was the last anomaly?', 'Is there any danger?', or 'Show recent status'."}

    # 3. Last / Recent Anomaly
    elif contains_word_or_phrase(msg, last_anomaly_queries) or ("last" in msg and "anomaly" in msg):
        last = history_service.get_last_anomaly()
        if last:
            return {"reply": f"The last anomaly detail is - Label: {last.get('label')}, Time: {last.get('time_str')}, Message: {last.get('message')} with {last.get('confidence', 0)*100:.1f}% confidence."}
        else:
            return {"reply": "No anomalies have been recorded in the history log."}

    # 4. Time query
    elif contains_word_or_phrase(msg, time_queries) and ("anomaly" in msg or "incident" in msg or "event" in msg):
        last = history_service.get_last_anomaly()
        if last:
            return {"reply": f"The last anomaly occurred at {last.get('time_str', 'unknown time')}."}
        else:
            return {"reply": "No anomalies have been detected, so there is no timestamp recorded."}

    # 5. Danger / Threat / Safety query
    elif contains_word_or_phrase(msg, danger_queries):
        history = history_service.get_history(limit=5)
        if history:
            return {"reply": f"I found {len(history)} recent suspicious events. The most recent one was '{history[-1].get('label')}' at {history[-1].get('time_str')}."}
        else:
            return {"reply": "No danger or suspicious events detected. The environment is currently safe and secure."}

    # 6. Action / Event / What happened / Status query
    elif contains_word_or_phrase(msg, event_queries) or contains_word_or_phrase(msg, status_queries):
        last = history_service.get_last_anomaly()
        if last:
            return {"reply": f"An anomaly was detected at {last.get('time_str', 'unknown time')}. The system reported: '{last.get('message', 'No message')}' with {last.get('confidence', 0)*100:.1f}% confidence."}
        else:
            return {"reply": "Everything is normal. No anomalies or suspicious sequences have been detected in the current session."}

    # 7. Gratitude / Thanks
    elif contains_word_or_phrase(msg, thanks_queries):
        return {"reply": "You're welcome! Let me know if you need anything else."}

    # 8. Fallback
    else:
        return {"reply": "I'm not sure I understand. I can help you with anomaly logs and system status. Try asking 'What happened?' or 'Is there any danger?'"}

@router.get("/history")
async def get_anomaly_history():
    return {"history": history_service.get_history(limit=50)}
