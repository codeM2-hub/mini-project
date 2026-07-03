"""
FastAPI Backend — Main application entry point.

Serves the REST API and WebSocket endpoint for real-time audio processing.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.routes.audio import router as audio_router
from backend.routes.websocket_handler import router as ws_router
from backend.routes.chatbot import router as chatbot_router
from ml.config import HOST, PORT, FRONTEND_URL

# Configure detailed logging (Requirement 5)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend")

app = FastAPI(
    title="Audio Action Recognition API",
    description="Real-time audio-based action sequence recognition system",
    version="1.0.0",
)

# CORS for frontend (Requirement 2)
# Specifically allowing 5173 as requested
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        FRONTEND_URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request logging (Requirement 5)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# Routes (Requirement 3: Removing /api prefix to match user requirements)
app.include_router(audio_router, tags=["Audio"])
app.include_router(ws_router, prefix="/ws", tags=["WebSocket"])
app.include_router(chatbot_router, tags=["Chatbot"])

# Requirement 3: Add explicit /detect route alias for /upload if needed
@app.post("/detect")
async def detect_alias(request: Request):
    # This just proxies to the upload logic or can be a custom endpoint
    # For now, let's just make it a health check or similar if not specified
    return {"status": "detect_endpoint_active"}

@app.get("/")
async def root():
    return {
        "name": "Audio Action Recognition API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # Requirement 1: Run on 8000
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
