"""
WebSocket Handler — Real-time streaming of detection results to the frontend.

The frontend connects via WebSocket and receives live updates including:
  - Detected actions with confidence scores
  - Active action list
  - Completed sequences
  - Audio level (RMS)
"""

import json
import asyncio
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.audio_service import AudioService

router = APIRouter()
audio_service = AudioService()


class ConnectionManager:
    """Manages active WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        for conn in self.active_connections[:]:
            try:
                await conn.send_json(data)
            except Exception:
                self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/audio")
async def websocket_audio(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming.

    Client sends raw audio chunks (base64 or binary).
    Server responds with detection results.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()

            # Convert bytes to numpy array (16-bit PCM)
            audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

            # Calculate audio level
            rms = float(np.sqrt(np.mean(audio_chunk ** 2)))

            # Process through detector
            result = audio_service.process_chunk(audio_chunk)
            result["audio_level"] = rms

            # Send results back
            await websocket.send_json(result)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
        print(f"[WS] Error: {e}")


@router.websocket("/status")
async def websocket_status(websocket: WebSocket):
    """WebSocket endpoint for periodic status updates."""
    await manager.connect(websocket)
    try:
        while True:
            status = audio_service.get_status()
            await websocket.send_json(status)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
