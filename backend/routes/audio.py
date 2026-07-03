import tempfile, os, traceback
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from backend.services.audio_service import AudioService

router = APIRouter()
audio_service = AudioService()


@router.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload and analyze an audio file. Returns per-window results + summary."""
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg"}:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.flush(); tmp.close()
        results = audio_service.process_file(tmp.name)
        return JSONResponse(content=results)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        try: os.unlink(tmp.name)
        except Exception: pass


@router.get("/status")
async def get_status():
    return audio_service.get_status()


@router.get("/labels")
async def get_labels():
    return audio_service.get_labels()


@router.get("/timeline")
async def get_timeline():
    return audio_service.get_timeline()


@router.get("/mic-result")
async def get_mic_result():
    """Get the latest microphone detection result (polled every second by UI)."""
    return audio_service.get_mic_result()


@router.get("/sequence-log")
async def get_sequence_log():
    """Get the last 50 per-second detection windows."""
    return {"log": audio_service.get_sequence_log()}


@router.post("/start-mic")
async def start_microphone():
    try:
        return audio_service.start_microphone()
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop-mic")
async def stop_microphone():
    audio_service.stop_microphone()
    return {"status": "stopped"}
