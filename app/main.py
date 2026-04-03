"""
Navora — Real-time assistive-vision API.

WebSocket server that receives camera frames, runs the ML pipeline
(BLIP-2 + YOLOv8 + MiDaS), and returns navigation guidance text
for client-side TTS.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict
from pathlib import Path
from collections import deque
from uuid import uuid4
import base64
import re
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
import uvicorn
import cv2
import numpy as np

from app.models.loader import load_models
from app.services.pipeline import run_pipeline_frame_data


# ═══════════════════════════════════════════════════════════
#  App setup
# ═══════════════════════════════════════════════════════════
models: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = load_models()
    print("Startup complete. Models are loaded and ready.")
    yield


app = FastAPI(title="Navora API", version="0.1.0", lifespan=lifespan)


# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════
TARGET_WIDTH = 640
SESSION_TTL_SECONDS = 120
SPEAK_COOLDOWN_SECONDS = 1.5
MAX_HISTORY_ITEMS = 5
CENTER_BAND_START = 0.42
CENTER_BAND_END = 0.58
STOP_CONFIDENCE_THRESHOLD = 0.55
STOP_AREA_THRESHOLD = 0.06
DANGER_LABELS = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"}

live_sessions: Dict[str, Any] = {}


# ═══════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════
def _normalize_text(text: str) -> str:
    lowered = re.sub(r"\s+", " ", text.lower().strip())
    return re.sub(r"[^a-z0-9\s]", "", lowered)


def _decode_frame_base64(frame_base64: str):
    payload = frame_base64
    if "," in frame_base64 and frame_base64.startswith("data:image"):
        payload = frame_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid frame payload. Could not decode image.")
    return frame


def _prune_expired_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in live_sessions.items() if now - s["last_update"] > SESSION_TTL_SECONDS]
    for sid in expired:
        live_sessions.pop(sid, None)


def _ensure_session(session_id: str):
    _prune_expired_sessions()
    if session_id not in live_sessions:
        live_sessions[session_id] = {
            "last_update": time.time(),
            "last_guidance_key": None,
            "last_spoken_at": 0.0,
            "history": deque(maxlen=MAX_HISTORY_ITEMS),
        }
    return live_sessions[session_id]


# ═══════════════════════════════════════════════════════════
#  Guidance logic
# ═══════════════════════════════════════════════════════════
def _direction_from_box(box, frame_width: int) -> str:
    center_x = (box[0] + box[2]) / 2.0
    if center_x < frame_width * CENTER_BAND_START:
        return "left"
    if center_x > frame_width * CENTER_BAND_END:
        return "right"
    return "center"


def _choose_priority_obstacle(detections, frame_width: int, frame_height: int):
    if not detections:
        return None

    frame_area = float(frame_width * frame_height)

    def risk_score(det):
        x1, y1, x2, y2 = det["box"]
        area_ratio = max(0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_area)
        direction = _direction_from_box(det["box"], frame_width)
        dir_weight = 1.3 if direction == "center" else 1.0
        label_weight = 1.5 if det["label"].lower() in DANGER_LABELS else 1.0
        return det["confidence"] * max(area_ratio, 1e-4) * dir_weight * label_weight

    priority = max(detections, key=risk_score)
    direction = _direction_from_box(priority["box"], frame_width)
    x1, y1, x2, y2 = priority["box"]
    area_ratio = max(0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_area)
    return {
        "label": priority["label"],
        "confidence": priority["confidence"],
        "direction": direction,
        "area_ratio": round(area_ratio, 4),
    }


def _guidance_from_priority(priority_obstacle):
    if priority_obstacle is None:
        return "forward", "Path looks clear. Move forward."

    label = priority_obstacle["label"].lower()
    direction = priority_obstacle["direction"]
    confidence = priority_obstacle["confidence"]
    area_ratio = priority_obstacle.get("area_ratio", 0.0)

    if label in DANGER_LABELS and direction == "center":
        if confidence >= STOP_CONFIDENCE_THRESHOLD and area_ratio >= STOP_AREA_THRESHOLD:
            return "stop", f"Stop. {label} ahead in the center."
        if confidence >= 0.65 and area_ratio >= 0.03:
            return "stop", f"Stop. Obstacle ahead: {label}."

    if direction == "left":
        return "right", f"Obstacle on your left: {label}. Move slightly right."
    if direction == "right":
        return "left", f"Obstacle on your right: {label}. Move slightly left."

    return "forward", f"{label} detected ahead. Continue carefully."


def _should_speak(session_state, guidance_key: str) -> bool:
    now = time.time()
    if session_state["last_guidance_key"] != guidance_key:
        session_state["last_guidance_key"] = guidance_key
        session_state["last_spoken_at"] = now
        return True
    if now - session_state["last_spoken_at"] >= SPEAK_COOLDOWN_SECONDS:
        session_state["last_spoken_at"] = now
        return True
    return False


# ═══════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════
@app.get("/")
def root():
    """Serve test client HTML or return API status JSON."""
    test_client = Path(__file__).resolve().parent.parent / "test_client.html"
    if test_client.exists():
        return FileResponse(str(test_client), media_type="text/html")
    return {"status": "ok", "service": "Navora API"}


@app.post("/session/start")
def start_live_session():
    session_id = uuid4().hex
    _ensure_session(session_id)
    return {"session_id": session_id, "ttl_seconds": SESSION_TTL_SECONDS}


@app.websocket("/ws/live-guidance")
async def websocket_live_guidance(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            session_id = payload.get("session_id") or uuid4().hex
            frame_base64 = payload.get("frame_base64")

            if not frame_base64:
                await websocket.send_json({"error": "frame_base64 is required.", "session_id": session_id})
                continue

            session_state = _ensure_session(session_id)

            try:
                frame = _decode_frame_base64(frame_base64)
                h, w = frame.shape[:2]
                if w > TARGET_WIDTH:
                    frame = cv2.resize(frame, (TARGET_WIDTH, int(h * (TARGET_WIDTH / w))))
                    h, w = frame.shape[:2]

                result = run_pipeline_frame_data(frame, models)
                priority = _choose_priority_obstacle(result.get("detections", []), w, h)
                action, guidance_text = _guidance_from_priority(priority)
                guidance_key = _normalize_text(f"{action} {guidance_text}")
                speak_now = _should_speak(session_state, guidance_key)

                session_state["last_update"] = time.time()
                session_state["history"].append(guidance_key)

                await websocket.send_json({
                    "session_id": session_id,
                    "action": action,
                    "speak_now": speak_now,
                    "guidance_text": guidance_text,
                    "priority_obstacle": priority,
                    "main_feature": result.get("main_feature"),
                    "latency": result.get("latency"),
                })
            except Exception as e:
                await websocket.send_json({
                    "session_id": session_id,
                    "error": str(e),
                    "speak_now": False,
                })
    except WebSocketDisconnect:
        return


# ═══════════════════════════════════════════════════════════
#  Entrypoint
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import socket

    project_root = Path(__file__).resolve().parent.parent
    cert_file = project_root / "cert.pem"
    key_file = project_root / "key.pem"

    ssl_kwargs = {}
    if cert_file.exists() and key_file.exists():
        ssl_kwargs["ssl_keyfile"] = str(key_file)
        ssl_kwargs["ssl_certfile"] = str(cert_file)
        protocol = "https"
    else:
        protocol = "http"

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "localhost"

    print(f"\n{'='*60}")
    print("  Navora API Server")
    print(f"{'='*60}\n")
    print(f"  Local:   {protocol}://localhost:8000")
    print(f"  Network: {protocol}://{local_ip}:8000")
    if ssl_kwargs:
        print(f"\n  📱 Open on your phone: {protocol}://{local_ip}:8000")
        print("  Accept the self-signed certificate warning.")
    else:
        print("\n  ⚠ No SSL certs found — phone camera won't work over HTTP.")
        print("  Run: pip install cryptography && python serve_https.py")
    print(f"\n{'='*60}\n")

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, **ssl_kwargs)