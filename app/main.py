# main.py

from contextlib import asynccontextmanager
from typing import Any, Dict
from fastapi import FastAPI
from fastapi import WebSocket, WebSocketDisconnect
import uvicorn
import cv2
import numpy as np
import base64
import re
import time
from uuid import uuid4
from collections import deque

# Import our refactored modules
from app.models.loader import load_models
from app.services.pipeline import run_pipeline_frame_data


@asynccontextmanager
async def lifespan(app: FastAPI):
    global models
    models = load_models()
    print("Startup complete. Models are loaded and ready.")
    yield


app = FastAPI(title="Navora API", version="0.1.0", lifespan=lifespan)


models: Dict[str, Any] = {}

# Real-time oriented processing knobs
MAX_PROCESSED_FRAMES = 3
FRAME_STRIDE = 3
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


def _normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered


def _decode_frame_base64(frame_base64: str):
    payload = frame_base64
    if "," in frame_base64 and frame_base64.startswith("data:image"):
        payload = frame_base64.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid frame payload. Could not decode image.")
    return frame


def _prune_expired_sessions() -> None:
    now = time.time()
    expired = [
        session_id
        for session_id, state in live_sessions.items()
        if now - state["last_update"] > SESSION_TTL_SECONDS
    ]
    for session_id in expired:
        live_sessions.pop(session_id, None)


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


def _direction_from_box(box, frame_width: int) -> str:
    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2.0
    left_band = frame_width * CENTER_BAND_START
    right_band = frame_width * CENTER_BAND_END
    if center_x < left_band:
        return "left"
    if center_x > right_band:
        return "right"
    return "center"


def _choose_priority_obstacle(detections, frame_width: int, frame_height: int):
    if not detections:
        return None

    # BUG FIX: frame_area must use width × height, not width².
    # Passing frame_width for frame_height caused area ratios to be wrong on all non-square frames.
    # frame_height is not directly available here, so callers must pass it.
    # Updated signature: _choose_priority_obstacle(detections, frame_width, frame_height)
    frame_area = float(frame_width * frame_height)

    def risk_score(det):
        x1, y1, x2, y2 = det["box"]
        area = max(0, (x2 - x1) * (y2 - y1))
        area_ratio = area / max(1.0, frame_area)
        direction = _direction_from_box(det["box"], frame_width)
        direction_weight = 1.3 if direction == "center" else 1.0
        label_weight = 1.5 if det["label"].lower() in DANGER_LABELS else 1.0
        return det["confidence"] * max(area_ratio, 1e-4) * direction_weight * label_weight

    priority = max(
        detections,
        key=risk_score,
    )
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

    if label in DANGER_LABELS and direction == "center" and confidence >= STOP_CONFIDENCE_THRESHOLD and area_ratio >= STOP_AREA_THRESHOLD:
        return "stop", f"Stop. {label} ahead in the center."

    if label in DANGER_LABELS and direction == "center" and confidence >= 0.65 and area_ratio >= 0.03:
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



@app.get("/")
def root():
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
            # If the client sends no session_id, generate one that persists for this
            # connection iteration and is echoed back so the client can reuse it.
            # Without this, every frame would get a new session, breaking speak_now dedup.
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
                    target_h = int(h * (TARGET_WIDTH / w))
                    frame = cv2.resize(frame, (TARGET_WIDTH, target_h))
                    # Recompute dimensions after resize
                    h, w = frame.shape[:2]

                pipeline_result = run_pipeline_frame_data(frame, models)
                priority_obstacle = _choose_priority_obstacle(
                    pipeline_result.get("detections", []), w, h
                )
                action, guidance_text = _guidance_from_priority(priority_obstacle)
                guidance_key = _normalize_text(f"{action} {guidance_text}")
                speak_now = _should_speak(session_state, guidance_key)

                session_state["last_update"] = time.time()
                session_state["history"].append(guidance_key)

                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "action": action,
                        "speak_now": speak_now,
                        "guidance_text": guidance_text,
                        "priority_obstacle": priority_obstacle,
                        "main_feature": pipeline_result.get("main_feature"),
                        "latency": pipeline_result.get("latency"),
                    }
                )
            except Exception as frame_exception:
                await websocket.send_json(
                    {
                        "session_id": session_id,
                        "error": str(frame_exception),
                        "speak_now": False,
                    }
                )
    except WebSocketDisconnect:
        return







if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)