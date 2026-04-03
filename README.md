# 🧭 Navora — Real-time Navigation Assistant

AI-powered assistive-vision backend for blind navigation.

Processes camera frames in real-time using:
- **BLIP-2** — Scene understanding & obstacle-centric captioning
- **YOLOv8** — Object detection with confidence scores
- **MiDaS** — Monocular depth estimation

Returns lightweight guidance text for client-side TTS. No audio files stored server-side.

## Architecture

```
Phone Camera → WebSocket frames → Server ML Pipeline → Guidance JSON → Client TTS
```

- Client streams JPEG frames over WebSocket
- Server returns `action` + `guidance_text` + `speak_now` flag
- Client speaks guidance using device TTS engine

## Endpoints

### `GET /`
Serves the test client HTML (or returns `{"status": "ok"}` if HTML not found).

### `POST /session/start`
Create a short-lived session.
```json
{ "session_id": "f6d4c4...", "ttl_seconds": 120 }
```

### `WS /ws/live-guidance`
Main real-time endpoint.

**Client sends:**
```json
{ "session_id": "f6d4c4...", "frame_base64": "data:image/jpeg;base64,..." }
```

**Server responds:**
```json
{
  "session_id": "f6d4c4...",
  "action": "left",
  "speak_now": true,
  "guidance_text": "Obstacle on your right: bicycle. Move slightly left.",
  "priority_obstacle": { "label": "bicycle", "confidence": 0.71, "direction": "right" },
  "main_feature": { "label": "bicycle", "confidence_sum": 0.71, "count": 1 },
  "latency": { "caption_seconds": 0.63, "detection_seconds": 0.08, "depth_seconds": 0.12 }
}
```

- `action` — navigation command: `forward`, `stop`, `left`, `right`
- `speak_now` — client should only speak when `true` (dedup/cooldown)
- `guidance_text` — text to speak via TTS

## Local Setup

```bash
# Create & activate venv
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run server
python -m app.main
```

## Cloud Deployment (Free GPU)

See `hf_deploy/` for Hugging Face Spaces deployment with ZeroGPU.

## Project Structure

```
navora/
├── app/
│   ├── main.py              # FastAPI server + WebSocket + guidance logic
│   ├── models/
│   │   └── loader.py         # Model loading (BLIP-2, YOLOv8, MiDaS)
│   └── services/
│       └── pipeline.py       # ML inference pipeline
├── hf_deploy/                # Hugging Face Spaces deployment files
│   ├── app.py                # Gradio app (ZeroGPU)
│   ├── requirements.txt
│   └── README.md
├── notebooks/
│   └── main.ipynb            # Original prototype notebook
├── test_client.html          # Browser-based test client
├── requirements.txt
└── yolov8n.pt
```
