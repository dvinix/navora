# Navora Realtime Guidance API

Realtime assistive-vision backend for blind navigation.

The API processes camera frames using:
- BLIP-2 for scene understanding
- YOLOv8 for obstacle detection
- MiDaS for depth context

Output is text guidance for client-side TTS. The server does not store audio files.

## Why This Architecture

For production realtime guidance:
- Client app streams frames over WebSocket.
- Server returns lightweight guidance text.
- Mobile app speaks guidance locally using device TTS.

This keeps latency low and avoids server-side audio generation bottlenecks.

## Endpoints

### GET /
Health/status endpoint.

### POST /session/start
Create a short-lived realtime session.

Response example:
```json
{
  "session_id": "f6d4c4...",
  "ttl_seconds": 120
}
```

### WS /ws/live-guidance
Main realtime endpoint.

Client message:
```json
{
  "session_id": "f6d4c4...",
  "frame_base64": "data:image/jpeg;base64,/9j/4AAQ..."
}
```

Server message:
```json
{
  "session_id": "f6d4c4...",
  "action": "left",
  "speak_now": true,
  "guidance_text": "Obstacle on your right: bicycle. Move slightly left.",
  "priority_obstacle": {
    "label": "bicycle",
    "confidence": 0.71,
    "direction": "right"
  },
  "main_feature": {
    "label": "bicycle",
    "confidence_sum": 0.71,
    "count": 1
  },
  "latency": {
    "caption_seconds": 0.63,
    "detection_seconds": 0.08,
    "depth_seconds": 0.12
  }
}
```

Meaning:
- `action`: navigation command (`stop`, `left`, `right`, `forward`)
- `speak_now`: client should speak only when true to avoid repetition
- `guidance_text`: text to speak via mobile TTS

### POST /process-video/
Upload-video test endpoint for offline validation.
Returns frame-wise guidance fields (`action`, `speak_now`, `narration`, detection metadata).
No audio URLs or stored audio files are returned.

## Local Run

1. Create and activate virtual environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run API:
```bash
uvicorn app.main:app --reload
```

## Client-Side TTS Recommendation

In production mobile app:
- speak only when `speak_now` is true
- throttle repeated phrases on device
- keep last guidance phrase and skip duplicates

## Notes

- Session memory is in-process and ephemeral.
- For horizontal scaling, move session state to Redis.
- No persistent user data storage is required for realtime guidance flow.
