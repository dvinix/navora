---
title: Navora - AI Navigation Assistant
emoji: 🧭
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.10.0
app_file: app.py
pinned: true
license: mit
hardware: zero-a10g
---

# 🧭 Navora — AI-Powered Navigation Assistant

Real-time assistive navigation system for visually impaired users. Uses your phone's camera to detect obstacles and provide voice-guided directions.

**Models:** BLIP-2 (scene understanding) + YOLOv8 (object detection) + MiDaS (depth estimation)

## How to Use
1. Allow camera access when prompted
2. Point your phone camera forward
3. The system will provide real-time voice guidance:
   - ⬆️ **Forward** — Path is clear
   - 🛑 **Stop** — Obstacle directly ahead
   - ⬅️ **Left** — Move left to avoid obstacle
   - ➡️ **Right** — Move right to avoid obstacle
