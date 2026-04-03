"""
Navora — AI Navigation Assistant (Hugging Face Spaces, Free CPU)
================================================================
Uses BLIP-large + YOLOv8n + MiDaS-small for real-time obstacle
detection and voice-guided navigation. Optimised for CPU inference
on the free HF Spaces tier (~3-4s per frame).
"""

import gradio as gr
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import time

# ═══════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════
TARGET_WIDTH = 480  # smaller than local (640) to speed up CPU inference
CENTER_BAND_START = 0.42
CENTER_BAND_END = 0.58
STOP_CONFIDENCE_THRESHOLD = 0.55
STOP_AREA_THRESHOLD = 0.06
DANGER_LABELS = {"person", "car", "truck", "bus", "motorcycle", "bicycle", "dog", "cat"}

ACTION_DISPLAY = {
    "forward": "MOVE FORWARD",
    "stop": "STOP!",
    "left": "MOVE LEFT",
    "right": "MOVE RIGHT",
}

# ═══════════════════════════════════════════════════════════
#  Model Loading (CPU — fits in 16GB free tier)
#
#  BLIP-large  ~1.8 GB  (vs BLIP-2 flan-t5-xl ~12 GB)
#  YOLOv8n     ~6 MB
#  MiDaS small ~100 MB
#  Total       ~2 GB RAM
# ═══════════════════════════════════════════════════════════
DEVICE = torch.device("cpu")

print("⏳ Loading BLIP-large captioning model...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large",
)
blip_model.eval()
print("✅ BLIP-large loaded (~1.8 GB)")

print("⏳ Loading YOLOv8n...")
yolo_model = YOLO("yolov8n.pt")
print("✅ YOLOv8n loaded")

print("⏳ Loading MiDaS (small)...")

# Patch: MiDaS loads sub-dependencies via torch.hub which trigger an interactive
# trust prompt — crashes in non-interactive environments (HF Spaces, Docker, etc.)

torch.hub._check_repo_is_trusted = lambda *a, **kw: None
midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
midas_model.eval()
print("✅ MiDaS loaded")

print("🚀 All models ready! (CPU mode, ~2 GB total)")


# ═══════════════════════════════════════════════════════════
#  Pipeline Functions
# ═══════════════════════════════════════════════════════════
def run_caption(frame):
    """Generate scene description using BLIP-large."""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = blip_processor(
        images=image,
        text="a photo showing",
        return_tensors="pt",
    )
    with torch.no_grad():
        ids = blip_model.generate(**inputs, max_new_tokens=20, num_beams=1, do_sample=False)
    return blip_processor.decode(ids[0], skip_special_tokens=True).strip()


def run_detection(frame):
    """Detect objects using YOLOv8n."""
    results = yolo_model(frame, conf=0.25, verbose=False)
    dets = {"boxes": [], "class_names": [], "confidences": []}
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            dets["boxes"].append((x1, y1, x2, y2))
            dets["class_names"].append(r.names[int(box.cls[0].item())])
            dets["confidences"].append(box.conf[0].item())
    return dets


def run_depth(frame):
    """Estimate normalised depth using MiDaS-small (256×256)."""
    h, w = frame.shape[:2]
    img = cv2.resize(frame, (256, 256)).astype(np.float32) / 255.0
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        d = midas_model(t)
    dn = d[0, 0].cpu().numpy()
    dmin, dmax = dn.min(), dn.max()
    norm = (dn - dmin) / (dmax - dmin) if dmax > dmin else np.zeros_like(dn)
    return cv2.resize(norm, (w, h))


# ═══════════════════════════════════════════════════════════
#  Guidance Logic
# ═══════════════════════════════════════════════════════════
def direction_from_box(box, fw):
    cx = (box[0] + box[2]) / 2.0
    if cx < fw * CENTER_BAND_START:
        return "left"
    if cx > fw * CENTER_BAND_END:
        return "right"
    return "center"


def get_guidance(detections, fw, fh):
    """Compute navigation action and guidance text from detection list."""
    if not detections:
        return "forward", "Path looks clear. Move forward."

    fa = float(fw * fh)

    def risk(d):
        x1, y1, x2, y2 = d["box"]
        ar = max(0, (x2 - x1) * (y2 - y1)) / max(1.0, fa)
        dir_w = 1.3 if direction_from_box(d["box"], fw) == "center" else 1.0
        lab_w = 1.5 if d["label"].lower() in DANGER_LABELS else 1.0
        return d["confidence"] * max(ar, 1e-4) * dir_w * lab_w

    pri = max(detections, key=risk)
    direction = direction_from_box(pri["box"], fw)
    x1, y1, x2, y2 = pri["box"]
    ar = max(0, (x2 - x1) * (y2 - y1)) / max(1.0, fa)
    label = pri["label"].lower()
    conf = pri["confidence"]

    if label in DANGER_LABELS and direction == "center":
        if conf >= STOP_CONFIDENCE_THRESHOLD and ar >= STOP_AREA_THRESHOLD:
            return "stop", f"Stop. {label} ahead in the center."
        if conf >= 0.65 and ar >= 0.03:
            return "stop", f"Stop. Obstacle ahead: {label}."

    if direction == "left":
        return "right", f"Obstacle on your left: {label}. Move slightly right."
    if direction == "right":
        return "left", f"Obstacle on your right: {label}. Move slightly left."

    return "forward", f"{label} detected ahead. Continue carefully."


# ═══════════════════════════════════════════════════════════
#  Main Processing
# ═══════════════════════════════════════════════════════════
def process_frame(image):
    """Process a single camera frame → guidance outputs."""
    if image is None:
        return "⏳ Waiting for camera...", "⏳ WAITING", "No frame received", ""

    t0 = time.time()

    # Convert to BGR
    frame = np.array(image)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Resize for speed
    h, w = frame.shape[:2]
    if w > TARGET_WIDTH:
        frame = cv2.resize(frame, (TARGET_WIDTH, int(h * (TARGET_WIDTH / w))))
        h, w = frame.shape[:2]

    # Run pipeline
    t1 = time.time()
    desc = run_caption(frame)
    caption_t = time.time() - t1

    t2 = time.time()
    dets = run_detection(frame)
    detect_t = time.time() - t2

    t3 = time.time()
    depth_map = run_depth(frame)
    depth_t = time.time() - t3

    # Merge depth into detections
    details = []
    for box, label, conf in zip(dets["boxes"], dets["class_names"], dets["confidences"]):
        x1, y1, x2, y2 = box
        region = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        md = float(np.median(region)) if region.size > 0 else None
        details.append({
            "label": label,
            "confidence": round(float(conf), 3),
            "box": [int(v) for v in box],
            "depth_score": round(md, 3) if md is not None else None,
        })

    # Compute guidance
    action, guidance_text = get_guidance(details, w, h)
    total = round(time.time() - t0, 2)

    # Format outputs
    action_display = ACTION_DISPLAY.get(action, "⬆️ MOVE FORWARD")

    if details:
        obs_lines = []
        for d in details[:5]:
            depth_str = f" │ depth: {d['depth_score']:.2f}" if d["depth_score"] is not None else ""
            obs_lines.append(f"• {d['label']} ({d['confidence']*100:.0f}%{depth_str})")
        obstacles_str = "\n".join(obs_lines)
    else:
        obstacles_str = "No obstacles detected"

    info = (
        f"🎯 Scene: {desc}\n"
        f"⏱️ BLIP: {caption_t:.2f}s │ YOLO: {detect_t:.2f}s │ MiDaS: {depth_t:.2f}s\n"
        f"📦 Total: {total}s │ Objects: {len(details)}"
    )

    return guidance_text, action_display, obstacles_str, info


# ═══════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════
css = """
.gradio-container {
    max-width: 1100px !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.navora-header {
    text-align: center;
    padding: 20px 16px 10px;
    background: linear-gradient(135deg, rgba(59,130,246,.08), rgba(99,102,241,.08));
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(59,130,246,.15);
}
.navora-header h1 {
    font-size: 32px; font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.navora-header p { color: #94a3b8; font-size: 14px; margin-top: 4px; }
#guidance-box textarea {
    font-size: 18px !important; font-weight: 600 !important;
    min-height: 60px !important; text-align: center !important;
}
#action-box textarea {
    font-size: 22px !important; font-weight: 800 !important;
    text-align: center !important; letter-spacing: 1px !important;
}
#obstacles-box textarea, #info-box textarea {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 13px !important;
}
#webcam-feed { border-radius: 12px; overflow: hidden; }
@media (max-width: 768px) {
    .navora-header h1 { font-size: 24px; }
    #guidance-box textarea { font-size: 16px !important; }
    #action-box textarea { font-size: 18px !important; }
}
.tts-section {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 16px;
    background: rgba(59,130,246,.06);
    border-radius: 10px;
    border: 1px solid rgba(59,130,246,.12);
    margin-top: 8px;
}
.tts-section label { font-size: 13px; font-weight: 600; }
"""

# ═══════════════════════════════════════════════════════════
#  TTS JavaScript (browser-side voice output)
# ═══════════════════════════════════════════════════════════
tts_js = """
<script>
(function() {
    let lastSpoken = '';
    let ttsEnabled = true;
    const synth = window.speechSynthesis;
    window.toggleNavTTS = function(checked) {
        ttsEnabled = checked;
        if (!checked) synth.cancel();
    };
    setInterval(() => {
        if (!ttsEnabled) return;
        const container = document.getElementById('guidance-box');
        if (!container) return;
        const textarea = container.querySelector('textarea');
        if (!textarea) return;
        const text = textarea.value;
        if (text && text !== lastSpoken && !text.startsWith('⏳')) {
            lastSpoken = text;
            synth.cancel();
            const utt = new SpeechSynthesisUtterance(text);
            utt.rate = 1.05;
            const voices = synth.getVoices();
            const pref = voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('google'))
                      || voices.find(v => v.lang.startsWith('en'));
            if (pref) utt.voice = pref;
            synth.speak(utt);
        }
    }, 800);
})();
</script>
"""

# ═══════════════════════════════════════════════════════════
#  Gradio UI
# ═══════════════════════════════════════════════════════════
with gr.Blocks(
    title="Navora — AI Navigation Assistant",
) as demo:

    gr.HTML("""
    <div class="navora-header">
        <h1>🧭 Navora</h1>
        <p>AI-Powered Navigation Assistant for the Visually Impaired</p>
        <p style="font-size:12px; color:#64748b; margin-top:2px">
            Point your camera forward • Real-time obstacle detection • Voice guidance
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            webcam = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="📷 Camera Feed",
                elem_id="webcam-feed",
            )
            gr.HTML("""
            <div class="tts-section">
                <label>🔊 Voice Guidance (TTS)</label>
                <input type="checkbox" checked onchange="toggleNavTTS(this.checked)"
                       style="width:18px; height:18px; accent-color:#3b82f6; cursor:pointer">
                <span style="font-size:12px; color:#94a3b8">Speaks guidance aloud</span>
            </div>
            """)

        with gr.Column(scale=2):
            action_out = gr.Textbox(
                label="🎯 Action", lines=1, interactive=False,
                elem_id="action-box", value="⏳ WAITING",
            )
            guidance_out = gr.Textbox(
                label="🗣️ Navigation Guidance", lines=2, interactive=False,
                elem_id="guidance-box", value="⏳ Waiting for camera stream...",
            )
            obstacles_out = gr.Textbox(
                label="⚠️ Detected Obstacles", lines=5, interactive=False,
                elem_id="obstacles-box", value="No obstacles detected yet",
            )
            info_out = gr.Textbox(
                label="📊 Pipeline Info", lines=3, interactive=False,
                elem_id="info-box",
            )

    gr.HTML(tts_js)

    gr.HTML("""
    <div style="text-align:center; padding:16px; margin-top:12px;
                background:rgba(59,130,246,.04); border-radius:12px;
                border:1px solid rgba(59,130,246,.1)">
        <p style="font-size:13px; color:#94a3b8; margin:0">
            <strong>How to use:</strong> Allow camera access → Point camera forward →
            Receive voice guidance with ⬆️ Forward, 🛑 Stop, ⬅️ Left, ➡️ Right.
            <br>
            <span style="font-size:11px">
                BLIP-large (scene) + YOLOv8n (objects) + MiDaS-small (depth) │ CPU inference ~3-4s/frame
            </span>
        </p>
    </div>
    """)

    # Stream every 5s — gives CPU enough time to process each frame
    webcam.stream(
        fn=process_frame,
        inputs=[webcam],
        outputs=[guidance_out, action_out, obstacles_out, info_out],
        stream_every=5,
        time_limit=600,  # 10 minute sessions
    )

theme = gr.themes.Soft(primary_hue="blue", secondary_hue="indigo", neutral_hue="slate")
demo.queue().launch(ssr=False, theme=theme, css=css)
