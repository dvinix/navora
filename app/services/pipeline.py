"""
ML inference pipeline — BLIP-2 (caption) + YOLOv8 (detection) + MiDaS (depth).
"""

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def description(frame: np.ndarray, model, processor, device) -> Tuple[str, float]:
    """Generate an obstacle-centric scene description using BLIP-2."""
    start = time.time()
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompt = "Question: What obstacles or objects are directly in front? Answer:"

    if device.type == "cuda":
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=15,
            num_beams=1,       # greedy decoding — ~2× faster than beam=5
            do_sample=False,
        )

    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return text, time.time() - start


def detect_objects(frame: np.ndarray, model, conf: float = 0.25) -> Tuple[Dict, float]:
    """Run YOLOv8 object detection on a BGR frame."""
    start = time.time()
    results = model(frame, conf=conf, verbose=False)
    detections: Dict[str, list] = {"boxes": [], "class_names": [], "confidences": []}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            detections["boxes"].append((x1, y1, x2, y2))
            detections["class_names"].append(result.names[int(box.cls[0].item())])
            detections["confidences"].append(box.conf[0].item())

    return detections, time.time() - start


def estimate_depth(frame: np.ndarray, model, device) -> Tuple[np.ndarray, float]:
    """Estimate normalised inverse-depth using MiDaS (small, 256×256 input)."""
    start = time.time()
    h, w = frame.shape[:2]

    img_resized = cv2.resize(frame, (256, 256)).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(img_tensor)

    depth_np = depth[0, 0].cpu().numpy()
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max > depth_min:
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_np)

    return cv2.resize(depth_normalized, (w, h)), time.time() - start


def extract_main_feature(detections: Dict) -> Optional[Dict]:
    """Find the dominant detected class by cumulative confidence."""
    class_names = detections.get("class_names", [])
    confidences = detections.get("confidences", [])
    if not class_names:
        return None

    score_by_class: Dict[str, float] = {}
    count_by_class: Dict[str, int] = {}
    for cls_name, conf in zip(class_names, confidences):
        score_by_class[cls_name] = score_by_class.get(cls_name, 0.0) + float(conf)
        count_by_class[cls_name] = count_by_class.get(cls_name, 0) + 1

    best_class = max(score_by_class, key=lambda c: score_by_class.get(c, 0.0))
    return {
        "label": best_class,
        "confidence_sum": round(score_by_class[best_class], 3),
        "count": count_by_class[best_class],
    }


def _unique_ordered(items: List[str]) -> List[str]:
    """Deduplicate a list while preserving insertion order."""
    seen: set = set()
    return [x for x in items if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]


def run_pipeline_frame_data(frame: np.ndarray, models: Dict) -> Dict:
    """
    Run the full inference pipeline on a single BGR frame.

    Returns a dict with: narration, main_feature, detected_objects,
    detections (with depth), and per-stage latency.
    """
    blip_model = models.get("blip2_model")
    blip_processor = models.get("blip2_processor")
    yolo_model = models.get("yolo_model")
    midas_model = models.get("midas_model")
    device = models.get("device")

    if any(v is None for v in [blip_model, blip_processor, yolo_model, midas_model, device]):
        missing = [
            k for k, v in [
                ("blip2_model", blip_model), ("blip2_processor", blip_processor),
                ("yolo_model", yolo_model), ("midas_model", midas_model), ("device", device),
            ] if v is None
        ]
        raise RuntimeError(f"Models not fully loaded. Missing: {missing}")

    # --- Run each model stage ---
    desc, caption_latency = description(frame, blip_model, blip_processor, device)
    dets, detection_latency = detect_objects(frame, yolo_model)
    depth_map, depth_latency = estimate_depth(frame, midas_model, device)

    main_feature = extract_main_feature(dets)
    unique_objects = _unique_ordered(dets["class_names"])

    # --- Merge depth into detections ---
    h, w = frame.shape[:2]
    detection_details = []
    for box, label, conf in zip(dets["boxes"], dets["class_names"], dets["confidences"]):
        x1, y1, x2, y2 = box
        region = depth_map[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        median_depth = float(np.median(region)) if region.size > 0 else None
        detection_details.append({
            "label": label,
            "confidence": round(float(conf), 3),
            "box": [int(v) for v in box],
            "depth_score": round(median_depth, 3) if median_depth is not None else None,
        })

    narration = desc
    if unique_objects:
        narration += f". Objects detected: {', '.join(unique_objects[:3])}."

    return {
        "narration": narration,
        "main_feature": main_feature,
        "detected_objects": unique_objects,
        "detections": detection_details,
        "latency": {
            "caption_seconds": round(caption_latency, 3),
            "detection_seconds": round(detection_latency, 3),
            "depth_seconds": round(depth_latency, 3),
        },
    }