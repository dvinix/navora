import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def description(frame: np.ndarray, model, processor, device) -> Tuple[str, float]:
    start = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_frame)
    # Flan-T5 prompt — obstacle-centric question gives more useful captions than "a photo of"
    prompt = "Question: What obstacles or objects are directly in front? Answer:"
    if str(device) == "cuda":
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=15,
            # Greedy decoding (num_beams=1) is ~2x faster than beam=5 — essential for real-time
            num_beams=1,
            do_sample=False,
        )

    text_description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    latency = time.time() - start
    return text_description, latency


def detect_objects(frame: np.ndarray, model, conf: float = 0.25) -> Tuple[Dict, np.ndarray, float]:
    start = time.time()
    image = frame
    results = model(image, conf=conf, verbose=False)
    detections = {"boxes": [], "class_names": [], "confidences": []}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            box_conf = box.conf[0].item()
            cls_name = result.names[int(box.cls[0].item())]
            detections["boxes"].append((x1, y1, x2, y2))
            detections["class_names"].append(cls_name)
            detections["confidences"].append(box_conf)

    latency = time.time() - start
    return detections, image, latency


def estimate_depth(frame: np.ndarray, model, device) -> Tuple[np.ndarray, float]:
    start = time.time()
    image = frame
    h, w = image.shape[:2]

    # MiDaS_small expects 256x256 input (DPT_Large uses 384x384)
    img_resized = cv2.resize(image, (256, 256))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(img_tensor)

    depth_np = depth[0, 0].cpu().numpy()
    # MiDaS outputs inverse-relative depth. Normalise to [0, 1] range for comparison.
    # Do NOT assign metric units — MiDaS output is unitless and scene-scale-dependent.
    depth_min, depth_max = depth_np.min(), depth_np.max()
    if depth_max > depth_min:
        depth_normalized = (depth_np - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_np)
    depth_original = cv2.resize(depth_normalized, (w, h))

    latency = time.time() - start
    return depth_original, latency


def extract_main_feature(detections: Dict) -> Optional[Dict]:
    class_names = detections.get("class_names", [])
    confidences = detections.get("confidences", [])
    if not class_names:
        return None

    score_by_class = {}
    count_by_class = {}
    for cls_name, conf in zip(class_names, confidences):
        score_by_class[cls_name] = score_by_class.get(cls_name, 0.0) + float(conf)
        count_by_class[cls_name] = count_by_class.get(cls_name, 0) + 1

    best_class = max(score_by_class, key=lambda cls: score_by_class.get(cls, 0.0))
    return {
        "label": best_class,
        "confidence_sum": round(float(score_by_class[best_class]), 3),
        "count": count_by_class[best_class],
    }


def unique_ordered(items: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def run_pipeline_frame_data(frame: np.ndarray, models: Dict) -> Dict:
    # Safe access — partial model-load failures produce clear errors instead of KeyError
    blip_model = models.get("blip2_model")
    blip_processor = models.get("blip2_processor")
    yolo_model = models.get("yolo_model")
    midas_model = models.get("midas_model")
    device = models.get("device")

    if any(v is None for v in [blip_model, blip_processor, yolo_model, midas_model, device]):
        missing = [k for k, v in [
            ("blip2_model", blip_model), ("blip2_processor", blip_processor),
            ("yolo_model", yolo_model), ("midas_model", midas_model), ("device", device)
        ] if v is None]
        raise RuntimeError(f"Models not fully loaded. Missing: {missing}")

    desc, caption_latency = description(frame, blip_model, blip_processor, device)
    dets, _, detection_latency = detect_objects(frame, yolo_model)
    depth_map, depth_latency = estimate_depth(frame, midas_model, device)

    main_feature = extract_main_feature(dets)
    unique_objects = unique_ordered(dets["class_names"])

    # Integrate depth: compute median normalised depth within each bounding box.
    # Higher value = relatively farther (inverse depth); lower = closer to camera.
    h, w = frame.shape[:2]
    detection_details = []
    for box, label, conf in zip(dets["boxes"], dets["class_names"], dets["confidences"]):
        x1, y1, x2, y2 = box
        # Clamp to frame bounds
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        region = depth_map[y1c:y2c, x1c:x2c]
        median_depth = float(np.median(region)) if region.size > 0 else None
        detection_details.append({
            "label": label,
            "confidence": round(float(conf), 3),
            "box": [int(v) for v in box],
            # normalised depth [0=closest, 1=farthest]; None if region was empty
            "depth_score": round(median_depth, 3) if median_depth is not None else None,
        })

    final_narration = desc
    if unique_objects:
        detected_objects = ", ".join(unique_objects[:3])
        final_narration += f". Objects detected: {detected_objects}."

    return {
        "narration": final_narration,
        "main_feature": main_feature,
        "detected_objects": unique_objects,
        "detections": detection_details,
        "latency": {
            "caption_seconds": round(caption_latency, 3),
            "detection_seconds": round(detection_latency, 3),
            "depth_seconds": round(depth_latency, 3),
        },
    }


def run_pipeline_frame(frame: np.ndarray, models: Dict) -> str:
    data = run_pipeline_frame_data(frame, models)
    return data["narration"]


def run_pipeline(image_path: str, models: Dict) -> str:
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Unable to read image at path: {image_path}")
    return run_pipeline_frame(frame, models)