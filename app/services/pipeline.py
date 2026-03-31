import cv2
import torch
import numpy as np
from PIL import  Image
import time
from typing import Dict, Tuple 

def description(image_path: str, model, processor, device) -> Tuple[str, float]:
    start = time.time()
    image = Image.open(image_path).convert('RGB')
    prompt = "a photo of"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        id = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=5,
            reptition_penalty=3.0,
            length_penalty=2.0
        )

    description = processor.batch_decode(id, tokens=True)[0].strip()
    latency = time.time() - start
    return description, latency



def detect_objects(image_path: str, model) -> Tuple[Dict, np.ndarray, float]:
   

    start = time.time()
    image = cv2.imread(image_path)
    results = model(image, conf=0.5, verbose=False)
    detections = {'boxes': [], 'class_names': [], 'confidences': []}

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].item()
            cls_name = result.names[int(box.cls[0].item())]
            detections['boxes'].append((x1, y1, x2, y2))
            detections['class_names'].append(cls_name)
            detections['confidences'].append(conf)
            
    latency = time.time() - start
    return detections, image, latency



def estimate_depth(image_path: str, model, device) -> Tuple[np.ndarray, float]:
    start = time.time()
    image = cv2.imread(image_path)
    h, w = image[:2]

      
    img_resized = cv2.resize(image, (384, 384))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        depth = model(img_tensor)
    
    depth_np = depth[0, 0].cpu().numpy()
    depth_scaled = 10.0 / (depth_np + 1e-5)
    depth_scaled = np.clip(depth_scaled, 0, 10)
    depth_original = cv2.resize(depth_scaled, (w, h))
    
    latency = time.time() - start
    return depth_original, latency




def run_pipeline(image_path: str, models: Dict) -> str:

    
    blip_model = models['blip2_model']
    blip_processor = models['blip2_processor']
    yolo_model = models['yolo_model']
    midas_model = models['midas_model']
    device = models['device']

    
    desc, _ = description(image_path, blip_model, blip_processor, device)
    dets, img, _ = detect_objects(image_path, yolo_model)
    depth, _ = estimate_depth(image_path, midas_model, device)

    
    final_narration = desc
    if dets['class_names']:
        detected_objects = ", ".join(dets['class_names'][:2])
        final_narration += f". Objects detected: {detected_objects}."

    return final_narration