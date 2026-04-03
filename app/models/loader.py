"""
Model loader — loads BLIP-2, YOLOv8, and MiDaS at startup.
"""

import logging
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_models() -> dict:
    """Load all ML models and return them in a dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    blip_dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Use smaller BLIP-2 model for CPU efficiency
    # blip2-opt-2.7b is 5x faster than blip2-flan-t5-xl on CPU
    model_name = "Salesforce/blip2-opt-2.7b" if device.type == "cpu" else "Salesforce/blip2-flan-t5-xl"
    log.info(f"Loading BLIP-2 model ({model_name})...")
    blip_processor = Blip2Processor.from_pretrained(model_name)
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=blip_dtype,
        device_map=device.type if device.type == "cuda" else None,
        low_cpu_mem_usage=True,
    )
    blip_model.eval()
    log.info("BLIP-2 loaded.")

    log.info("Loading YOLOv8n...")
    # Try to find yolov8n.pt in parent directory or download
    yolo_path = Path(__file__).parent.parent.parent.parent / "yolov8n.pt"
    if yolo_path.exists():
        yolo_model = YOLO(str(yolo_path))
    else:
        yolo_model = YOLO("yolov8n.pt")  # Will download if not found
    yolo_model.to(device)
    log.info("YOLOv8 loaded.")

    log.info("Loading MiDaS (small)...")
    midas_model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    midas_model.to(device)
    midas_model.eval()
    log.info("MiDaS loaded.")

    log.info("All models ready.")
    return {
        "blip2_processor": blip_processor,
        "blip2_model": blip_model,
        "yolo_model": yolo_model,
        "midas_model": midas_model,
        "device": device,
    }
