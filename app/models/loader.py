"""
Model loader — loads BLIP-2, YOLOv8, and MiDaS at startup.
"""

import logging
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_models() -> dict:
    """Load all ML models and return them in a dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    blip_dtype = torch.float16 if device.type == "cuda" else torch.float32

    log.info("Loading BLIP-2 model (Salesforce/blip2-flan-t5-xl)...")
    blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        torch_dtype=blip_dtype,
        device_map=device.type,
    )
    blip_model.eval()
    log.info("BLIP-2 loaded.")

    log.info("Loading YOLOv8n...")
    yolo_model = YOLO("yolov8n.pt")
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
