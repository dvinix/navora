import logging
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from ultralytics import YOLO 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

def load_models():

    logging.info("Detecting compute device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    models = {}

    try:
        # Load BLIP-2 Model
        logging.info("Loading BLIP-2 model ...")

        blip_dtype = torch.float16 if device.type == "cuda" else torch.float32
        blip_device_map = device.type  # "cuda" or "cpu"
        models['blip2_processor'] = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        models['blip2_model'] = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-flan-t5-xl",
            torch_dtype=blip_dtype,
            device_map=blip_device_map,
        )
        models['blip2_model'].eval()
        logging.info("BLIP-2 model loaded successfully.")

        # Load YOLO Model
        logging.info("Loading YOLOv8 model (yolov8n.pt)...")
        models['yolo_model'] = YOLO('yolov8n.pt')
        models['yolo_model'].to(device)
        logging.info("YOLOv8 model loaded successfully.")

        # Load MiDaS Model
        logging.info("Loading MiDaS model (intel-isl/MiDaS)...")
        models['midas_model'] = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        models['midas_model'].to(device)
        models['midas_model'].eval()
        logging.info("MiDaS model loaded successfully.")

        models['device'] = device
        logging.info("All models have been loaded and configured.")

    except Exception as e:
        logging.error(f"An error occurred during model loading: {e}")
        # Depending on the application, you might want to exit or handle this differently
        raise e

    return models
