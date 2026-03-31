from PIL import Image
import time
import torch

def generate_description(image_path: str, model, processor, device) -> str:
    """
    Generates a scene description for an image using the BLIP-2 model.
    """
    start_time = time.time()
    
    image = Image.open(image_path).convert('RGB')
    
    prompt = "a photo of"
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=20,
            num_beams=5,
            repetition_penalty=3.0,
            length_penalty=2.0,
            temperature=1.0
        )

    description = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    latency = time.time() - start_time
    return description, latency
