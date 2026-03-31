# main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import cv2

# Import our refactored modules
from app.models.loader import load_models
from app.services.pipeline import run_pipeline

# Create the main application instance
app = FastAPI(title="Navora API", version="0.1.0")

# This dictionary will hold our loaded models
models = {}

@app.on_event("startup")
def startup_event():
    """
    On server startup, load all AI models into the 'models' dictionary.
    """
    global models
    models = load_models()
    print("Startup complete. Models are loaded and ready.")

@app.post("/process-video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    """
    This endpoint receives a video, processes its frames, and returns narrations.
    """
    # 1. Validate the file type with a professional error message
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="The provided file is not a valid video format.")

    # 2. Save the uploaded video to a temporary file
    temp_video_path = "temp_uploaded_video.mp4"
    with open(temp_video_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # 3. Process the video
    cap = cv2.VideoCapture(temp_video_path)
    all_narrations = [] # Use a new list to store results
    frame_count = 0
    
    # Process the first 3 frames
    while cap.isOpened() and frame_count < 3:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        # Use a clean temporary filename
        temp_frame_path = f"temp_frame_{frame_count}.jpg"
        cv2.imwrite(temp_frame_path, frame)
        
        try:
            # Call the pipeline and store the result in a new variable
            single_narration = run_pipeline(temp_frame_path, models)
            # Append the formatted result to our list
            all_narrations.append(f"Frame {frame_count}: {single_narration}")
        except Exception as e:
            all_narrations.append(f"Frame {frame_count}: Error processing frame - {e}")
        finally:
            # Ensure the temporary frame is always deleted
            if os.path.exists(temp_frame_path):
                os.remove(temp_frame_path)

    cap.release()
    os.remove(temp_video_path)

    # 4. Return the collected narrations
    return JSONResponse(content={"filename": file.filename, "narrations": all_narrations})

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)