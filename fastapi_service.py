from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from typing import List
import os
from datetime import datetime
from pathlib import Path
import video_upload_helper
import yolo_model_trainer
from status_manager import training_status

# FastAPI app instance
app = FastAPI()

# Configuration
video_folder = '/var/www/html/dataset-creator/videos/'

# Main output directory
MAIN_OUTPUT_DIR = Path("./output")
MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bbox_test_dir = os.path.join(MAIN_OUTPUT_DIR, "bbox_test_images")
os.makedirs(bbox_test_dir, exist_ok=True)

@app.post("/upload/")
async def upload_videos(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...), 
    class_name: str = Form(...),  # New parameter for class name
):
    upload_dir = Path(video_folder) / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video files
    video_files = video_upload_helper.save_uploaded_files(files, upload_dir)

    # Start model training in the background
    background_tasks.add_task(yolo_model_trainer.run_model_training, upload_dir, class_name, training_status)

    return {"message": "Videos uploaded successfully. Training will start shortly."}

@app.get("/status/")
async def get_training_status():
    return training_status


