from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form, HTTPException
from typing import List
from pathlib import Path
from datetime import datetime
import os
import shutil
import video_upload_helper
import yolo_model_trainer
from status_manager import training_status
from fastapi.responses import FileResponse

# FastAPI app instance
app = FastAPI()

# Configuration
video_folder = '/var/www/html/dataset-creator/videos/'

# Main output directory
MAIN_OUTPUT_DIR = Path("./output")
MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bbox_test_dir = os.path.join(MAIN_OUTPUT_DIR, "bbox_test_images")
os.makedirs(bbox_test_dir, exist_ok=True)

# Path to the trained models directory
TRAINED_MODELS_DIR = Path("./trained_models")

@app.post("/upload/")
async def upload_videos(
    background_tasks: BackgroundTasks, 
    files: List[UploadFile] = File(...), 
    class_name: str = Form(...),
):
    """
    Handles video upload and starts a YOLO training session.

    Args:
        background_tasks: Background task handler to run training asynchronously.
        files: List of video files to upload.
        class_name: The name of the class for YOLO training.

    Returns:
        A response containing the session ID and a success message.
    """
    # Generate a unique session ID using the current timestamp
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    upload_dir = Path(video_folder) / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Save uploaded video files
    video_files = video_upload_helper.save_uploaded_files(files, upload_dir)

    # Start model training in the background
    background_tasks.add_task(yolo_model_trainer.run_model_training, upload_dir, class_name, training_status)

    return {
        "message": "Videos uploaded successfully. Training will start shortly.",
        "session_id": session_id
    }

@app.get("/status/")
async def get_training_status():
    return training_status

@app.delete("/delete-model/")
async def delete_model(folder_name: str):
    """
    Deletes the specified model folder from the trained_models directory.
    Args:
        folder_name: Name of the folder to delete.

    Returns:
        A success message or raises an HTTPException if deletion fails.
    """
    folder_path = TRAINED_MODELS_DIR / folder_name

    if not folder_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found.")
    
    if not folder_path.is_dir():
        raise HTTPException(status_code=400, detail="The specified path is not a folder.")

    try:
        shutil.rmtree(folder_path)
        return {"message": f"Folder '{folder_name}' has been deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the folder: {e}")

@app.get("/download-model/")
async def download_model(folder_name: str):
    """
    Sends the requested YOLO model (best.pt) from the server to the client.
    Args:
        folder_name: Name of the folder inside `trained_models`.

    Returns:
        FileResponse with the `best.pt` model file or raises an HTTPException if the file doesn't exist.
    """
    # Path to the best.pt file
    model_path = TRAINED_MODELS_DIR / folder_name / "best.pt"

    if not model_path.exists():
        raise HTTPException(status_code=404, detail="Model file not found.")
    
    return FileResponse(
        path=model_path,
        media_type="application/octet-stream",
        filename="best.pt"
    )

@app.post("/stop-training/")
async def stop_training(session_id: str):
    """
    Stops an ongoing training process by its session ID.
    Args:
        session_id: The session ID of the training process to stop.

    Returns:
        A success message or raises an HTTPException if stopping fails.
    """
    try:
        message = yolo_model_trainer.stop_training(session_id)
        return {"message": message}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/model-folders/")
async def get_model_folders():
    """
    Retrieves the names of folders in the trained_models directory.

    Returns:
        A list of folder names or an empty list if no folders exist.
    """
    try:
        # Get the list of folders in the trained_models directory
        folder_names = [
            folder.name for folder in TRAINED_MODELS_DIR.iterdir() if folder.is_dir()
        ]
        return {"folders": folder_names}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
