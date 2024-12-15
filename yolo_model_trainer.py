from pathlib import Path
from datetime import datetime
import os
import random
import subprocess
from subprocess import Popen
from threading import Lock
import video_frame_extractor
import auto_labeler
import shutil
from status_manager import training_status


split_ratio = {"train": 0.7, "valid": 0.2, "test": 0.1}

# Main output directory
MAIN_OUTPUT_DIR = Path("./output")
MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bbox_test_dir = os.path.join(MAIN_OUTPUT_DIR, "bbox_test_images")
os.makedirs(bbox_test_dir, exist_ok=True)

# Dictionary to store active training processes
training_processes = {}
process_lock = Lock()  # Lock for thread-safe access to the dictionary

# Function to create data.yml file
def create_data_yml(output_dir: Path, class_name: str):
    data_yml_path = output_dir / "data.yml"
    with open(data_yml_path, "w") as f:
        content = f"""train: train/images
val: ../valid/images
test: ../test/images

nc: 1
names: ['{class_name}']

beltz-ai:
  project: car-part-xt4qx
  version: 0.1
  url: https://beltz.ai
"""
        f.write(content)
    print(f"data.yml created at: {data_yml_path}")

# Function to run the model training
def run_model_training(upload_dir: Path, class_name: str, training_status: dict):
    try:
        
        session_id = upload_dir.name
        training_status.update({
            "status": "Training",
            "session_id": upload_dir.name,
            "message": "Model training in progress..."
        })

        # Create main output directory with timestamp
        timestamp = datetime.now().strftime("model_%d_%m_%y_%H_%M_%S")
        main_output_dir = MAIN_OUTPUT_DIR / timestamp
        output_dirs = {
            "train": main_output_dir / "train/images",
            "valid": main_output_dir / "valid/images",
            "test": main_output_dir / "test/images"
        }
        label_dirs = {k: v.with_name(v.name.replace("images", "labels")) for k, v in output_dirs.items()}

        # Create necessary directories
        for path in list(output_dirs.values()) + list(label_dirs.values()):
            os.makedirs(path, exist_ok=True)

        # Extract frames from uploaded videos
        video_files = [f for f in os.listdir(upload_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
        all_frames = video_frame_extractor.extract_frames(video_files, upload_dir)
        random.shuffle(all_frames)

        # Calculate split counts
        train_count = int(len(all_frames) * split_ratio["train"])
        valid_count = int(len(all_frames) * split_ratio["valid"])

        # Split frames into train, valid, and test sets
        frame_splits = {
            "train": all_frames[:train_count],
            "valid": all_frames[train_count:train_count + valid_count],
            "test": all_frames[train_count + valid_count:]
        }

        # Save images
        for split, frames in frame_splits.items():
            for frame, filename in frames:
                video_frame_extractor.process_and_save_image(frame, filename, output_dirs[split])

        # Process all images and create YOLO labels
        for split in frame_splits:
            image_paths = [str(output_dirs[split] / f[1]) for f in frame_splits[split]]
            auto_labeler.process_images_and_create_yolo_labels(image_paths, label_dirs[split], class_id=0)

        # Create data.yml file
        create_data_yml(main_output_dir, class_name)

        print(f"All videos processed. Dataset, labels, and labeled images saved in folder: {main_output_dir}")

        # Train the YOLO model and store the process
        training_process = train_yolo(main_output_dir, class_name, timestamp)

        # Store the process in the dictionary
        with process_lock:
            training_processes[session_id] = training_process

        training_process.wait()  # Wait for the process to finish

        # Remove the process from the dictionary once it finishes
        with process_lock:
            training_processes.pop(session_id, None)

        # Update status on success
        training_status.update({
            "status": "Completed",
            "message": f"Training completed. Output saved to {main_output_dir}"
        })
    except Exception as e:
        # Handle errors during processing
        training_status.update({
            "status": "Error",
            "message": str(e)
        })


def train_yolo(main_output_dir: Path, class_name: str, timestamp: str):
    data_yml_path = main_output_dir / "data.yml"
    model_output_dir = main_output_dir
    additional_save_dir = Path(f"./trained_models/{timestamp}")  # Timestamped folder
    additional_save_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

    # Ensure data_yml_path uses an absolute path
    command = [
        "yolo",
        "task=detect",
        "mode=train",
        "model=yolov8s.pt",
        f"data={data_yml_path.resolve()}",  # Use absolute path to data.yml
        "epochs=50",
        "batch=4",
        "imgsz=640",
        "device=0",
        "workers=8",
        "optimizer=Adam",
        "pretrained=true",
        "val=true",
        "plots=true",
        "save=True",
        "show=true",
        "optimize=true",
        "lr0=0.001",
        "lrf=0.01",
        "fliplr=0.0",
        "amp=False",
        f"project={model_output_dir.resolve()}",  # Custom save path
        "name=" + class_name
    ]
    return Popen(command)

async def get_training_status():
    return training_status

def stop_training(session_id: str) -> str:
    """
    Stops an ongoing training process by its session ID.

    Args:
        session_id (str): The session ID of the training process to stop.

    Returns:
        str: A message indicating the process has been stopped.

    Raises:
        ValueError: If the session ID is not found.
        RuntimeError: If an error occurs during termination.
    """
    with process_lock:
        process = training_processes.get(session_id)

    if not process:
        raise ValueError(f"No training process found for session ID '{session_id}'.")

    try:
        process.terminate()  # Terminate the process
        with process_lock:
            training_processes.pop(session_id, None)  # Remove from the dictionary
        return f"Training process with session ID '{session_id}' has been stopped successfully."
    except Exception as e:
        raise RuntimeError(f"Failed to stop training process: {e}")
