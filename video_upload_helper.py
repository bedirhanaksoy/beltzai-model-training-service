from typing import List
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Form
from pathlib import Path
import shutil

# Helper function to save uploaded files
def save_uploaded_files(files: List[UploadFile], upload_dir: Path) -> List[Path]:
    file_paths = []
    for file in files:
        file_path = upload_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        file_paths.append(file_path)
    return file_paths
