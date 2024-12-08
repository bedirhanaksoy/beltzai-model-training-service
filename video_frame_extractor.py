import cv2
import os

fps_interval = 1  # 1 frame per second
resize_dims = (640, 640)

# Extract frames from video
def extract_frames(video_files, upload_dir):
    all_frames = []
    global_frame_count = 0

    for video_file in video_files:
        video_path = upload_dir / video_file
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}.")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        interval = fps * fps_interval
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % interval == 0:
                frame_filename = f'frame_{global_frame_count:06d}.jpg'
                all_frames.append((frame, frame_filename))
                global_frame_count += 1

            frame_count += 1
        cap.release()
        print(f"Extracted frames from {video_file}.")

    return all_frames

# Function to process and save images
def process_and_save_image(image, base_filename, output_path):
    resized_image = cv2.resize(image, resize_dims)
    cv2.imwrite(os.path.join(output_path, base_filename), resized_image)
    rotated_clockwise = cv2.rotate(resized_image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(os.path.join(output_path, f'cw_{base_filename}'), rotated_clockwise)
    rotated_counterclockwise = cv2.rotate(resized_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(os.path.join(output_path, f'ccw_{base_filename}'), rotated_counterclockwise)

