import cv2
from backgroundremover.bg import remove
import numpy as np
import os
from pathlib import Path

# Main output directory
MAIN_OUTPUT_DIR = Path("./output")
MAIN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bbox_test_dir = os.path.join(MAIN_OUTPUT_DIR, "bbox_test_images")
os.makedirs(bbox_test_dir, exist_ok=True)

# Function to process multiple images, remove backgrounds, find bounding boxes, and create YOLO labels
def process_images_and_create_yolo_labels(image_paths, output_dir, class_id=0):
    for image_path in image_paths:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found or cannot be loaded: {image_path}")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.jpg', gray_image)
        image_no_bg = remove(buffer.tobytes(), model_name="u2net", 
                             alpha_matting=True,
                             alpha_matting_foreground_threshold=240,
                             alpha_matting_background_threshold=10,
                             alpha_matting_erode_structure_size=10,
                             alpha_matting_base_size=1000)
        image_no_bg = np.frombuffer(image_no_bg, np.uint8)
        image_no_bg = cv2.imdecode(image_no_bg, cv2.IMREAD_UNCHANGED)

        if image_no_bg is None:
            print(f"Could not decode the image after background removal: {image_path}")
            continue

        if image_no_bg.shape[2] == 4:
            b, g, r, a = cv2.split(image_no_bg)
            white_background = np.ones(image_no_bg.shape, dtype=np.uint8) * 255
            for c in range(3):
                white_background[:, :, c] = (a / 255.0 * image_no_bg[:, :, c] +
                                             (1 - a / 255.0) * white_background[:, :, c])
            image_no_bg = white_background.astype(np.uint8)

        gray_no_bg = cv2.cvtColor(image_no_bg, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray_no_bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 100, 200)
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No object found in the image: {image_path}")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # Draw the bounding box on the image for testing purposes
        labeled_image = image.copy()
        cv2.rectangle(labeled_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the labeled image to the test directory
        labeled_image_path = os.path.join(bbox_test_dir, os.path.basename(image_path))
        cv2.imwrite(labeled_image_path, labeled_image)

        # YOLO label calculations
        original_height, original_width, _ = image.shape
        x_center = (x + w / 2) / original_width
        y_center = (y + h / 2) / original_height
        norm_width = w / original_width
        norm_height = h / original_height
        yolo_format = f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}"

        label_file = os.path.join(output_dir, os.path.basename(image_path).replace(".jpg", ".txt"))
        with open(label_file, 'w') as f:
            f.write(yolo_format + "\n")
        print(f"YOLO label saved to: {label_file}")
