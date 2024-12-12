from ultralytics import YOLO
import os
import cv2
import easyocr
import re
import numpy as np

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

# Folder containing images
image_folder = r'C:\Users\Kenneth\Downloads\repos\Plate detection\dataset\test\images'

# Character replacement map to fix common OCR errors
CHARACTER_MAP = {
    "O": "0", "I": "1", "L": "1", "B": "8", "S": "5",
    "G": "6", "Q": "0", "Z": "2"
}

def preprocess_image(image):
    """Preprocess the image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise using GaussianBlur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive thresholding for better contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def refine_text(text):
    """Refine OCR output by replacing common misinterpreted characters."""
    refined = ''.join(CHARACTER_MAP.get(char, char) for char in text)
    return refined.strip()

def run_inference(image_path):
    # Load the YOLO model
    model_path = r'C:\Users\Kenneth\Downloads\automatic-number-plate-recognition-python-yolov8-main\best.pt'
    model = YOLO(model_path)

    # Perform inference
    results = model.predict(image_path, conf=0.25)

    # Read the input image
    img = cv2.imread(image_path)

    # Loop through detections
    for result in results:
        boxes = result.boxes

        for i, box in enumerate(boxes):
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Crop the license plate area
            cropped_img = img[y1:y2, x1:x2]

            # Preprocess the cropped image
            preprocessed_img = preprocess_image(cropped_img)

            # OCR Processing
            ocr_result = reader.readtext(preprocessed_img)
            print(f"OCR Raw Result: {ocr_result}")

            license_plate = ''
            for detection in ocr_result:
                text = detection[1]
                # Validate and refine detected text
                if re.match(r'^[A-Z0-9\s\-]{4,10}$', text):  # Example regex
                    license_plate = refine_text(text)
                    break

            # Display results
            if license_plate:
                print(f"Detected License Plate: {license_plate}")
                cv2.putText(img, license_plate, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("No valid license plate detected.")

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the final image with results
    cv2.imshow("License Plate Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_images_in_folder(image_folder):
    # Get all image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Run inference on each image
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing: {image_path}")
        run_inference(image_path)

# Run inference on all images in the folder
process_images_in_folder(image_folder)