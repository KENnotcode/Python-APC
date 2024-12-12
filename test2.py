from ultralytics import YOLO
import cv2
import easyocr
import re
import keyboard  # For detecting keypress
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'], gpu=True)

# Character replacement map to fix common OCR errors
CHARACTER_MAP = {
    "O": "0", "I": "1", "L": "1", "B": "8", "S": "5",
    "G": "6", "Q": "0", "Z": "2", "W": "U", "F" : "F"
}

def preprocess_image(image):
    """Preprocess the image for OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def refine_text(text):
    """Refine OCR output by replacing common misinterpreted characters."""
    refined = ''.join(CHARACTER_MAP.get(char, char) for char in text)
    return refined.strip()

def run_inference_on_frame(frame, model):
    """Run inference on a single video frame."""
    results = model.predict(frame, conf=0.25)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]

            # Crop the license plate area
            cropped_img = frame[y1:y2, x1:x2]

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
                cv2.putText(frame, license_plate, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                print("No valid license plate detected.")

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return frame

def main():
    # Load the YOLO model
    model_path = r'C:\Users\Kenneth\Downloads\automatic-number-plate-recognition-python-yolov8-main\best.pt'
    model = YOLO(model_path)

    # Open webcam
    cap = cv2.VideoCapture(0)  # 0 for the default camera; change index for external cameras

    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    print("Press 's' to stop the camera.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting.")
            break

        # Run inference on the frame
        frame_with_results = run_inference_on_frame(frame, model)

        # Display the frame
        cv2.imshow("License Plate Detection", frame_with_results)

        # Check if the 's' key has been pressed
        if keyboard.is_pressed('s'):
            print("Stopping the camera feed...")
            break

        # Exit the OpenCV window on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting on 'q' key...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the program
if __name__ == "__main__":
    main()