import cv2
from ultralytics import YOLO
import time
import sys
import platform
import numpy as np

def check_camera_permissions():
    if platform.system() == 'Darwin':  # macOS
        print("\nCamera Access Required:")
        print("1. Open System Settings/System Preferences")
        print("2. Go to Privacy & Security/Security & Privacy")
        print("3. Click on Camera in the left sidebar")
        print("4. Enable camera access for your Terminal/IDE")
        print("\nAfter granting permission, please run the script again.\n")

def add_overlay(frame, people_count):
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Create overlay settings
    overlay_height = 80
    padding = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    # Create a semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    
    # Add the overlay with transparency
    alpha = 0.7
    frame[:overlay_height] = cv2.addWeighted(overlay[:overlay_height], alpha, frame[:overlay_height], 1 - alpha, 0)
    
    # Add text
    text = f"People Detected: {people_count}"
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x = padding
    text_y = (overlay_height + text_size[1]) // 2
    
    # Add text with a subtle shadow effect
    cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), font_thickness)  # shadow
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)    # main text
    
    return frame

def count_people_live():
    # Load the YOLO model
    model = YOLO('yolov8n.pt')
    
    # Open the webcam (0 is usually the default webcam)
    video = cv2.VideoCapture(0)
    
    # Check if the webcam is opened successfully
    if not video.isOpened():
        print("Error: Could not open webcam")
        check_camera_permissions()
        return
    
    print("Starting live person detection... Press 'q' to quit")
    
    try:
        while True:
            # Read a frame from the webcam
            success, frame = video.read()
            if not success:
                print("Error: Could not read frame")
                break
            
            # Run YOLO detection
            results = model(frame)
            
            # Count people (class 0 in COCO dataset is 'person')
            people_count = sum(1 for box in results[0].boxes if box.cls == 0)
            
            # Draw detection boxes
            annotated_frame = results[0].plot()
            
            # Add the overlay with counter
            annotated_frame = add_overlay(annotated_frame, people_count)
            
            # Display the frame
            cv2.imshow("Live Person Detection", annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping detection gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Release the video capture object and destroy windows
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    count_people_live() 