import cv2
import time
import os
import platform
import sys
from datetime import datetime

def check_camera_permissions():
    """Check and inform about camera permissions on macOS"""
    if platform.system() == 'Darwin':  # macOS
        print("\nCamera Access Required:")
        print("1. Open System Settings/System Preferences")
        print("2. Go to Privacy & Security/Security & Privacy")
        print("3. Click on Camera in the left sidebar")
        print("4. Enable camera access for your Terminal/IDE")
        print("\nAfter granting permission, please run the script again.\n")

def record_video(output_file="test_video.mp4", duration=30, fps=20):
    """
    Record video from ESP32-CAM or local camera
    
    Args:
        output_file: Name of the output video file
        duration: Recording duration in seconds
        fps: Frames per second for the output video
    """
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename, extension = os.path.splitext(output_file)
    output_file = f"{filename}_{timestamp}{extension}"
    
    print("Starting video recording...")
    
    # Determine camera source based on environment variable
    use_macos_camera = os.environ.get('USE_MACOS_CAMERA', 'false').lower() == 'true'
    
    if use_macos_camera:
        print("Using macOS camera...")
        # Check for camera permissions on macOS
        if platform.system() == 'Darwin':
            check_camera_permissions()
        # Use default camera (usually built-in webcam)
        video = cv2.VideoCapture(0)
    else:
        # ESP32-CAM stream URL
        esp32_stream_url = "http://172.20.10.3:81/stream"
        print(f"Using ESP32-CAM stream: {esp32_stream_url}")
        # Open the ESP32-CAM stream
        video = cv2.VideoCapture(esp32_stream_url)
    
    # Check if the camera/stream is opened successfully
    if not video.isOpened():
        if use_macos_camera:
            print("Error: Could not access macOS camera")
            print("Please check camera permissions")
        else:
            print("Error: Could not connect to ESP32-CAM stream")
            print("Please check if the ESP32-CAM is powered on and connected to the network")
            print(f"Make sure the stream URL is correct: {esp32_stream_url}")
        return
    
    # Get video properties
    success, frame = video.read()
    if not success:
        print("Failed to read initial frame. Exiting.")
        video.release()
        return
    
    height, width = frame.shape[:2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec for MP4 files
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file: {output_file}")
        video.release()
        return
    
    source_text = "macOS camera" if use_macos_camera else "ESP32-CAM"
    print(f"Recording video from {source_text} for {duration} seconds...")
    print(f"Output will be saved to: {output_file}")
    print("Press 'q' to stop recording early")
    
    # Variables for recording
    start_time = time.time()
    frames_recorded = 0
    
    try:
        while True:
            # Check if we've reached the duration
            elapsed_time = time.time() - start_time
            if elapsed_time >= duration:
                break
            
            # Read a frame from the stream
            success, frame = video.read()
            if not success:
                print(f"Error: Could not read frame from {source_text}")
                if not use_macos_camera:
                    print("Attempting to reconnect...")
                    video = cv2.VideoCapture(esp32_stream_url)
                continue
            
            # Write the frame to the output file
            out.write(frame)
            frames_recorded += 1
            
            # Display the frame with recording indicator
            # Add recording indicator and time remaining
            remaining = max(0, int(duration - elapsed_time))
            cv2.putText(frame, f"RECORDING | Time left: {remaining}s", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow("Recording Video", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nRecording stopped early by user")
                break
                
    except KeyboardInterrupt:
        print("\nRecording stopped by keyboard interrupt")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Release the video capture and writer objects and destroy windows
        end_time = time.time()
        actual_duration = end_time - start_time
        
        video.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f"\nRecording completed:")
        print(f"- Duration: {actual_duration:.2f} seconds")
        print(f"- Frames recorded: {frames_recorded}")
        print(f"- Actual FPS: {frames_recorded / actual_duration:.2f}")
        print(f"- Output file: {os.path.abspath(output_file)}")
        
        # Check if the file was created successfully
        if os.path.exists(output_file):
            file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"- File size: {file_size_mb:.2f} MB")
        else:
            print("Warning: Output file was not created successfully")

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Record video from ESP32-CAM')
    parser.add_argument('--output', type=str, default='test_video.mp4',
                        help='Output video file name')
    parser.add_argument('--duration', type=int, default=30,
                        help='Recording duration in seconds')
    parser.add_argument('--fps', type=int, default=20,
                        help='Frames per second for output video')
    
    args = parser.parse_args()
    
    # Start recording
    record_video(args.output, args.duration, args.fps)
