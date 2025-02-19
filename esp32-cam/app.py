import cv2
import threading
from threading import Thread
import time
import sys

class VideoStreamTimeout(Thread):
    def __init__(self, url):
        Thread.__init__(self)
        self.url = url
        self.video = None
        self.success = False
        
    def run(self):
        self.video = cv2.VideoCapture(self.url)
        if self.video.isOpened():
            self.success = True

def stream_esp32_cam():
    print("Starting ESP32-CAM stream...")
    # ESP32-CAM stream URL
    esp32_stream_url = "http://192.168.11.87:81/stream"

    TIMEOUT_SECONDS = 2
    
    stream_thread = VideoStreamTimeout(esp32_stream_url)
    stream_thread.start()
    stream_thread.join(timeout=TIMEOUT_SECONDS)
    
    if not stream_thread.is_alive() and stream_thread.success:
        video = stream_thread.video
    else:
        if stream_thread.is_alive():
            print(f"Error: Connection timeout after {TIMEOUT_SECONDS} seconds")
        else:
            print("Error: Could not connect to ESP32-CAM stream")
        print("Please check if the ESP32-CAM is powered on and connected to the network")
        print(f"Make sure the stream URL is correct: {esp32_stream_url}")
        return 
    
    print("Starting ESP32-CAM stream... Press 'q' to quit")
    
    try:
        while True:
            # Read a frame from the stream with timeout
            success, frame = video.read()
            if not success:
                print("Error: Could not read frame from ESP32-CAM stream")
                print("Attempting to reconnect...")
                video.release()
                # Try to reconnect
                stream_thread = VideoStreamTimeout(esp32_stream_url)
                stream_thread.start()
                stream_thread.join(timeout=5)
                if stream_thread.success:
                    video = stream_thread.video
                continue
            
            # Display the frame
            cv2.imshow("ESP32-CAM Stream", frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopping stream gracefully...")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        # Release the video capture object and destroy windows
        if 'video' in locals() and video is not None:
            video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_esp32_cam()
