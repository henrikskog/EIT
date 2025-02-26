from ultralytics import YOLO
from bytetrack.byte_tracker import BYTETracker
import cv2
import numpy as np

class PersonTrackingSystem:
    def __init__(self):
        # Initialize YOLO model for person detection
        self.model = YOLO('yolov8n.pt')
        
        # Initialize ByteTracker
        self.tracker = BYTETracker(
            track_thresh=0.25,    # Detection threshold
            track_buffer=30,      # How many frames to keep dead tracklets
            match_thresh=0.8,     # IOU threshold for matching
            frame_rate=30         # Frame rate of video
        )
        
        # Dictionary to store tracking history
        self.track_history = {}
        
        # Counter for people
        self.people_count = {
            'in': 0,
            'out': 0,
            'current': 0
        }

    def process_frame(self, frame):
        # Run YOLO detection
        results = self.model(frame)[0]
        
        # Convert YOLO detections to ByteTrack format [x1, y1, x2, y2, score, class_id]
        detections = []
        for det in results.boxes.data:
            x1, y1, x2, y2, conf, cls = det
            if int(cls) == 0:  # class 0 is person in COCO
                detections.append([x1, y1, x2, y2, conf, 0])
        
        # Convert to numpy array for ByteTracker
        if len(detections) > 0:
            detections = np.array(detections)
        else:
            detections = np.empty((0, 6))

        # Update tracker
        frame_size = [frame.shape[0], frame.shape[1]]
        online_targets = self.tracker.update(
            detections,
            frame_size,
            frame_size
        )

        # Process and visualize tracking results
        tracked_objects = []
        for t in online_targets:
            # Get track ID and bounding box
            track_id = t.track_id
            bbox = t.tlbr  # top-left bottom-right format
            
            # Calculate centroid
            centroid = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(centroid)
            
            # Keep only recent history
            max_history = 30  # frames
            self.track_history[track_id] = self.track_history[track_id][-max_history:]
            
            # Create tracking info dictionary
            track_info = {
                'track_id': track_id,
                'bbox': bbox,
                'centroid': centroid,
                'history': self.track_history[track_id]
            }
            tracked_objects.append(track_info)

            # Draw visualization
            self._draw_tracking(frame, track_info)

        return frame, tracked_objects

    def _draw_tracking(self, frame, track_info):
        """Draw bounding boxes, IDs, and trails on the frame"""
        bbox = track_info['bbox']
        track_id = track_info['track_id']
        history = track_info['history']

        # Draw bounding box
        cv2.rectangle(frame, 
                     (int(bbox[0]), int(bbox[1])), 
                     (int(bbox[2]), int(bbox[3])), 
                     (0, 255, 0), 2)
        
        # Draw ID
        cv2.putText(frame, f"ID: {track_id}", 
                    (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                    (0, 255, 0), 2)

        # Draw movement trail
        for i in range(1, len(history)):
            pt1 = (int(history[i-1][0]), int(history[i-1][1]))
            pt2 = (int(history[i][0]), int(history[i][1]))
            cv2.line(frame, pt1, pt2, (0, 0, 255), 2)

def main():
    # Initialize the tracking system
    tracking_system = PersonTrackingSystem()
    
    # Initialize video capture (replace with your camera source)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        processed_frame, tracked_objects = tracking_system.process_frame(frame)
        
        # Display the processed frame
        cv2.imshow('Person Tracking', processed_frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
