import cv2
import numpy as np
import imutils
import time
import csv
from scipy.spatial import distance as dist

# Global counters for people entering and exiting
totalEntered = 0
totalExited = 0

# ------------------ CSV Logging Setup ------------------
csv_filename = "people_count.csv"
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)
# Write header row: timestamp, total entered, total exited, and current count
csv_writer.writerow(["timestamp", "entered", "exited", "current"])

# Timer for logging every second
last_logged_time = time.time()


# ------------------ Centroid Tracker Class ------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        global totalEntered
        # Register a new object and count as "entered"
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        totalEntered += 1
        self.nextObjectID += 1

    def deregister(self, objectID):
        global totalExited
        # Deregister an object and count as "exited"
        del self.objects[objectID]
        del self.disappeared[objectID]
        totalExited += 1

    def update(self, rects):
        # If no detections, mark existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids from bounding boxes: 𝒄 = (𝑐ₓ, 𝑐ᵧ)
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are being tracked, register all centroids
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab object IDs and centroids for existing objects
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the Euclidean distance between each pair of existing and new centroids
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Associate existing objects with new centroids
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Identify unassigned object IDs and new centroids
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # For objects not matched with a new centroid, increment the disappeared counter
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            else:
                # Register new objects for remaining centroids
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


# ------------------ Load Pre-trained Person Detector ------------------
# The model uses MobileNetSSD with an SSD detector, pre-trained in Caffe.
prototxt = "MobileNetSSD_deploy.prototxt.txt"  # Update path if necessary
model = "MobileNetSSD_deploy.caffemodel"  # Update path if necessary
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ------------------ Initialize Video Stream and Tracker ------------------
vs = cv2.VideoCapture(0)  # Using the default webcam
ct = CentroidTracker(maxDisappeared=40)

while True:
    ret, frame = vs.read()
    if not ret:
        break

    # Resize frame for processing
    frame = imutils.resize(frame, width=600)
    (H, W) = frame.shape[:2]

    # Prepare the frame for the DNN
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []
    # Loop over the detections and filter for "person" class (class index 15)
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            if idx != 15:
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append((startX, startY, endX, endY))
            # (Optional) Draw bounding box on the frame
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Update the centroid tracker with the current frame's detections
    objects = ct.update(rects)

    # (Optional) Draw object IDs and centroids on the frame
    for objectID, centroid in objects.items():
        text = f"ID {objectID}"
        cv2.putText(
            frame,
            text,
            (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Log data to CSV every second
    current_time = time.time()
    if current_time - last_logged_time >= 1:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
        current_count = len(ct.objects)
        csv_writer.writerow([timestamp, totalEntered, totalExited, current_count])
        csv_file.flush()
        last_logged_time = current_time

    # Display counts on the frame
    info = [
        ("Entered", totalEntered),
        ("Exited", totalExited),
        ("Current", len(ct.objects)),
    ]
    for i, (k, v) in enumerate(info):
        text = f"{k}: {v}"
        cv2.putText(
            frame,
            text,
            (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    # Show the processed video
    cv2.imshow("Frame", frame)

    # Press 'q' to quit the program gracefully
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
vs.release()
csv_file.close()
cv2.destroyAllWindows()
