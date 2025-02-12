import cv2
import numpy as np
import imutils
from scipy.spatial import distance as dist


# ------------------ Centroid Tracker Class ------------------
class CentroidTracker:
    def __init__(self, maxDisappeared=40):
        # Next available ID for a new object
        self.nextObjectID = 0
        # Dictionary mapping object ID → centroid, i.e. 𝑥 and 𝑦 coordinates
        self.objects = {}
        # Dictionary mapping object ID → number of consecutive frames it has been missing
        self.disappeared = {}
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # Register a new object with centroid 𝒄 = (𝑐ₓ, 𝑐ᵧ)
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # Remove object from tracking
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # If no detections, mark all existing objects as disappeared
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        # Compute centroids for the current set of bounding boxes
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for i, (startX, startY, endX, endY) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # If no objects are tracked, register each centroid 𝒄ᵢ
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            # Grab the set of object IDs and their centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # Compute the Euclidean distance 𝑑(𝒄ᵢ, 𝒄ⱼ) between each pair
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            # Associate each object with the nearest centroid
            for row, col in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            # Identify unused rows and columns
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # If there are more tracked objects than new detections,
            # mark the unmatched objects as disappeared
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
            # Otherwise, register each new input centroid as a new object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        return self.objects


# ------------------ Trackable Object Class ------------------
class TrackableObject:
    def __init__(self, objectID, centroid):
        # 𝑂 represents a trackable object with unique ID and its centroid history
        self.objectID = objectID
        self.centroids = [centroid]
        self.counted = False


# ------------------ Load Pre-trained Person Detector ------------------
prototxt = "MobileNetSSD_deploy.prototxt.txt"  # Path to Caffe 'deploy' prototxt file
model = "MobileNetSSD_deploy.caffemodel"  # Path to pre-trained Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# ------------------ Initialize Video Stream and Variables ------------------
vs = cv2.VideoCapture(0)
ct = CentroidTracker(maxDisappeared=40)
trackableObjects = {}

totalIn = 0  # Count of people entering (e.g. moving downward)
totalOut = 0  # Count of people exiting (e.g. moving upward)

(W, H) = (None, None)

# ------------------ Main Loop ------------------
while True:
    ret, frame = vs.read()
    if not ret:
        break

    frame = imutils.resize(frame, width=600)
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # Define the counting line at 𝑦 = H/2
    linePosition = H // 2

    # Prepare the image for detection
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()

    rects = []
    # Loop over the detections
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.4:
            idx = int(detections[0, 0, i, 1])
            # Class label 15 corresponds to "person"
            if idx != 15:
                continue
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")
            rects.append((startX, startY, endX, endY))
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Update our centroid tracker with the current set of bounding boxes
    objects = ct.update(rects)

    # Loop over the tracked objects to determine direction and count crossings
    for objectID, centroid in objects.items():
        text = "ID {}".format(objectID)
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

        # If we haven't seen this object before, create a new trackable object 𝑂
        if objectID not in trackableObjects:
            trackableObjects[objectID] = TrackableObject(objectID, centroid)
        else:
            to = trackableObjects[objectID]
            # Compute the difference in the y-coordinate (𝑑𝑦)
            y_positions = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y_positions)
            to.centroids.append(centroid)

            # If the object has not yet been counted, check if it crossed the line
            if not to.counted:
                # If moving downward and crosses the line (𝑦 from < H/2 to > H/2), count as "In"
                if (
                    direction > 0
                    and centroid[1] > linePosition
                    and np.mean(y_positions) <= linePosition
                ):
                    totalIn += 1
                    to.counted = True
                # If moving upward and crosses the line (𝑦 from > H/2 to < H/2), count as "Out"
                elif (
                    direction < 0
                    and centroid[1] < linePosition
                    and np.mean(y_positions) >= linePosition
                ):
                    totalOut += 1
                    to.counted = True

    # Draw the counting line on the frame
    cv2.line(frame, (0, linePosition), (W, linePosition), (0, 0, 255), 2)

    # Display the counts on the frame
    info = [("In", totalIn), ("Out", totalOut)]
    for i, (k, v) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(
            frame,
            text,
            (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()
