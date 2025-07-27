from ultralytics import YOLO
import cv2
import numpy as np
from scipy.spatial import distance as dist

class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=50, max_trail_length=10):
        self.nextObjectID = 0
        self.objects = {}
        self.disappeared = {}
        self.trails = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.max_trail_length = max_trail_length

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.trails[self.nextObjectID] = [centroid]
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.trails[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x1, y1, x2, y2)) in enumerate(rects):
            cX = int((x1 + x2) / 2.0)
            cY = int((y1 + y2) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            objectIDs = list(self.objects.keys())

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                if D[row, col] > self.max_distance:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                
                self.trails[objectID].append(tuple(input_centroids[col]))
                if len(self.trails[objectID]) > self.max_trail_length:
                    self.trails[objectID].pop(0)
                
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unused_cols:
                    self.register(input_centroids[col])

        return self.objects

def detect_people_quadrants(frame, model):
    results = model(frame, classes=[0], conf=0.3, verbose=False)
    
    people_boxes = []
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                people_boxes.append((x1, y1, x2, y2))
    
    h, w = frame.shape[:2]
    quadrant_counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}
    
    for (x1, y1, x2, y2) in people_boxes:
        cX = int((x1 + x2) / 2.0)
        cY = int((y1 + y2) / 2.0)
        
        if cX < w // 2 and cY < h // 2:
            quadrant_counts["top_left"] += 1
        elif cX >= w // 2 and cY < h // 2:
            quadrant_counts["top_right"] += 1
        elif cX < w // 2 and cY >= h // 2:
            quadrant_counts["bottom_left"] += 1
        else:
            quadrant_counts["bottom_right"] += 1
    
    return quadrant_counts, len(people_boxes), people_boxes

def detect_fire_smoke(frame, fire_model, conf=0.40):
    results = fire_model(frame, conf=conf, verbose=False)
    fire_boxes = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                label = fire_model.model.names[cls]
                color = (0, 0, 255) if label == "fire" else (255, 0, 0)
                fire_boxes.append((x1, y1, x2, y2, label, color))
    
    return fire_boxes

def compute_density(count, total):
    return count / max(total, 1)

fire_model = YOLO("fire.pt")
fire_model.model.names = ["fire", "smoke"]
people_model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("final1.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

frame_skip, frame_count = 2, 0
fire_boxes, quadrant_counts, densities = [], {}, []
line_color, line_thickness, border_thickness = (255, 255, 255), 3, 3
threat_detected = False

tracker = CentroidTracker(max_disappeared=40, max_distance=50, max_trail_length=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if frame_count % frame_skip == 0:
        fire_boxes = detect_fire_smoke(frame, fire_model, conf=0.40)
        quadrant_counts, total_people, person_boxes = detect_people_quadrants(frame, people_model)
        densities = [
            compute_density(quadrant_counts.get(key, 0), total_people)
            for key in ["top_left", "top_right", "bottom_left", "bottom_right"]
        ]
        tracked_objects = tracker.update(person_boxes)
    else:
        _, _, person_boxes = detect_people_quadrants(frame, people_model)
        tracked_objects = tracker.update(person_boxes)
        if not densities:
            densities = [0, 0, 0, 0]

    cv2.line(frame, (w // 2, 0), (w // 2, h), line_color, line_thickness)
    cv2.line(frame, (0, h // 2), (w, h // 2), line_color, line_thickness)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), line_color, border_thickness)

    for x1, y1, x2, y2, label, color in fire_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    fire_quadrants = set()
    for x1, y1, x2, y2, _, _ in fire_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if cx < w // 2 and cy < h // 2: 
            fire_quadrants.add("top_left")
        elif cx >= w // 2 and cy < h // 2: 
            fire_quadrants.add("top_right")
        elif cx < w // 2 and cy >= h // 2: 
            fire_quadrants.add("bottom_left")
        else: 
            fire_quadrants.add("bottom_right")

    threat_detected = bool(fire_quadrants)

    positions = {
        "top_left": (0, 0), "top_right": (w // 2, 0),
        "bottom_left": (0, h // 2), "bottom_right": (w // 2, h // 2)
    }
    labels = {"top_left": "A1", "top_right": "A2", "bottom_left": "A3", "bottom_right": "A4"}

    for i, key in enumerate(["top_left", "top_right", "bottom_left", "bottom_right"]):
        x_off, y_off = positions[key]
        label = labels[key]
        density = densities[i] if i < len(densities) else 0.0
        people_count = quadrant_counts.get(key, 0)

        if key in fire_quadrants:
            cv2.rectangle(frame, (x_off, y_off), (x_off + w // 2, y_off + h // 2), (0, 0, 255), 2)

        text = f"{label} D: {density:.2f} P: {people_count}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        box_x, box_y = x_off + 10, y_off + 30

        cv2.rectangle(frame, (box_x - 5, box_y - 25), (box_x + tw + 5, box_y + 5), (50, 50, 50), -1)
        cv2.rectangle(frame, (box_x - 5, box_y - 25), (box_x + tw + 5, box_y + 5), (0, 255, 0), 2)
        cv2.putText(frame, text, (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (173, 216, 230), 2)

    if threat_detected:
        cv2.putText(frame, "THREAT DETECTED", (w // 2 + 20, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for objectID, centroid in tracked_objects.items():
        trail = tracker.trails.get(objectID, [])
        
        for i in range(1, len(trail)):
            thickness = max(1, int(3 * (i / len(trail))))
            alpha = i / len(trail)
            color = (int(255 * alpha), int(255 * alpha), 0)
            cv2.line(frame, trail[i-1], trail[i], color, thickness)
        
        cv2.circle(frame, tuple(centroid), 4, (255, 255, 0), -1)
        cv2.putText(frame, f"ID {objectID}", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    if densities:
        cv2.putText(frame, f"Total Density: {sum(densities):.2f}", (30, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.putText(frame, f"People: {len(tracked_objects)}", (30, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Detection", frame)
    frame_count += 1
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()