from ultralytics import YOLO
import cv2
from utils.detection_utils import (
    detect_people_quadrants,
    global_tracker_ids,
    detect_fire_smoke,
)
from utils.density_utils import compute_density

fire_model = YOLO("fire.pt")
fire_model.model.names = ["fire", "smoke"]
people_model = YOLO("yolov8m.pt")

cap = cv2.VideoCapture("final1.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))
frame_skip, frame_count = 2,5
fire_boxes, quadrant_counts, densities = [], {}, []
line_color = (0, 0, 0)
line_thickness = 3
border_thickness = 3
threat_detected = False
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    if frame_count % frame_skip == 0:
        fire_boxes = detect_fire_smoke(frame, fire_model,conf=0.40)
        quadrant_counts, total_people, _ = detect_people_quadrants(frame, people_model)
        densities = [
            compute_density(quadrant_counts[key], total_people)
            for key in ["top_left", "top_right", "bottom_left", "bottom_right"]
        ]
    else:
      _, _, _ = detect_people_quadrants(frame, people_model) 
    cv2.line(frame, (w // 2, 0), (w // 2, h), line_color, line_thickness)
    cv2.line(frame, (0, h // 2), (w, h // 2), line_color, line_thickness)

    # Draw outer frame border
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), line_color, border_thickness)
    for x1, y1, x2, y2, label, color in fire_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

    positions = {
        "top_left": (0, 0),
        "top_right": (w // 2, 0),
        "bottom_left": (0, h // 2),
        "bottom_right": (w // 2, h // 2),
    }
    labels = {
        "top_left": "A1",
        "top_right": "A2",
        "bottom_left": "A3",
        "bottom_right": "A4",
    }
   

# Step 1: Determine fire quadrants
    fire_quadrants = set()
    for x1, y1, x2, y2, label, color in fire_boxes:
      cx = (x1 + x2) // 2
      cy = (y1 + y2) // 2
      if cx < w // 2 and cy < h // 2:
        fire_quadrants.add("top_left")
      elif cx >= w // 2 and cy < h // 2:
        fire_quadrants.add("top_right")
      elif cx < w // 2 and cy >= h // 2:
        fire_quadrants.add("bottom_left")
      else:
        fire_quadrants.add("bottom_right")
        
    if fire_quadrants:
       threat_detected = True 
    # Step 2: Draw overlay for each quadrant
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

        cv2.rectangle(
            frame,
            (box_x - 5, box_y - 25),
            (box_x + tw + 5, box_y + 5),
            (50, 50, 50),
            -1,
        )
        cv2.rectangle(
            frame,
            (box_x - 5, box_y - 25),
            (box_x + tw + 5, box_y + 5),
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame, text, (box_x, box_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
    if threat_detected:
       cv2.putText(frame, "THREAT DETECTED", (w//2 + 20, h//2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    if densities:
        cv2.putText(
            frame,
            f"Total Density: {sum(densities):.2f}",
            (30, h - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

    cv2.putText(
        frame,
        f"People: {len(global_tracker_ids)}",
        (30, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 200, 255),
        2,
    )
    
    cv2.imshow("Detection", frame)
    frame_count += 1
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()

cv2.destroyAllWindows()
