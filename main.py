from ultralytics import YOLO
import cv2
from utils.detection_utils import detect_people_quadrants, global_tracker_ids, detect_fire_smoke
from utils.density_utils import compute_density

fire_model = YOLO("fire.pt")
fire_model.model.names = ["fire", "smoke"]

people_model = YOLO("yolov8m.pt")
people_model.model.name = ("people")

video_path = "final1.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]

    fire_boxes = detect_fire_smoke(frame, fire_model)
    for x1, y1, x2, y2, label, color in fire_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    quadrant_counts, total_people, frame = detect_people_quadrants(frame, people_model)

    densities = []
    for key in ["top_left", "top_right", "bottom_left", "bottom_right"]:
        density = compute_density(quadrant_counts[key], total=total_people)
        densities.append(density)

    positions = {
        "top_left": (30, 30),
        "top_right": (w // 2 + 30, 30),
        "bottom_left": (30, h // 2 + 30),
        "bottom_right": (w // 2 + 30, h // 2 + 30),
    }
    for i, key in enumerate(positions):
        cv2.putText(frame, f"{key}: {densities[i]:.2f}", positions[key],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    total_density = round(sum(densities), 2)
    cv2.putText(frame, f'Total Weighted Density: {total_density}', (30, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.putText(frame, f'Total Unique People: {len(global_tracker_ids)}', (30, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Fire + Density Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
