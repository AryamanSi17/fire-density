from ultralytics import YOLO
import cv2

global_tracker_ids = set()

def detect_fire_smoke(frame, model, conf=0.5):
    # Restrict to only classes in your custom model
    results = model(frame, conf=conf, classes=[0, 1], verbose=False)
    fire_boxes = []

    for result in results:
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box, cls_id, conf_score in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                if conf_score < conf:
                    continue  # Skip detections below new threshold
                idx = int(cls_id)
                label = model.names[idx].lower() if 0 <= idx < len(model.names) else f"class_{idx}"
                color = (0, 0, 255) if "fire" in label else (128, 128, 128)
                x1, y1, x2, y2 = map(int, box)
                fire_boxes.append((x1, y1, x2, y2, label, color))

    return fire_boxes


def detect_people_quadrants(frame, model):
    h, w = frame.shape[:2]
    counts = {"top_left": 0, "top_right": 0, "bottom_left": 0, "bottom_right": 0}

    results = model.track(source=frame, persist=True, verbose=False, tracker="botsort.yaml")[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if model.names[cls_id] != "person":
            continue

        track_id = int(box.id[0]) if hasattr(box, "id") and box.id is not None else None
        if track_id is not None:
            global_tracker_ids.add(track_id)

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if cx < w // 2 and cy < h // 2:
            counts["top_left"] += 1
        elif cx >= w // 2 and cy < h // 2:
            counts["top_right"] += 1
        elif cx < w // 2 and cy >= h // 2:
            counts["bottom_left"] += 1
        else:
            counts["bottom_right"] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if track_id is not None:
            cv2.putText(frame, f'ID:{track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    total = sum(counts.values())
    return counts, total, frame
