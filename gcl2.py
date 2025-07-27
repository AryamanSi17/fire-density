import cv2
import time
import argparse
from google.cloud import aiplatform

def get_args():
    parser = argparse.ArgumentParser(description="Vehicle Detection using Vertex AI")
    parser.add_argument("--gcs_path", type=str, required=True,
                        help="GCS image URI, e.g. gs://your-bucket/frame.jpg")
    parser.add_argument("--local_frame", type=str, default="frame.jpg",
                        help="Local frame image path for visualization")
    parser.add_argument("--confidence", type=float, default=0.4,
                        help="Confidence threshold")
    parser.add_argument("--project", type=str, required=True,
                        help="GCP Project ID")
    parser.add_argument("--region", type=str, default="us-central1",
                        help="GCP Region where model is deployed")
    parser.add_argument("--endpoint_id", type=str, required=True,
                        help="Vertex AI Endpoint ID")
    parser.add_argument("--show", action="store_true", help="Show annotated image")
    return parser.parse_args()

def main():
    args = get_args()

    frame = cv2.imread(args.local_frame)
    if frame is None:
        raise FileNotFoundError(f"Cannot read {args.local_frame}")
    h, w = frame.shape[:2]


    aiplatform.init(project=args.project, location=args.region)
    endpoint = aiplatform.Endpoint(endpoint_name=args.endpoint_id)

    # Prepare instance
    instance = {
        "image": {"image_uri": args.gcs_path}
    }

    print("🚀 Sending image to Vertex AI endpoint...")
    start = time.time()
    prediction = endpoint.predict(instances=[instance])
    elapsed = time.time() - start
    print(f"✅ Prediction completed in {elapsed:.2f}s")

    detections = prediction.predictions[0].get("detections", [])
    count = 0

    # Draw bounding boxes
    for det in detections:
        label = det["displayName"]
        score = float(det["confidence"])
        if score < args.confidence:
            continue
        box = det["boundingBox"]
        x1, y1 = int(box["xMin"] * w), int(box["yMin"] * h)
        x2, y2 = int(box["xMax"] * w), int(box["yMax"] * h)

        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1

    print(f"📦 Total detections above threshold: {count}")
    out_path = "vertex_output.jpg"
    cv2.imwrite(out_path, frame)
    print(f"🖼 Saved result as {out_path}")

    if args.show:
        cv2.imshow("Detection Output", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if _name_ == "_main_":
    main()