from google.cloud import aiplatform
from google.cloud import storage
import cv2
import numpy as np
import base64

def detect_vehicles_persons(gcs_image_uri, project_id, location="us-east1", confidence_threshold=0.5):
    aiplatform.init(project=project_id, location=location)
    
    storage_client = storage.Client()
    bucket_name = gcs_image_uri.split('/')[2]
    blob_name = '/'.join(gcs_image_uri.split('/')[3:])
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_bytes = blob.download_as_bytes()
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = image_cv.shape[:2]
    
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    
    client = aiplatform.gapic.PredictionServiceClient()
    
    endpoint = f"projects/{project_id}/locations/{location}/publishers/google/models/vehicle-detector"
    
    instance = {
        "image": {
            "bytesBase64Encoded": encoded_image
        }
    }
    
    instances = [instance]
    response = client.predict(endpoint=endpoint, instances=instances)
    
    person_count = 0
    vehicle_count = 0
    
    for prediction in response.predictions:
        for detection in prediction:
            confidence = detection.get('confidence', 0)
            if confidence < confidence_threshold:
                continue
                
            display_name = detection.get('displayName', '').lower()
            
            bbox = detection['boundingBox']
            x1 = int(bbox['xMin'] * w)
            y1 = int(bbox['yMin'] * h)
            x2 = int(bbox['xMax'] * w)
            y2 = int(bbox['yMax'] * h)
            
            if 'person' in display_name:
                person_count += 1
                color = (0, 255, 0)
                label = f"Person {confidence:.2f}"
            elif any(vehicle in display_name for vehicle in ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle']):
                vehicle_count += 1
                color = (0, 0, 255)
                label = f"{detection['displayName']} {confidence:.2f}"
            else:
                continue
            
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_cv, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return image_cv, person_count, vehicle_count

project_id = "computervisionporject"
gcs_image_uri = "gs://my-vision1-bucket/frame.jpg"

# Try different regions
regions = ["us-east1", "us-west1", "europe-west1", "asia-southeast1"]

for region in regions:
    try:
        print(f"Trying region: {region}")
        result_image, persons, vehicles = detect_vehicles_persons(gcs_image_uri, project_id, location=region, confidence_threshold=0.4)
        
        print(f"Success with region: {region}")
        print(f"Persons detected: {persons}")
        print(f"Vehicles detected: {vehicles}")
        print(f"Total detections: {persons + vehicles}")
        
        output_path = "vehicle_detector_output.jpg"
        cv2.imwrite(output_path, result_image)
        print(f"Result saved: {output_path}")
        
        cv2.imshow("Vehicle/Person Detection", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
        
    except Exception as e:
        print(f"Failed with region {region}: {str(e)}")
        continue