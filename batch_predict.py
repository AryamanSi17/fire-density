from google.cloud import aiplatform
from datetime import datetime

# Constants
PROJECT_ID = "computervisionporject"
REGION = "us-central1"
MODEL_NAME = "publishers/google/models/vehicle-detector"
INPUT_GCS = "gs://my-vision1-bucket/frame.jpg"
OUTPUT_GCS = "gs://my-vision1-bucket/predictions/"
JOB_DISPLAY_NAME = f"vehicle-detect-batch-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

def run_batch_prediction():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    batch_predict_job = aiplatform.BatchPredictionJob.create(
        job_display_name=JOB_DISPLAY_NAME,
        model=MODEL_NAME,
        instances_format="jsonl",
        predictions_format="jsonl",
        gcs_source=INPUT_GCS,
        gcs_destination_prefix=OUTPUT_GCS,
        sync=True  # Wait for job to finish
    )

    print(f"✅ Job completed. Output saved to: {OUTPUT_GCS}")

if __name__ == "__main__":
    run_batch_prediction()
