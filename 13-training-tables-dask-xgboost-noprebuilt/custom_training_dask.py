# https://github.com/GoogleCloudPlatform/vertex-ai-samples/blob/main/notebooks/official/training/xgboost_data_parallel_training_on_cpu_using_dask.ipynb

import os
from google.cloud import aiplatform

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
BUCKET_URI = 'gs://argolis-vertex-europewest4'
OUTPUT_URI = f"{BUCKET_URI}/output"
LOCATION = 'europe-west4'

TRAIN_IMAGE = f'europe-west4-docker.pkg.dev/{PROJECT_ID}/ml-pipelines-repo/13-training-tables-xgboost-noprebuilt:latest'
DEPLOY_IMAGE = "europe-west4-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest"

aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET_URI, location=LOCATION)

custom_container_training_job = aiplatform.CustomContainerTrainingJob(
    display_name="xgboost_dask",
    model_serving_container_image_uri=DEPLOY_IMAGE,
    container_uri=TRAIN_IMAGE,
)

custom_container_training_job.run(
    base_output_dir=OUTPUT_URI,
    replica_count=2,
    machine_type="n1-standard-4",
    enable_dashboard_access=True,
    enable_web_access=True,
    sync=True,
)

try:
    print(f"Custom Training Job Name: {custom_container_training_job.resource_name}")
    print(f"GCS Output URI Prefix: {OUTPUT_URI}")
except Exception as e:
    print(e)

try:
    print(
        f"Custom Training Job URI: {custom_container_training_job._custom_job_console_uri()}"
    )
except Exception as e:
    print(e)

try:
    print(
        f"Web Access and Dashboard URIs: {custom_container_training_job.web_access_uris}"
    )
except Exception as e:
    print(e)