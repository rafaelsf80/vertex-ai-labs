# Cloud profiler demo, with script located at 'script_cloud_profiler.py'
# !pip install google-cloud-aiplatform[cloud-profiler] protobuf==3.20.3
# Service account must have the following permissions:
#  1. Storage Admin (to write logs to GCS)
#  2. Vertex AI User
#  3. Cloud Profiler Agent (otherwise tensorboard will be empty)
# Note: if Profile tab shows waiting, it may be bcause it's a toy dataset and data is not batched.

from google.cloud import aiplatform

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE = 'projects/989788194604/locations/europe-west4/tensorboards/6949581990614007808'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

# Launch Training pipeline, a type of Vertex Training Job.
# A Training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 
job = aiplatform.CustomTrainingJob(
    display_name="cloud_profiler_simple",
    script_path="script_cloud_profiler.py",
    container_uri="europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-16.py310:latest",
    requirements=['google-cloud-aiplatform[cloud-profiler]', 'protobuf==3.20.3'],
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-16.py310:latest",
)


EPOCHS = 20
training_args = [
    "--epochs=" + str(EPOCHS),
]

model = job.run(
    model_display_name='cloud_profiler_simple',
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE,
    bigquery_destination=f'bq://{PROJECT_ID}',   # must provide a destination as Dataset source is BQ
    args=training_args,
)
print(model)
