from google.cloud import aiplatform
import logging

SOURCE_FILE_URI= 'gs://argolis-vertex-europewest4/batch_ulb_gcs_source_names_5.jsonl'

STAGING_BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

model = aiplatform.Model('projects/argolis-rafaelsanchez-ml-dev/locations/europe-west4/models/8042426179879174144') # <--- CHANGE THIS
# destination in GCS: prediction-<model-display-name>-<job-create-time>, where
# timestamp is in YYYY-MM-DDThh:mm:ss.sssZ ISO-8601 format.
# Inside of it files of type predictions_0001.<extension>
batch_prediction_job = model.batch_predict(
    job_display_name='ulb-batch-names-5-final-comma',
    gcs_source=SOURCE_FILE_URI, 
    instances_format='jsonl',
    gcs_destination_prefix=STAGING_BUCKET,
    machine_type="n1-standard-4",
    #accelerator_type= "NVIDIA_TESLA_T4",
    #accelerator_count = 1
)

batch_prediction_job.wait()

logging.info('destination: %s',  STAGING_BUCKET)

logging.info('batch_prediction_job.display_name: %s', batch_prediction_job.display_name)
logging.info('batch_prediction_job.resource_name: %s', batch_prediction_job.resource_name)
logging.info('batch_prediction_job.state: %s', batch_prediction_job.state)