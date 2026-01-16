""" Custom training pipeline, with script located at 'script_custom_training.py' """


from google.cloud import aiplatform

BUCKET = 'gs://argolis-vertex-europewest4'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'europe-west4'
SERVICE_ACCOUNT = 'tensorboard-sa@argolis-rafaelsanchez-ml-dev.iam.gserviceaccount.com'
TENSORBOARD_RESOURCE = 'projects/989788194604/locations/europe-west4/tensorboards/3449511023961178112'
BQ_SOURCE = 'bq://argolis-rafaelsanchez-ml-dev.ml_datasets_europewest4.ulb_'

# Initialize the *client* for Vertex
aiplatform.init(project=PROJECT_ID, staging_bucket=BUCKET, location=LOCATION)

# BigQuery dataset must be in same location as Vertex project
ds = aiplatform.TabularDataset.create(
    display_name='ulb_dataset',
    bq_source=BQ_SOURCE
)

# Launch Training pipeline, a type of Vertex Training Job.
# A Training pipeline integrates three steps into one job: Accessing a Managed Dataset (not used here), Training, and Model Upload. 
job = aiplatform.CustomTrainingJob(
    display_name="ulb_tf27_custom_training_simple",
    script_path="script_custom_training.py",
    container_uri="europe-docker.pkg.dev/vertex-ai/training/tf-cpu.2-7:latest",
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest",
)

model = job.run(ds,
    model_display_name='ulb-custom-model-simple',
    replica_count=1,
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE,
    bigquery_destination=f'bq://{PROJECT_ID}'   # must provide a destination as Dataset source is BQ
)
print(model)

# Deploy endpoint
endpoint = model.deploy(machine_type='n1-standard-4')
print(endpoint.resource_name)



