# Custom training pipeline, with script located at 'script_custom_training.py"'

from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

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

# Launch Training Job
job = aiplatform.CustomJob.from_local_script(display_name='ulb_tf27_custom_training_ht', 
        script_path='script_custom_training_ht.py',
        container_uri='us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-7:latest',
        requirements=['gcsfs==0.7.1', 'cloudml-hypertune'],
        #model_serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-4:latest',
        machine_type="n1-standard-4",
        accelerator_type= "NVIDIA_TESLA_K80",
        accelerator_count = 1)

hp_job = aiplatform.HyperparameterTuningJob(
    display_name='ulb_tf27_custom_training_ht',
    custom_job=job,
    metric_spec={'accuracy': 'maximize'},
    parameter_spec={
        'lr': hpt.DoubleParameterSpec(min=0.01, max=0.1, scale='log'),
        'units': hpt.IntegerParameterSpec(min=8, max=32, scale='linear'), # not used
        'activation': hpt.CategoricalParameterSpec(values=['relu', 'tanh']),  
        'batch_size': hpt.DiscreteParameterSpec(values=[32, 64, 128], scale='linear')
    },
    max_trial_count=16,
    parallel_trial_count=4,    
    )

hp_job.run( 
    service_account = SERVICE_ACCOUNT,
    tensorboard = TENSORBOARD_RESOURCE)


