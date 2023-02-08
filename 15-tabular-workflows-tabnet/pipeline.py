import os
from typing import Any, Dict, List
import uuid

from google.cloud import aiplatform, storage
from google_cloud_pipeline_components.experimental.automl.tabular import \
    utils as automl_tabular_utils


BUCKET_URI = 'gs://argolis-vertex-europewest4'  # <---- CHANGE THIS !!!
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'     # <---- CHANGE THIS !!!
REGION = 'europe-west4'                         # <---- CHANGE THIS !!!
BQ_SOURCE = 'bq://argolis-rafaelsanchez-ml-dev.ml_datasets_europewest4.ulb_'         # <---- CHANGE THIS !!!

RUN_FEATURE_SELECTION = True 
FEATURE_SELECTION_ALGORITHM = "AMI"  
MAX_SELECTED_FEATURES = 10  

data_source_bigquery_table_path = BQ_SOURCE # if GCS, set a different parameter
#data_source_csv_filenames = "gs://argolis-vertex-europewest4/ulb.csv"


auto_transform_features=[  # must include all features available
    'Time',
    'Amount',
    'V1',
    'V2',
    'V3',
    'V4',
    'V5',
    'V6',
    'V7',
    'V8',
    'V9',
    'V10',
    'V11',
    'V12',
    'V13',
    'V14',
    'V15',
    'V16',
    'V17',
    'V18',
    'V19',
    'V20',
    'V21',
    'V22',
    'V23',
    'V24',
    'V25',
    'V26',
    'V27',
    'V28',
]

run_evaluation = True 
prediction_type = "classification"
target_column = "Class"

# Fraction split
training_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1

timestamp_split_key = None  # timestamp column name when using timestamp split
stratified_split_key = None  # target column name when using stratified split
training_fraction = 0.8
validation_fraction = 0.1
test_fraction = 0.1

predefined_split_key = None
if predefined_split_key:
    training_fraction = None
    validation_fraction = None
    test_fraction = None

weight_column = None

dataflow_subnetwork = "regions/europe-west4/subnetworks/argolis-rafaelsanchez-vpc-network"
dataflow_use_public_ips = True  
pipeline_job_root_dir = os.path.join(BUCKET_URI, "tabnet_custom_job")

# max_steps and/or max_train_secs must be set. If both are
# specified, training stop after either condition is met.
# By default, max_train_secs is set to -1.

max_steps = 1000
max_train_secs = -1

learning_rate = 0.01

aiplatform.init(project=PROJECT_ID, location=REGION)

# Get the model artifacts path from task details.
def get_model_artifacts_path(task_details: List[Dict[str, Any]], task_name: str) -> str:
    task = get_task_detail(task_details, task_name)
    return task.outputs["unmanaged_container_model"].artifacts[0].uri


# Get the model uri from the task details.
def get_model_uri(task_details: List[Dict[str, Any]]) -> str:
    task = get_task_detail(task_details, "model-upload")
    # in format https://-aiplatform.googleapis.com/v1/projects//locations//models/
    model_id = task.outputs["model"].artifacts[0].uri.split("/")[-1]
    return f"https://console.cloud.google.com/vertex-ai/locations/{REGION}/models/{model_id}?project={PROJECT_ID}"


# Get the bucket name and path.
def get_bucket_name_and_path(uri: str) -> str:
    no_prefix_uri = uri[len("gs://") :]
    splits = no_prefix_uri.split("/")
    return splits[0], "/".join(splits[1:])


# Get the content from the bucket.
def download_from_gcs(uri: str) -> str:
    bucket_name, path = get_bucket_name_and_path(uri)
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    return blob.download_as_string()


# Upload content into the bucket.
def write_to_gcs(uri: str, content: str):
    bucket_name, path = get_bucket_name_and_path(uri)
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(path)
    blob.upload_from_string(content)


# Get the task details by using task name.
def get_task_detail(
    task_details: List[Dict[str, Any]], task_name: str
) -> List[Dict[str, Any]]:
    for task_detail in task_details:
        if task_detail.task_name == task_name:
            return task_detail


# Get the model name from pipeline job ID.
def get_model_name(job_id: str) -> str:
    pipeline_task_details = aiplatform.PipelineJob.get(
        job_id
    ).gca_resource.job_detail.task_details
    upload_task_details = get_task_detail(pipeline_task_details, "model-upload")
    return upload_task_details.outputs["model"].artifacts[0].metadata["resourceName"]


# Get the evaluation metrics.
def get_evaluation_metrics(
    task_details: List[Dict[str, Any]],
) -> str:
    ensemble_task = get_task_detail(task_details, "model-evaluation")
    return download_from_gcs(
        ensemble_task.outputs["evaluation_metrics"].artifacts[0].uri
    )



worker_pool_specs_override = [
    {"machine_spec": {"machine_type": "c2-standard-16"}}  # Override for TF chief node
]

# To test GPU training, the worker_pool_specs_override can be specified like this.
# worker_pool_specs_override =  [
#     {"machine_spec": {
#       'machine_type': "n1-highmem-32",
#       "accelerator_type": "NVIDIA_TESLA_V100",
#       "accelerator_count": 2
#       }
#     }
#   ]

# If your system does not use Python, you can save the JSON file (`template_path`),
# and use another programming language to submit the pipeline.
(
    template_path,
    parameter_values,
) = automl_tabular_utils.get_tabnet_trainer_pipeline_and_parameters(
    project=PROJECT_ID,
    location=REGION,
    root_dir=pipeline_job_root_dir,
    max_steps=max_steps,
    max_train_secs=max_train_secs,
    learning_rate=learning_rate,
    target_column=target_column,
    prediction_type=prediction_type,
    tf_auto_transform_features=auto_transform_features,
    run_feature_selection=RUN_FEATURE_SELECTION,
    feature_selection_algorithm=FEATURE_SELECTION_ALGORITHM,
    max_selected_features=MAX_SELECTED_FEATURES,
    training_fraction=training_fraction,
    validation_fraction=validation_fraction,
    test_fraction=test_fraction,
    #data_source_csv_filenames=data_source_csv_filenames,
    data_source_bigquery_table_path=data_source_bigquery_table_path,
    worker_pool_specs_override=worker_pool_specs_override,
    dataflow_use_public_ips=dataflow_use_public_ips,
    dataflow_subnetwork=dataflow_subnetwork,
    run_evaluation=run_evaluation,
)

pipeline_job_id = f"tabnet-{uuid.uuid4()}"
# More info on parameters PipelineJob accepts:
# https://cloud.google.com/vertex-ai/docs/pipelines/run-pipeline#create_a_pipeline_run
pipeline_job = aiplatform.PipelineJob(
    display_name=pipeline_job_id,
    template_path=template_path,
    job_id=pipeline_job_id,
    pipeline_root=pipeline_job_root_dir,
    parameter_values=parameter_values,
    enable_caching=False,
)

pipeline_job.run()


tabnet_trainer_pipeline_task_details = aiplatform.PipelineJob.get(
    pipeline_job_id
).gca_resource.job_detail.task_details
CUSTOM_JOB_MODEL = get_model_name(pipeline_job_id)
print("model uri:", get_model_uri(tabnet_trainer_pipeline_task_details))
print(
    "model artifacts:",
    get_model_artifacts_path(tabnet_trainer_pipeline_task_details, "tabnet-trainer"),
)



