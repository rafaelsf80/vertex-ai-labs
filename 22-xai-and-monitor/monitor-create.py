""" Step 2: Create Model monitoring job """

import copy

from google.cloud.aiplatform_v1beta1.services.endpoint_service import \
    EndpointServiceClient
from google.cloud.aiplatform_v1beta1.services.job_service import \
    JobServiceClient
from google.cloud.aiplatform_v1beta1.types.io import BigQuerySource
from google.cloud.aiplatform_v1beta1.types.model_deployment_monitoring_job import (
    ModelDeploymentMonitoringJob, ModelDeploymentMonitoringObjectiveConfig,
    ModelDeploymentMonitoringScheduleConfig)
from google.cloud.aiplatform_v1beta1.types.model_monitoring import (
    ModelMonitoringAlertConfig, ModelMonitoringObjectiveConfig,
    SamplingStrategy, ThresholdConfig)
from google.protobuf.duration_pb2 import Duration


PROJECT_ID = "argolis-rafaelsanchez-ml-dev"    # <---- CHANGE THIS
REGION = "europe-west4"                        # <---- CHANGE THIS
SUFFIX = "aiplatform.googleapis.com"
API_ENDPOINT = f"{REGION}-{SUFFIX}"
PREDICT_API_ENDPOINT = f"{REGION}-prediction-{SUFFIX}"

DEFAULT_THRESHOLD_VALUE = 0.001
ENDPOINT_ID_VALUE = '97091274779131904'     # <---- CHANGE THIS


def create_monitoring_job(objective_configs):
    # Create sampling configuration.
    random_sampling = SamplingStrategy.RandomSampleConfig(sample_rate=LOG_SAMPLE_RATE)
    sampling_config = SamplingStrategy(random_sample_config=random_sampling)

    # Create schedule configuration.
    duration = Duration(seconds=MONITOR_INTERVAL)
    schedule_config = ModelDeploymentMonitoringScheduleConfig(monitor_interval=duration)

    # Create alerting configuration.
    emails = [USER_EMAIL]
    email_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails=emails)
    alerting_config = ModelMonitoringAlertConfig(email_alert_config=email_config)

    # Create the monitoring job.
    endpoint = f"projects/{PROJECT_ID}/locations/{REGION}/endpoints/{ENDPOINT_ID_VALUE}"
    predict_schema = ""
    analysis_schema = ""
    job = ModelDeploymentMonitoringJob(
        display_name=JOB_NAME,
        endpoint=endpoint,
        model_deployment_monitoring_objective_configs=objective_configs,
        logging_sampling_strategy=sampling_config,
        model_deployment_monitoring_schedule_config=schedule_config,
        model_monitoring_alert_config=alerting_config,
        predict_instance_schema_uri=predict_schema,
        analysis_instance_schema_uri=analysis_schema,
    )
    options = dict(api_endpoint=API_ENDPOINT)
    client = JobServiceClient(client_options=options)
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    response = client.create_model_deployment_monitoring_job(
        parent=parent, model_deployment_monitoring_job=job
    )
    print("Created monitoring job:")
    print(response)
    return response


def get_thresholds(default_thresholds, custom_thresholds):
    thresholds = {}
    default_threshold = ThresholdConfig(value=DEFAULT_THRESHOLD_VALUE)
    for feature in default_thresholds.split(","):
        feature = feature.strip()
        thresholds[feature] = default_threshold
    for custom_threshold in custom_thresholds.split(","):
        pair = custom_threshold.split(":")
        if len(pair) != 2:
            print(f"Invalid custom skew threshold: {custom_threshold}")
            return
        feature, value = pair
        thresholds[feature] = ThresholdConfig(value=float(value))
    return thresholds


def get_deployed_model_ids(endpoint_id):
    client_options = dict(api_endpoint=API_ENDPOINT)
    client = EndpointServiceClient(client_options=client_options)
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    response = client.get_endpoint(name=f"{parent}/endpoints/{endpoint_id}")
    model_ids = []
    for model in response.deployed_models:
        model_ids.append(model.id)
    return model_ids


def set_objectives(model_ids, objective_template):
    # Use the same objective config for all models.
    objective_configs = []
    for model_id in model_ids:
        objective_config = copy.deepcopy(objective_template)
        objective_config.deployed_model_id = model_id
        objective_configs.append(objective_config)
    return objective_configs


def list_monitoring_jobs():
    client_options = dict(api_endpoint=API_ENDPOINT)
    parent = f"projects/{PROJECT_ID}/locations/{REGION}"
    client = JobServiceClient(client_options=client_options)
    response = client.list_model_deployment_monitoring_jobs(parent=parent)
    print(response)


def pause_monitoring_job(job):
    client_options = dict(api_endpoint=API_ENDPOINT)
    client = JobServiceClient(client_options=client_options)
    response = client.pause_model_deployment_monitoring_job(name=job)
    print(response)


def delete_monitoring_job(job):
    client_options = dict(api_endpoint=API_ENDPOINT)
    client = JobServiceClient(client_options=client_options)
    response = client.delete_model_deployment_monitoring_job(name=job)
    print(response)



##########################
##### CREATE MODEL MONITOR
##########################

USER_EMAIL = "rafaelsanchez@google.com"
JOB_NAME = "churn"

# Sampling rate (optional, default=.8)
LOG_SAMPLE_RATE = 0.8

# Monitoring Interval in seconds (optional, default=3600).
MONITOR_INTERVAL = 3600 

# URI to training dataset.
DATASET_BQ_URI = "bq://argolis-rafaelsanchez-ml-dev.bqmlga4.train"  # <---- CHANGE THIS: Copy in EU region of "bq://mco-mm.bqmlga4.train" 
TARGET = "churned"

# Skew and drift thresholds.
SKEW_DEFAULT_THRESHOLDS = "country,language"  
SKEW_CUSTOM_THRESHOLDS = "cnt_user_engagement:.5"  
DRIFT_DEFAULT_THRESHOLDS = "country,language" 
DRIFT_CUSTOM_THRESHOLDS = "cnt_user_engagement:.5" 

skew_thresholds = get_thresholds(SKEW_DEFAULT_THRESHOLDS, SKEW_CUSTOM_THRESHOLDS)
drift_thresholds = get_thresholds(DRIFT_DEFAULT_THRESHOLDS, DRIFT_CUSTOM_THRESHOLDS)
skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
    skew_thresholds=skew_thresholds
)
drift_config = ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
    drift_thresholds=drift_thresholds
)
training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(target_field=TARGET)
training_dataset.bigquery_source = BigQuerySource(input_uri=DATASET_BQ_URI)
objective_config = ModelMonitoringObjectiveConfig(
    training_dataset=training_dataset,
    training_prediction_skew_detection_config=skew_config,
    prediction_drift_detection_config=drift_config,
)
model_ids = get_deployed_model_ids(ENDPOINT_ID_VALUE)
objective_template = ModelDeploymentMonitoringObjectiveConfig(
    objective_config=objective_config
)
objective_configs = set_objectives(model_ids, objective_template)

monitoring_job = create_monitoring_job(objective_configs)