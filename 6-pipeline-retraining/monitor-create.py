""" This script will create a Vertex Model monitoring job """

from google.protobuf.duration_pb2 import Duration
from google.cloud import aiplatform

from google.cloud.aiplatform_v1beta1.services.job_service import \
    JobServiceClient
from google.cloud.aiplatform_v1beta1.types.io import GcsSource, BigQuerySource
from google.cloud.aiplatform_v1beta1.types.model_deployment_monitoring_job import (
    ModelDeploymentMonitoringJob, ModelDeploymentMonitoringObjectiveConfig,
    ModelDeploymentMonitoringScheduleConfig)
from google.cloud.aiplatform_v1beta1.types.model_monitoring import (
    ModelMonitoringAlertConfig, ModelMonitoringObjectiveConfig,
    SamplingStrategy, ThresholdConfig)

from datetime import datetime

#### MODIFY ENDPOINT_ID
ENDPOINT_ID = 'projects/655797269815/locations/us-central1/endpoints/8576410598978355200' 

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S") 
PROJECT_ID = 'windy-site-254307'
REGION = 'us-central1'
BUCKET_NAME = 'gs://vertex-model-governance-lab'
STAGING_BUCKET = 'gs://vertex-model-governance-lab/training_job/'
SAMPLE_CSV_EXPORTED_URI = 'gs://vertex-model-governance-lab/000000000000.csv'

features = [
'cnt_post_score', 'cnt_completed_5_levels', 'cnt_level_start_quickplay', 'cnt_ad_reward', 'cnt_level_complete_quickplay', 'cnt_spend_virtual_currency', 'cnt_use_extra_steps', 'cnt_level_end_quickplay', 'cnt_challenge_a_friend', 'cnt_user_engagement', 'cnt_level_reset_quickplay', 'country', 'language', 'operating_system'
]

aiplatform.init(project=PROJECT_ID, location=REGION)
endpoint = aiplatform.Endpoint(project=PROJECT_ID, endpoint_name=ENDPOINT_ID)

random_sampling = SamplingStrategy.RandomSampleConfig(sample_rate=.8)
sampling_config = SamplingStrategy(random_sample_config=random_sampling)
duration = Duration(seconds=3600)
schedule_config = ModelDeploymentMonitoringScheduleConfig(monitor_interval=duration)

emails = ["rafaelsanchez@google.com"]
email_config = ModelMonitoringAlertConfig.EmailAlertConfig(user_emails=emails)
alerting_config = ModelMonitoringAlertConfig(email_alert_config=email_config)

skew_config = ModelMonitoringObjectiveConfig.TrainingPredictionSkewDetectionConfig(
    skew_thresholds={f: ThresholdConfig(value=.001) for f in features})
drift_config = ModelMonitoringObjectiveConfig.PredictionDriftDetectionConfig(
    drift_thresholds={f: ThresholdConfig(value=.001) for f in features})

# This field hasto be set only if TrainingPredictionSkewDetectionConfig is specified.
training_dataset = ModelMonitoringObjectiveConfig.TrainingDataset(
    target_field='churned',
    data_format='csv',
    gcs_source = GcsSource(uris=[SAMPLE_CSV_EXPORTED_URI]))

objective_config = ModelMonitoringObjectiveConfig(
    training_dataset=training_dataset,
    training_prediction_skew_detection_config=skew_config,
    prediction_drift_detection_config=drift_config,
)
#training_dataset.bigquery_source = BigQuerySource(input_uri=DATASET_BQ_URI)

objective_configs = [ModelDeploymentMonitoringObjectiveConfig(
    objective_config=objective_config,
    deployed_model_id=deployed_model.id
) for deployed_model in endpoint.list_models()]

print(endpoint.resource_name)

job = ModelDeploymentMonitoringJob(
    display_name=PROJECT_ID + "-" + TIMESTAMP,
    endpoint=endpoint.resource_name,
    model_deployment_monitoring_objective_configs=objective_configs,
    logging_sampling_strategy=sampling_config,
    model_deployment_monitoring_schedule_config=schedule_config,
    model_monitoring_alert_config=alerting_config,
    predict_instance_schema_uri="",
    analysis_instance_schema_uri="",
)

response = JobServiceClient(client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"}).create_model_deployment_monitoring_job(
    parent=f"projects/{PROJECT_ID}/locations/{REGION}", model_deployment_monitoring_job=job
)
print(response)

