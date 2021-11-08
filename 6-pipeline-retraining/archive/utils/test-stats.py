""" Script to call Model Monitoring jobs API """

from google.cloud import aiplatform
from google.cloud.aiplatform_v1beta1.services import job_service

deployed_endpoint = aiplatform.Endpoint('projects/655797269815/locations/us-central1/endpoints/7369586636331417600')
from google.cloud.aiplatform_v1beta1.services.job_service import \
    JobServiceClient
from google.cloud.aiplatform_v1beta1.types import job_service
from google.cloud.aiplatform_v1beta1.types.model_deployment_monitoring_job import (
    ModelDeploymentMonitoringJob, ModelDeploymentMonitoringObjectiveConfig,
    ModelDeploymentMonitoringScheduleConfig, ModelDeploymentMonitoringObjectiveType)

PROJECT_ID = 'windy-site-254307'
REGION = 'us-central1'
MONITORING_JOB='projects/655797269815/locations/us-central1/modelDeploymentMonitoringJobs/245487961133547520'

request=job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest(
    model_deployment_monitoring_job=MONITORING_JOB, 
    deployed_model_id='2508443419794210816', 
    feature_display_name='cnt_user_engagement', 
    objectives=[
                job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest.StatsAnomaliesObjective(
                    type_=ModelDeploymentMonitoringObjectiveType.RAW_FEATURE_SKEW)
                ]
)

print(request)

API_ENDPOINT = 'us-central1-aiplatform.googleapis.com'
def search_model_deployment_monitoring_stats_anomalies():
  client_options = dict(
      api_endpoint = API_ENDPOINT
  )
  client = JobServiceClient(client_options=client_options)
  response = client.search_model_deployment_monitoring_stats_anomalies(request)
  print((response.monitoring_stats)[0].anomaly_count)
  page = list(response.monitoring_stats)
  objective_anomaly_count_map = {}
  for objective_anomaly in page:
    objective_anomaly_count_map[objective_anomaly.objective] = objective_anomaly.anomaly_count
  return objective_anomaly_count_map

search_model_deployment_monitoring_stats_anomalies()



#dict = deployed_endpoint.list()
#print(deployed_endpoint.gca_resource)
#print(deployed_endpoint.gca_resource.deployed_models[0].id)

#list_model_deployment_monitoring_jobs

# endpoint = new_model.deploy(
#         deployed_model_display_name="retraining-B",
#         endpoint = deployed_endpoint,
#         machine_type='n1-standard-4',
#         traffic_split = {"0": 50, deployed_endpoint._gca_resource.deployed_models[0].id: 50}
#     )

