""" Code for a Cloud Function 
    Launch a pipeline if and only if there are alerts in the Model Monitoring service.
    Modify the config.json file accordingly (MONITORING_JOB, DEPLOYED_MODEL_ID, ENDPOINT) 
    Both config.json and retraining-demo-uscentral1.json must be uploaded to GCS """

import base64
import datetime
import json
import logging

from google.cloud import logging as cloudlogging
from google.cloud import aiplatform
from google.cloud import storage

from google.cloud.aiplatform_v1beta1.services.job_service import \
        JobServiceClient
from google.cloud.aiplatform_v1beta1.types import job_service
from google.cloud.aiplatform_v1beta1.types.model_deployment_monitoring_job import (
  ModelDeploymentMonitoringObjectiveType)


logging_client = cloudlogging.Client()
logging_client.get_default_handler()
logging_client.setup_logging(log_level=logging.INFO)

def trigger_pipeline(
    event: dict, context: dict 
) -> None:
    """Triggers a Vertex AI pipeline job

    Triggers a Vertex AI pipeline job, given an event which has
    base-64 encoded data. The input string should additionally be
    utf-8 encoded as well. The data needs to be passed in the
    form of a json string, with the key being "model_config" and
    its respective value being the name of the json file.

    In case of running this the method in the cloud function
    environment, the manual encoding steps can be skipped.

    Args:
        event: The payload data in the form of encoded string.
        context: Optional; The metadata of the triggering event.
    """

    msg = base64.b64decode(event["data"]).decode("utf-8")
    storage_client = storage.Client()

    bucket = storage_client.get_bucket("vertex-labs-uscentral1")
    blob = bucket.blob("config.json")
    contents = blob.download_as_string().decode('utf-8')
    # Process the file contents, etc...
        #logging.info("message received: ", msg)
        #msg_dict = ast.literal_eval(msg)
        #logging.info("msg_dict: ", msg_dict)
        #model_config_filename = msg_dict["model_config"]
    config_dict = json.loads(contents) ## TODO: get from pubsub message

    package_path = "gs://{}/{}/{}.json".format(
        config_dict["BUCKET_NAME"],
        config_dict["COMPILED_PIPELINE_DIR"],  # pylint: disable=line-too-long
        config_dict["PIPELINE_NAME"],
    )

    job_id = "{}-{}".format(
        config_dict["PIPELINE_NAME"],
        datetime.datetime.now().strftime(
            "%Y%m%d%H%M%S"
        ),  # pylint:disable=line-too-long
    )
    logging.info(f"Job id created is {job_id}")

    project_id = config_dict["PROJECT_ID"]
    api_endpoint = config_dict["API_ENDPOINT"]
    monitoring_job = config_dict["MONITORING_JOB"]

    request=job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest(
        model_deployment_monitoring_job=monitoring_job, 
        deployed_model_id=config_dict["DEPLOYED_MODEL_ID"], 
        feature_display_name='cnt_user_engagement', 
        objectives=[
                    job_service.SearchModelDeploymentMonitoringStatsAnomaliesRequest.StatsAnomaliesObjective(
                        type_=ModelDeploymentMonitoringObjectiveType.RAW_FEATURE_SKEW)
                    ]
    )

    client_options = dict(
        api_endpoint = api_endpoint
    )
    client = JobServiceClient(client_options=client_options)

    response = client.search_model_deployment_monitoring_stats_anomalies(request)
    logging.info(response)

    if (config_dict["EXEC_PIPELINE"] == "true"):
        if ((response.monitoring_stats)[0].anomaly_count > 0):
            aiplatform.PipelineJob(
                project=project_id,
                location=config_dict["GCP_REGION"],
                display_name=config_dict["MODEL_NAME"],
                template_path=package_path,
                pipeline_root=f'gs://{config_dict["BUCKET_NAME"]}/{config_dict["PIPELINE_ROOT_DIR"]}',  # pylint: disable=line-too-long
                job_id=job_id,
                enable_caching=False,
                parameter_values={
                    "endpoint": config_dict[
                        "ENDPOINT"
                    ],
                    "previous_model": config_dict[
                        "PREVIOUS_MODEL"
                    ],
                },
            ).run(sync=False)
            logging.info("training pipeline completed sucessfully")
    else:
        logging.info("training not started (EXEC_PIPELINE set to false")

    print("done")

 


# if __name__ == "__main__":
#     payload = {}
#     payload["data"] = base64.b64encode(
#         "{'model_config':'config.json'}".encode("utf-8")
#     )
#     trigger_pipeline(payload, {})
