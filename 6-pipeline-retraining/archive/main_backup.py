# Copyright 2021 Google LLC. This software is provided as-is, without warranty
# or representation for any use or purpose. Your use of it is subject to your
# agreement with Google.
import ast
import base64
import datetime
import json
import logging
import uuid
import os

from google.cloud import logging as cloudlogging
from google.cloud import aiplatform


logging_client = cloudlogging.Client()
logging_client.get_default_handler()
logging_client.setup_logging(log_level=logging.INFO)


def get_model_config_json(filename: str) -> dict:
    """Fetches model config json.

    Reads the model config json file from the
    model_configs directory.

    Args:
        filename: The name of json file.

    Returns:
        Json obejct of the model config
    """

    with open(f"./model_configs/{filename}", "r") as f:
        return json.load(f)


def load_sql_query(filename: str) -> str:
    """Fetches SQL query string.

    Reads the SQL query file from the
    dataset_sql directory.

    Args:
        filename: The name of SQL query file.

    Returns:
        String containing the SQL query.
    """

    with open(f"./dataset_sql/{filename}", "r") as f:
        return f.read()


def create_unique_model_name(model_string: str) -> str:
    """Generates unique model name string.

    Generates a unique model display name, this name
    will be reflected in the Vertex AI Models UI.

    Args:
        model_string: String containing the model name.

    Returns:
        String containing the unique model display name.
    """

    uuid_string = str(uuid.uuid4())
    return f"{model_string}-{uuid_string}"


def trigger_pipeline(
    event: dict, context: dict  # pylint: disable=unused-argument
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
    try:
        msg_dict = ast.literal_eval(msg)
        model_config_filename = msg_dict["model_config"]
        config_dict = get_model_config_json(model_config_filename)
    except Exception as e:
        logging.error(
            '"model_config" is either missing or not specified properly and is required'  # pylint:disable=line-too-long
        )  # pylint: disable=line-too-long
        raise e

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


    project_id = os.environ["PROJECT_ID"]
    sql_query = load_sql_query(config_dict["SQL_QUERY_FILENAME"])

    unique_model_string = create_unique_model_name(
        config_dict["MODEL_NAME"]
    )  # pylint:disable=line-too-long

    bq_snapshot = "bq://{}.{}.{}".format(
        project_id, config_dict["SNAPSHOT_DATASET"], unique_model_string
    )

    aiplatform.PipelineJob(
        project=project_id,
        location=config_dict["GCP_REGION"],
        display_name=config_dict["MODEL_NAME"],
        template_path=package_path,
        pipeline_root=f'gs://{config_dict["BUCKET_NAME"]}/{config_dict["PIPELINE_ROOT_DIR"]}',  # pylint: disable=line-too-long
        job_id=job_id,
        parameter_values={
            "project_id": project_id,
            "optimization_prediction_type": config_dict[
                "PIPELINE_TYPE"
            ],
            "optimization_objective": config_dict[
                "OPTIMIZATION_OBJECTIVE"
            ],
            "snapshot_dataset": config_dict["SNAPSHOT_DATASET"],
            "target_column": config_dict["TARGET_COLUMN"],
            "split_column": config_dict["SPLIT_COLUMN"]
            if "SPLIT_COLUMN" in config_dict
            else "",
            "api_endpoint": config_dict["API_ENDPOINT"],
            "deployment_endpoint_name": config_dict["DEPLOYMENT_ENDPOINT_NAME"],
            "model_type": config_dict["MODEL_TYPE"],
            "training_metadata_dataset": config_dict[
                "TRAINING_METADATA_DATASET"
            ],
            "training_metadata_table": config_dict[
                "TRAINING_METADATA_TABLE"
            ],
            "training_attribution_table": config_dict[
                "TRAINING_ATTRIBUTION_TABLE"
            ],
            "serving_machine_type": config_dict["SERVING_MACHINE_TYPE"],
            "budget_milli_node_hours": config_dict[
                "BUDGET_MILLI_NODE_HOURS"
            ],
            "feature_type": json.dumps(
                config_dict["FEATURE_TYPE"]
            ),
            "unique_model_string": unique_model_string,
            "sql_query": sql_query,
            "bq_snapshot": bq_snapshot,
            "metric_column_mapper": config_dict["METRIC_COL_MAPPER"],
            "thresholds_dict_str": json.dumps(config_dict["THRESHOLDS_DICT"]) # pylint: disable=line-too-long
        },
    ).run(sync=False)
    logging.info("training pipeline completed sucessfully")
    print("done")


if __name__ == "__main__":
    payload = {}
    payload["data"] = base64.b64encode(
        "{'model_config':'lane_regression.json'}".encode("utf-8")
    )
    trigger_pipeline(payload, {})
