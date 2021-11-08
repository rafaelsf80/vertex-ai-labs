import datetime
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import xgboost as xgb

from google.cloud import storage
from google.cloud import bigquery

from google.api_core import operations_v1
from google.cloud.aiplatform_v1beta1 import FeaturestoreOnlineServingServiceClient
from google.cloud.aiplatform_v1beta1 import FeaturestoreServiceClient
from google.cloud.aiplatform_v1beta1.types import featurestore_online_service as featurestore_online_service_pb2
from google.cloud.aiplatform_v1beta1.types import entity_type as entity_type_pb2
from google.cloud.aiplatform_v1beta1.types import feature as feature_pb2
from google.cloud.aiplatform_v1beta1.types import featurestore as featurestore_pb2
from google.cloud.aiplatform_v1beta1.types import featurestore_service as featurestore_service_pb2
from google.cloud.aiplatform_v1beta1.types import io as io_pb2

# Set up project, location, featurestore ID and endpoints
PROJECT_ID = "windy-site-254307"  
LOCATION = "us-central1" 
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"  
FEATURESTORE_ID = "fraud_and_risk"
FEATURESTORE_RESOURCE_NAME = "projects/655797269815/locations/us-central1/featurestores/fraud_and_risk"

# Constants required for Batch serving
INPUT_CSV_FILE = "gs://YOURBUCKET-bucket-us/read_entity_instance.csv"  # ground-truth data in GCS. BatchReadFeatureValues API takes this table as input and returns a complete table for training.
DESTINATION_DATA_SET = "OUTPUT_DATASET"  # output dataset in BigQuery
DESTINATION_TABLE_NAME = "batch_serving" # output table. IMPORTANT: Make sure that the table does NOT already exist; otherwise the BatchReadFeatureValues API will not be able to overwrite the table
DESTINATION_PATTERN = "bq://{project}.{dataset}.{table}"
DESTINATION_TABLE_URI = DESTINATION_PATTERN.format(project=PROJECT_ID,
    dataset=DESTINATION_DATA_SET, table=DESTINATION_TABLE_NAME) 


def batch_serving():
    batch_serving_request = featurestore_service_pb2.BatchReadFeatureValuesRequest(
    # featurestore info
    featurestore=admin_client.featurestore_path(PROJECT_ID, LOCATION,
                                                FEATURESTORE_ID),
    # Input file specifying the entities to be read
    csv_read_instances=io_pb2.CsvSource(
        gcs_source=io_pb2.GcsSource(uris=[INPUT_CSV_FILE])),
    # Output info
    destination=featurestore_service_pb2.FeatureValueDestination(
        bigquery_destination=io_pb2.BigQueryDestination(
            # output to BigQuery table
            output_uri=DESTINATION_TABLE_URI)),
    # Select features to read
    entity_type_specs=[
        featurestore_service_pb2.BatchReadFeatureValuesRequest.EntityTypeSpec(
            # read feature values of features from "comercio"
            entity_type_id="comercio", 
            feature_selector=entity_type_pb2.FeatureSelector(
                id_matcher=entity_type_pb2.IdMatcher(ids=[
                    # features, use "*" if you want to select all features within this entity type
                    "num_comercio",  "num_contrato"
                ]))),
        featurestore_service_pb2.BatchReadFeatureValuesRequest.EntityTypeSpec(
            # read feature values of features from "tarjeta"
            entity_type_id="tarjeta",
            feature_selector=entity_type_pb2.FeatureSelector(
                id_matcher=entity_type_pb2.IdMatcher(
                    ids=["is_oper_segura", "tipo_lectura_tarjeta"])))
    ])

    serving_lro = admin_client.batch_read_feature_values(batch_serving_request)

    serving_lro.result()

###################
# Launch batch serving
###################

# Create admin_client for CRUD and data_client for reading feature values.
admin_client = FeaturestoreServiceClient(
    client_options={"api_endpoint": API_ENDPOINT})
data_client = FeaturestoreOnlineServingServiceClient(
    client_options={"api_endpoint": API_ENDPOINT})

# Represents featurestore resource path.
BASE_RESOURCE_PATH = admin_client.common_location_path(PROJECT_ID, LOCATION)

# Batch serving
batch_serving()