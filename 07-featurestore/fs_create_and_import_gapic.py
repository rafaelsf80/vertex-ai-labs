from google.api_core import operations_v1
from google.cloud.aiplatform_v1 import FeaturestoreOnlineServingServiceClient
from google.cloud.aiplatform_v1 import FeaturestoreServiceClient
from google.cloud.aiplatform_v1.types import FeatureSelector, IdMatcher

from google.cloud.aiplatform_v1.types import featurestore_online_service as featurestore_online_service_pb2
from google.cloud.aiplatform_v1.types import entity_type as entity_type_pb2
from google.cloud.aiplatform_v1.types import feature as feature_pb2
from google.cloud.aiplatform_v1.types import featurestore as featurestore_pb2
from google.cloud.aiplatform_v1.types import featurestore_service as featurestore_service_pb2
from google.cloud.aiplatform_v1.types import io as io_pb2
from google.cloud.aiplatform_v1.types import featurestore_monitoring as featurestore_monitoring_pb2
from google.protobuf.duration_pb2 import Duration

from datetime import datetime

# Set up project, location, featurestore ID and endpoints
PROJECT_ID = "argolis-rafaelsanchez-ml-dev"  
LOCATION = "us-central1" 
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"  
FEATURESTORE_ID = "fraud_detection_demo_monitoring"
FEATURESTORE_RESOURCE_NAME = "projects/989788194604/locations/us-central1/featurestores/fraud_detection_demo_monitoring"

# Create admin_client for CRUD and data_client for reading feature values.
admin_client = FeaturestoreServiceClient(
    client_options={"api_endpoint": API_ENDPOINT})
data_client = FeaturestoreOnlineServingServiceClient(
    client_options={"api_endpoint": API_ENDPOINT})


# Create operation client to poll LRO status.
def create_fs():
    lro_client = operations_v1.OperationsClient(admin_client.transport.grpc_channel)

    create_lro = admin_client.create_featurestore(
        featurestore_service_pb2.CreateFeaturestoreRequest(
            parent=BASE_RESOURCE_PATH,
            featurestore_id=FEATURESTORE_ID,
            featurestore=featurestore_pb2.Featurestore(
                name="Fraud Detection features with monitoring",
                online_serving_config=featurestore_pb2.Featurestore
                .OnlineServingConfig(fixed_node_count=3))))

    print(create_lro.result())
    print('Checking feature store global details...')
    admin_client.get_featurestore(name = admin_client.featurestore_path(PROJECT_ID, LOCATION, FEATURESTORE_ID))


## Create entity 'transaction' and 5 features: amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
def create_transaction_entity():
    print(
         admin_client.create_entity_type(
             featurestore_service_pb2.CreateEntityTypeRequest(
                 parent=admin_client.featurestore_path(PROJECT_ID, LOCATION,
                                                     FEATURESTORE_ID),
                 entity_type_id="transaction",
                 entity_type=entity_type_pb2.EntityType(
                     description="Transaction features",
                     monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            monitoring_interval=Duration(seconds=3600),  # 1 day
                )),            
            ))).result())


    admin_client.batch_create_features(
        parent=admin_client.entity_type_path(PROJECT_ID, LOCATION,
                                            FEATURESTORE_ID, "transaction"),
        requests=[
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description="Amount of transaction",
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            monitoring_interval=Duration(seconds=3600),  # 2 days
                        ),
                    ),
                ),
                feature_id="amount"),
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description="Old balance origin",
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            monitoring_interval=Duration(seconds=3600),  # 2 days
                        ),
                    ),
                ),
                feature_id="oldbalance_orig"),
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description="New balance origin",
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            monitoring_interval=Duration(seconds=3600),  # 2 days
                        ),
                    ),
                ),
                feature_id="newbalance_orig"),
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description="Old balance destination",
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            disabled=True,
                        ),
                    ),
                ),
                feature_id="oldbalance_dest"),
            featurestore_service_pb2.CreateFeatureRequest(
                feature=feature_pb2.Feature(
                    value_type=feature_pb2.Feature.ValueType.DOUBLE,
                    description="New balance destination",
                    monitoring_config=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig(
                        snapshot_analysis=featurestore_monitoring_pb2.FeaturestoreMonitoringConfig.SnapshotAnalysis(
                            disabled=True,
                        ),
                    ),
                ),
                feature_id="newbalance_dest")
        ])

# Batch ingest (import data from GCS)
# Transaction entity. Expect o(10 minutes) if the number of rows is large.
TIMESTAMP=datetime(2021, 1, 1, 9, 30)
def batch_ingestion_transactions():

    import_request_transaction = featurestore_service_pb2.ImportFeatureValuesRequest(
        entity_type=admin_client.entity_type_path(PROJECT_ID, LOCATION,
                                                FEATURESTORE_ID, "transaction"),
        csv_source=io_pb2.CsvSource(
            gcs_source=io_pb2.GcsSource(uris=["gs://argolis-vertex-uscentral1/fraud_data_kaggle_5000.csv"]) # 500 internal error for 1M rows, works with 5000 rows
        ),
        feature_specs=[
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id="amount"),
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id="oldbalance_orig", source_field = 'oldbalanceOrg'),
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id="newbalance_orig", source_field = 'newbalanceOrig'),
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id="oldbalance_dest", source_field = 'newbalanceDest'),
            featurestore_service_pb2.ImportFeatureValuesRequest.FeatureSpec(
                id="newbalance_dest", source_field = 'newbalanceDest')
        ],
        entity_id_field="nameOrig", # <--- NOTICE ENTITY_ID. MUST BE STRING TYPE AND MUST EXIST IN THE TABLE
        #feature_time_field="isFraud", 
        feature_time=TIMESTAMP, # unique timestamp for all
        worker_count=5)
    print('Batch ingestion for "transaction" entity ...')
    ingestion_lro = admin_client.import_feature_values(import_request_transaction)
    
    # Polls for the LRO status and prints when the LRO has completed
    ingestion_lro.result()


def online_serving():
    print("Online serving...")

    feature_selector = FeatureSelector(
        id_matcher=IdMatcher(ids=["oldbalance_orig", "newbalance_orig"])
    )

    print(data_client.read_feature_values(    
        featurestore_online_service_pb2.ReadFeatureValuesRequest(
            entity_type=admin_client.entity_type_path(PROJECT_ID, LOCATION, FEATURESTORE_ID, "transaction"),
            entity_id="C1231006815",
            feature_selector=feature_selector,)))
    



###################
# Uncomment only the STEPS you want to execute
###################



# Represents featurestore resource path.
BASE_RESOURCE_PATH = admin_client.common_location_path(PROJECT_ID, LOCATION)

# STEP 1: Create feature store, entities and features. You MUST only do this ONCE

create_fs()
# STEP 2: Create entity "transaction" with its 5 features You MUST only do this ONCE
create_transaction_entity()

# Check that features are created
for entity_type in admin_client.list_entity_types(
    parent=FEATURESTORE_RESOURCE_NAME):
    num_features = len(list(admin_client.list_features(parent=entity_type.name)))
    print("Entity type {} with {} features".format(entity_type.name, 
                                                num_features))

# Check all features from all Feature stores in the project
print("********** All features from all Feature Stores")
print(list(admin_client.search_features(location=BASE_RESOURCE_PATH)))
print("********** All Feature Stores")
print(list(admin_client.list_featurestores(parent=BASE_RESOURCE_PATH)))

# # STEP 3: Batch ingestion. GCS must be in the same region as Feature Store
batch_ingestion_transactions()

# # STEP 4: Online serving
online_serving()

# STEP 5: Remove Feature Store
# FEATURESTORE_ID="fraud_detection_demo_monitoring"
# print("********** Remove Feature Store ", FEATURESTORE_ID)
# admin_client.delete_featurestore(name=admin_client.featurestore_path(PROJECT_ID, LOCATION,
#                                         FEATURESTORE_ID), force=True).result()