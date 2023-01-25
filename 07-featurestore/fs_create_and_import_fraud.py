from datetime import datetime
from google.cloud.aiplatform import Feature, Featurestore


PROJECT_ID = "argolis-rafaelsanchez-ml-dev"  
LOCATION = "europe-west4" 
FEATURESTORE_ID = "fraud_detection_demo_monitoring"
ONLINE_STORE_FIXED_NODE_COUNT = 1


# Create operation client to poll LRO status.
fs = Featurestore.create(
    featurestore_id=FEATURESTORE_ID,
    online_store_fixed_node_count=ONLINE_STORE_FIXED_NODE_COUNT,
    project=PROJECT_ID,
    location=LOCATION,
    sync=True,
)
fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=LOCATION,
)
print(fs.gca_resource)


## Create entity 'transaction' and 5 features: amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest

transactions_entity_type = fs.create_entity_type(
    entity_type_id="transaction",
    description="Transaction features",
)
transactions_entity_type = fs.get_entity_type(entity_type_id="transaction")
print(transactions_entity_type)

transactions_feature_configs = {
    "amount": {
        "value_type": "DOUBLE",
        "description": "Amount of transaction",
    },
    "oldbalance_orig": {
        "value_type": "DOUBLE",
        "description": "Old balance origin",
    },
    "newbalance_orig": {
        "value_type": "DOUBLE",
        "description": "New balance origin",
    },
    "oldbalance_dest": {
        "value_type": "DOUBLE",
        "description": "Old balance destination",
    },
    "newbalance_dest": {
        "value_type": "DOUBLE",
        "description": "New balance destination",
    }
}

transaction_features = transactions_entity_type.batch_create_features(
    feature_configs=transactions_feature_configs,
)

# Comment this, since it runs too quickly and features ar not ready in some regions
#my_features = Feature.search(query="featurestore_id={}".format(FEATURESTORE_ID))
#print(my_features)

#double_features = Feature.search(
#    query="value_type=DOUBLE AND featurestore_id={}".format(FEATURESTORE_ID)
#)
#print(double_features[1].gca_resource)


# Batch ingest (import data from GCS)
# Transaction entity. Expect o(10 minutes) if the number of rows is large.
print("Batch ingestion...")
TRANSACTIONS_FEATURES_IDS = [feature.name for feature in transactions_entity_type.list_features()]
print(TRANSACTIONS_FEATURES_IDS)

FEATURE_SOURCE_FIELDS = {
    'oldbalance_orig': 'oldbalanceOrg',
    'newbalance_orig': 'newbalanceOrig',
    'oldbalance_dest': 'oldbalanceDest',
    'newbalance_dest': 'newbalanceDest',
}

TRANSACTIONS_ENTITY_ID_FIELD = "nameOrig" # <--- NOTICE ENTITY_ID. MUST BE STRING TYPE AND MUST EXIST IN THE TABLE
TRANSACTIONS_GCS_SOURCE_URI = "gs://argolis-vertex-europewest4/fraud_data_kaggle_5000.csv" 
GCS_SOURCE_TYPE = "csv" # could be avro
WORKER_COUNT = 1

transactions_entity_type.ingest_from_gcs(
    feature_ids=TRANSACTIONS_FEATURES_IDS,
    feature_time=datetime(2021, 1, 1, 9, 30),  # unique timestamp for all
    entity_id_field=TRANSACTIONS_ENTITY_ID_FIELD,
    feature_source_fields=FEATURE_SOURCE_FIELDS,
    gcs_source_uris=TRANSACTIONS_GCS_SOURCE_URI,
    gcs_source_type=GCS_SOURCE_TYPE,
    worker_count=WORKER_COUNT,
    sync=True
)


# Online serving
print("Online serving...")
print(transactions_entity_type.read(entity_ids=["C1231006815", "C840083671"]))