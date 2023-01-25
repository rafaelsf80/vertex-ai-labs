from datetime import datetime
from google.cloud import aiplatform
from google.cloud.aiplatform import Featurestore

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
REGION = 'europe-west4'
FEATURESTORE_ID = "ulb"
ONLINE_STORE_FIXED_NODE_COUNT = 1

aiplatform.init(project=PROJECT_ID, location=REGION)

# create featurestore
fs = Featurestore.create(
    featurestore_id=FEATURESTORE_ID,
    online_store_fixed_node_count=ONLINE_STORE_FIXED_NODE_COUNT,
    project=PROJECT_ID,
    location=REGION,
    sync=True,
)

fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=REGION,
)
print(fs.gca_resource)

# Create users entity type
transactions_entity_type = fs.create_entity_type(
    entity_type_id="transactions",
    description="Transactions entity",
)

transactions_entity_type = fs.get_entity_type(entity_type_id="transactions")
print(transactions_entity_type)
fs.list_entity_types()

ulb_feature_configs = {
    "amount": {
        "value_type": "DOUBLE",
        "description": "Amount feature",
    },
    # "time": {
    #     "value_type": "TIMESTAMP",
    #     "description": "Time feature",
    # },
    # "id": {
    #     "value_type": "STRING",
    #     "description": "ID feature",
    # },
    "class": {
        "value_type": "INT64",
        "description": "Class feature",
    },
    "v1": {
        "value_type": "DOUBLE",
        "description": "V1 feature",
    },
    "v2": {
        "value_type": "DOUBLE",
        "description": "V2 feature",
    },
    "v3": {
        "value_type": "DOUBLE",
        "description": "V3 feature",
    },
    "v4": {
        "value_type": "DOUBLE",
        "description": "V4 feature",
    },
    "v5": {
        "value_type": "DOUBLE",
        "description": "V5 feature",
    },
    "v6": {
        "value_type": "DOUBLE",
        "description": "V6 feature",
    },
    "v7": {
        "value_type": "DOUBLE",
        "description": "V7 feature",
    },
    "v8": {
        "value_type": "DOUBLE",
        "description": "V8 feature",
    },
    "v9": {
        "value_type": "DOUBLE",
        "description": "V9 feature",
    },
    "v10": {
        "value_type": "DOUBLE",
        "description": "V10 feature",
    },
    "v11": {
        "value_type": "DOUBLE",
        "description": "V11 feature",
    },
    "v12": {
        "value_type": "DOUBLE",
        "description": "V12 feature",
    },
    "v13": {
        "value_type": "DOUBLE",
        "description": "V13 feature",
    },
    "v14": {
        "value_type": "DOUBLE",
        "description": "V14 feature",
    },
    "v15": {
        "value_type": "DOUBLE",
        "description": "V15 feature",
    },
    "v16": {
        "value_type": "DOUBLE",
        "description": "V16 feature",
    },
    "v17": {
        "value_type": "DOUBLE",
        "description": "V17 feature",
    },
    "v18": {
        "value_type": "DOUBLE",
        "description": "V18 feature",
    },
    "v19": {
        "value_type": "DOUBLE",
        "description": "V19 feature",
    },
    "v20": {
        "value_type": "DOUBLE",
        "description": "V20 feature",
    },
    "v21": {
        "value_type": "DOUBLE",
        "description": "V21 feature",
    },
    "v22": {
        "value_type": "DOUBLE",
        "description": "V22 feature",
    },
    "v23": {
        "value_type": "DOUBLE",
        "description": "V23 feature",
    },
    "v24": {
        "value_type": "DOUBLE",
        "description": "V24 feature",
    },
    "v25": {
        "value_type": "DOUBLE",
        "description": "V25 feature",
    },
    "v26": {
        "value_type": "DOUBLE",
        "description": "V26 feature",
    },
    "v27": {
        "value_type": "DOUBLE",
        "description": "V27 feature",
    },
    "v28": {
        "value_type": "DOUBLE",
        "description": "V28 feature",
    },
}

ulb_features = transactions_entity_type.batch_create_features(
    feature_configs=ulb_feature_configs,
)


# Import data
TRANSACTIONS_FEATURES_IDS = [feature.name for feature in transactions_entity_type.list_features()]
TRANSACTIONS_FEATURE_TIME = "time"
TRANSACTIONS_ENTITY_ID_FIELD = "id"
BQ_SOURCE = 'bq://argolis-rafaelsanchez-ml-dev.ml_datasets_europewest4.ulb_1'
WORKER_COUNT = 1
print(TRANSACTIONS_FEATURES_IDS)

transactions_entity_type.ingest_from_bq(
    feature_ids=TRANSACTIONS_FEATURES_IDS,
    feature_time=TRANSACTIONS_FEATURE_TIME,#datetime(2021, 1, 1, 9, 30),
    entity_id_field=TRANSACTIONS_ENTITY_ID_FIELD,
    bq_source_uri=BQ_SOURCE,
    worker_count=WORKER_COUNT,
    sync=True,
)

## Online serving: Reading some entries
print(transactions_entity_type.read(entity_ids=["f6aaa0a4_d66e_4981_afb6_2d22bd53e664", "20e3a40d_b7e6_446b_a33c_df734ee4dab9"]))

print(transactions_entity_type.read(entity_ids=["f6aaa0a4_d66e_4981_afb6_2d22bd53e664", "20e3a40d_b7e6_446b_a33c_df734ee4dab9"], feature_ids=["v1", "v2"]))