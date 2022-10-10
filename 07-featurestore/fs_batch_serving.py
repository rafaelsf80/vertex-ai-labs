from datetime import datetime
from google.cloud import bigquery
from google.cloud.aiplatform import Featurestore


PROJECT_ID = "argolis-rafaelsanchez-ml-dev"  
LOCATION = "us-central1" 
FEATURESTORE_ID = "fraud_detection_demo_monitoring_18"
INPUT_CSV_FILE = "gs://argolis-vertex-uscentral1/read_entity_instance.csv"  # ground-truth data in GCS. BatchReadFeatureValues API takes this table as input and returns a complete table for training.


# Output dataset
DESTINATION_DATA_SET = "OUTPUT_DATASET1"  
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
DESTINATION_DATA_SET = "{prefix}_{timestamp}".format(
    prefix=DESTINATION_DATA_SET, timestamp=TIMESTAMP
)

# Output table. Make sure that the table does NOT already exist; the BatchReadFeatureValues API cannot overwrite an existing table
DESTINATION_TABLE_NAME = "batch_serving"  

DESTINATION_PATTERN = "bq://{project}.{dataset}.{table}"
DESTINATION_TABLE_URI = DESTINATION_PATTERN.format(
    project=PROJECT_ID, dataset=DESTINATION_DATA_SET, table=DESTINATION_TABLE_NAME
)

# Create dataset
client = bigquery.Client(project=PROJECT_ID)
dataset_id = "{}.{}".format(client.project, DESTINATION_DATA_SET)
dataset = bigquery.Dataset(dataset_id)
dataset.location = LOCATION
dataset = client.create_dataset(dataset)
print("Created dataset {}.{}".format(client.project, dataset.dataset_id))



fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=LOCATION,
)
print(fs.gca_resource)


SERVING_FEATURE_IDS = {
    # to choose all the features use 'entity_type_id: ['*']'
    'transaction': ['oldbalance_orig', 'newbalance_orig', 'oldbalance_dest', 'newbalance_dest', 'amount']
}

print(f'Running batch serving...')
fs.batch_serve_to_bq(
    bq_destination_output_uri=DESTINATION_TABLE_URI,
    serving_feature_ids=SERVING_FEATURE_IDS,
    read_instances_uri=INPUT_CSV_FILE,
)


print(f'See results at: {DESTINATION_TABLE_URI}')