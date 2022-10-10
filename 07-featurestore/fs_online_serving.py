from google.cloud.aiplatform import Featurestore

PROJECT_ID = "argolis-rafaelsanchez-ml-dev"  
LOCATION = "us-central1" 
FEATURESTORE_ID = "fraud_detection_demo_monitoring_old4"

print("********** Online Serving Feature Store ", FEATURESTORE_ID)

fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=LOCATION,
)
print(fs.gca_resource)

transactions_entity_type = fs.get_entity_type(entity_type_id="transaction")

print("Online serving...")
print(transactions_entity_type.read(entity_ids=["C1231006815", "C840083671"]))