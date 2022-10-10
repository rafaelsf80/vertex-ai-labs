from google.cloud.aiplatform import Featurestore

PROJECT_ID = "argolis-rafaelsanchez-ml-dev"  
LOCATION = "us-central1" 
FEATURESTORE_ID = "fraud_detection_demo_monitoring"

print("********** Remove Feature Store ", FEATURESTORE_ID)

fs = Featurestore(
    featurestore_name=FEATURESTORE_ID,
    project=PROJECT_ID,
    location=LOCATION,
)
print(fs.gca_resource)

fs.delete(sync=True, force=True)