from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-me-central1'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'me-central1'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

DEPLOY_IMAGE = 'me-central1-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-workloads-qatar-prediction/13-xgboost_dask' 
HEALTH_ROUTE = "/health"
PREDICT_ROUTE = "/predict"
SERVING_CONTAINER_PORTS = [7080]

model = aiplatform.Model.upload(
    display_name=f'custom-xgboost-model-uvicorn',    
    description=f'XGBoost model with Dask',
    serving_container_image_uri=DEPLOY_IMAGE,
    serving_container_predict_route=PREDICT_ROUTE,
    serving_container_health_route=HEALTH_ROUTE,
    serving_container_ports=SERVING_CONTAINER_PORTS,
)

print(model.resource_name)

# Retrieve a Vertex Model
#model = aiplatform.Model('projects/989788194604/locations/me-central1/models/6042898314070851584@1')
model = aiplatform.Model(model.resource_name)


# Deploy model. THIS MAY TAKE 20-30 MINUTES
endpoint = model.deploy(
      machine_type='n2-standard-4', 
      sync=False
)
endpoint.wait()

# Retrieve a Vertex Endpoint
#endpoint = aiplatform.Endpoint('projects/989788194604/locations/me-central1/endpoints/1374028894906089472')
instances = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2]
]

response = endpoint.predict(instances=instances)
print(f"Predictions: {response.predictions}")

# Endpoint model deployed. Resource name: projects/989788194604/locations/me-central1/endpoints/1374028894906089472
# Predictions: [-0.3882404267787933, -0.3882404267787933]