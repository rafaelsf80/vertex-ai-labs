from google.cloud import aiplatform

STAGING_BUCKET = 'gs://argolis-vertex-me-central1'
PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'me-central1'

aiplatform.init(project=PROJECT_ID, staging_bucket=STAGING_BUCKET, location=LOCATION)

endpoint = aiplatform.Endpoint('projects/989788194604/locations/me-central1/endpoints/1374028894906089472')
instances = [
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2]
]

response = endpoint.predict(instances=instances)
print(f"Predictions: {response.predictions}")

# Predictions: [-0.3882404267787933, -0.3882404267787933]