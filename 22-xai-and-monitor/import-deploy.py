""" Step 1: Import and deploy a model that will be used for Model Monitoring """

from google.cloud import aiplatform

PROJECT_ID = "argolis-rafaelsanchez-ml-dev"    # <---- CHANGE THIS
REGION = "europe-west4"                        # <---- CHANGE THIS
MODEL_NAME = "churn"
IMAGE = "us-docker.pkg.dev/cloud-aiplatform/prediction/tf2-cpu.2-5:latest"
ARTIFACT = "gs://mco-mm/churn"


aiplatform.init(project=PROJECT_ID, location=REGION)
model = aiplatform.Model.upload(
       display_name=f'churn',
       artifact_uri=ARTIFACT,
       serving_container_image_uri=IMAGE
)

print(model.resource_name)

# Retrieve a Model on Vertex
#model_resource_name ='projects/32667170988/locations/europe-west4/models/5995874400774127616'
model = aiplatform.Model(model.resource_name)
print(model.resource_name)


# Deploy model
endpoint = model.deploy(
     machine_type='n1-standard-4', 
     sync=False
)
endpoint.wait()