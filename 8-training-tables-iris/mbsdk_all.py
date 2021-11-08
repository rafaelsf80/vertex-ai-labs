from google.cloud import aiplatform

MY_PROJECT = 'windy-site-254307'
MY_STAGING_BUCKET = 'gs://caip-vertex-europewest4'
LOCATION = 'europe-west4'

aiplatform.init(project=MY_PROJECT, staging_bucket=MY_STAGING_BUCKET, location=LOCATION)

# BigQuery dataset must be in same location as AI Platform project
ds = aiplatform.TabularDataset.create(
    display_name='bq_iris_dataset',
    bq_source=f'bq://{MY_PROJECT}.ml_datasets_europewest4.iris')

job = aiplatform.CustomContainerTrainingJob(
    display_name='train-bq-iris',
    container_uri=f'europe-west4-docker.pkg.dev/{MY_PROJECT}/ml-pipelines-repo/vertex-iris-demo:latest',
    model_serving_container_image_uri='europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-2:latest',)

model = job.run(ds,
    replica_count=1, 
    machine_type="n1-standard-4",
    accelerator_type= "NVIDIA_TESLA_T4",
    accelerator_count = 1,
    model_display_name='bq-iris-model',
    bigquery_destination=f'bq://{MY_PROJECT}')
print(model.resource_name)

# model_name = 'projects/655797269815/locations/europe-west4/models/7067766296460394496'
endpoint = model.deploy(machine_type='n1-standard-4', 
    accelerator_type= "NVIDIA_TESLA_T4",
    accelerator_count = 1)
print(endpoint.resource_name)

#endpoint_name = 'projects/655797269815/locations/europe-west4/endpoints/7236334623137988608'
endpoint = aiplatform.Endpoint(endpoint.name)
print(endpoint.predict([{'sepal_length':5.1, 'sepal_width':2.5, 'petal_length':3.0,
                   'petal_width':1.1}]))

# JSON request for UI predictions
#{
#      "instances": [
#          {"sepal_length":5.1, "sepal_width":2.5, "petal_length":3.0,
#                   "petal_width":1.1}
#      ]
#}
