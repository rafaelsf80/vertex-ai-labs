import kfp
from kfp.v2 import compiler

from google_cloud_pipeline_components import aiplatform as gcc_aip

PROJECT_ID = 'windy-site-254307'
MY_STAGING_BUCKET = 'caip-prediction-custom-uscentral1'
LOCATION = 'us-central1'
USER = 'rafaelsanchez'
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(MY_STAGING_BUCKET, USER)
CONTAINER_AR_URI = 'us-central1-docker.pkg.dev/windy-site-254307/ml-pipelines-repo-us/vertex-custom-container:b2ce020-dirty'
# If Docker permission error or Docker image could not be pulled error, just run "gcloud auth configure-docker us-central1-docker.pkg.dev,europe-west4-docker.pkg.dev"
# Set also project proprtly with gcloud config set project BEFORE gcloud auth
gcc_aip.utils.DEFAULT_CONTAINER_IMAGE=CONTAINER_AR_URI

@kfp.dsl.pipeline(name='fraud-detection-demo-gccaip-uscentral1')
def pipeline():
  dataset_create_op = gcc_aip.TabularDatasetCreateOp(
      project=PROJECT_ID, 
      location=LOCATION,
      display_name='fraud-detection-demo-gccaip',
      bq_source=f'bq://bigquery-public-data.ml_datasets.ulb_fraud_detection')

  training_op = gcc_aip.AutoMLTabularTrainingJobRunOp(
      project=PROJECT_ID,
      location=LOCATION,
      display_name='fraud-detection-demo-gccaip',
      optimization_prediction_type='classification',
      optimization_objective='maximize-au-prc',    
      column_transformations=[
          {"numeric": {"column_name": "Amount"}},
          {"numeric": {"column_name": "V1"}},
          {"numeric": {"column_name": "V2"}},
          {"numeric": {"column_name": "V3"}},
          {"numeric": {"column_name": "V4"}},
          {"numeric": {"column_name": "V5"}},
          {"numeric": {"column_name": "V6"}},
          {"numeric": {"column_name": "V7"}},
          {"numeric": {"column_name": "V8"}},
          {"numeric": {"column_name": "V9"}},
          {"numeric": {"column_name": "V10"}},
          {"numeric": {"column_name": "V11"}},
          {"numeric": {"column_name": "V12"}},
          {"numeric": {"column_name": "V13"}},
          {"numeric": {"column_name": "V14"}},
          {"numeric": {"column_name": "V15"}},
          {"numeric": {"column_name": "V16"}},
          {"numeric": {"column_name": "V17"}},
          {"numeric": {"column_name": "V18"}},
          {"numeric": {"column_name": "V19"}},
          {"numeric": {"column_name": "V20"}},
          {"numeric": {"column_name": "V21"}},
          {"numeric": {"column_name": "V22"}},
          {"numeric": {"column_name": "V23"}},
          {"numeric": {"column_name": "V24"}},
          {"numeric": {"column_name": "V25"}},
          {"numeric": {"column_name": "V26"}},
          {"numeric": {"column_name": "V27"}},
          {"numeric": {"column_name": "V28"}},
      ],
      dataset = dataset_create_op.outputs['dataset'],
      target_column = "Class"
  )

  deploy_op = gcc_aip.ModelDeployOp(
      model=training_op.outputs['model'],
      project=PROJECT_ID,
      location=LOCATION,
      machine_type='n1-standard-4')


# Compile and run the pipeline
compiler.Compiler().compile(pipeline_func=pipeline, 
        package_path='fraud_detection_demo_gccaip_uscentral1.json')
 
PIPELINE_ROOT='gs://caip-pipelines-xgb-demo-fraud-detection-uscentral1'

from google.cloud.aiplatform import pipeline_jobs
pipeline_jobs.PipelineJob(
    display_name='fraud-detection-demo-gccaip-uscentral1',
    template_path='fraud_detection_demo_gccaip_uscentral1.json',
    pipeline_root=PIPELINE_ROOT,
    enable_caching=True
).run(sync=False)