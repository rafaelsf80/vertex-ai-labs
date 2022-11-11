import kfp
from kfp.v2 import compiler
from kfp.v2.dsl import pipeline

# This pipeline uses a public dataset at:
# gs://financial_fraud_detection/fraud_data_kaggle.csv

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
LOCATION = 'us-central1'
MY_STAGING_BUCKET = 'argolis-vertex-uscentral'
PIPELINE_ROOT = 'gs://argolis-vertex-uscentral1'
SOURCE_DATA = 'gs://financial_fraud_detection/fraud_data_kaggle.csv'
EXPERIMENT_NAME = 'exp01'
SERVING_CONTAINER_IMAGE = 'europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest'

@pipeline(name='01-fraud-detection-demo-custom-uscentral1')
def pipeline():
  generate_op = kfp.components.load_component_from_text("""
    name: GenerateData
    inputs:
    - {name: project_id,          type: String, default: 'argolis-rafaelsanchez-ml-dev', description: 'GCP project_id'}
    - {name: temp_bucket,         type: String, default: 'argolis-vertex-uscentral1', description: 'Temporal bucket in GCS'}
    - {name: gcs_dataset_filename,type: String, default: 'gs://financial_fraud_detection/fraud_data_kaggle.csv', description: 'dataset'}
    - {name: model_output_bucket, type: String, default: 'argolis-vertex-uscentral1', description: 'Model output bucket in GCS'}
    - {name: experiment_name,     type: String, default: 'exp01', description: 'Experiment name'}
    outputs:
    - {name: x_train_artifact, type: Dataset}
    - {name: x_test_artifact, type: Dataset}
    - {name: y_train_artifact, type: Dataset}
    - {name: y_test_artifact, type: Dataset}
    implementation:
      container:
        image: europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/xgboost-fraud-detection-generatedata:bcb8403
        command: [python, /app/generatedata_script.py]
        args: [
              --executor_input, {executorInput: null},
              --function_to_execute, main
            ]
    """)

  train_op = kfp.components.load_component_from_text("""
    name: Train
    inputs:
    - {name: xgboost_param_max_depth,     type: Integer, default: 10, description: 'hyperparameter'}
    - {name: xgboost_param_learning_rate, type: Float,   default: 0.2, description: 'hyperparameter'}
    - {name: xgboost_param_n_estimators,  type: Integer, default: 200, description: 'hyperparameter'}
    - {name: x_train_artifact, type: Dataset}
    - {name: x_test_artifact, type: Dataset}
    - {name: y_train_artifact, type: Dataset}
    - {name: y_test_artifact, type: Dataset}
    outputs:
    - {name: model, type: Model}
    - {name: metrics, type: Metrics}
    - {name: metricsc, type: ClassificationMetrics}
    implementation:
      container:
        image: europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/xgboost-fraud-detection-trainer:bcb8403-dirty
        command: [python, /app/trainer_script.py]
        args: [
              --executor_input, {executorInput: null},
              --function_to_execute, main
            ]
    """)

  deploy_op = kfp.components.load_component_from_text("""
    name: Deploy
    inputs:
    - {name: project_id,              type: String, default: 'argolis-rafaelsanchez-ml-dev', description: 'GCP project_id'}
    - {name: location,                type: String, default: 'us-central1', description: 'GCP region'}
    - {name: serving_container_image, type: String, default: 'europe-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-1:latest', description: 'Serving Container Image URI'}
    - {name: model, type: Model}
    outputs:
    - {name: endpoint, type: Artifact}
    implementation:
      container:
        image: europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/xgboost-fraud-detection-deploy:bcb8403-dirty
        command: [python, /app/deploy_script.py]
        args: [
              --executor_input, {executorInput: null},
              --function_to_execute, main
            ]
    """)

  generate = generate_op(
      project_id=PROJECT_ID, 
      temp_bucket=MY_STAGING_BUCKET, 
      gcs_dataset_filename=SOURCE_DATA, 
      model_output_bucket=MY_STAGING_BUCKET, 
      experiment_name=EXPERIMENT_NAME)

  train = (train_op(
      xgboost_param_max_depth=10, 
      xgboost_param_learning_rate = 0.2, 
      xgboost_param_n_estimators = 200,
      x_train_artifact=generate.outputs['x_train_artifact'],
      x_test_artifact=generate.outputs['x_test_artifact'],
      y_train_artifact=generate.outputs['y_train_artifact'],
      y_test_artifact=generate.outputs['y_test_artifact']).
    set_cpu_limit('4').
    set_memory_limit('14Gi').
    add_node_selector_constraint(
      'cloud.google.com/gke-accelerator',
      'nvidia-tesla-k80').
    set_gpu_limit(1))

  deploy = (deploy_op(
      project_id=PROJECT_ID, 
      location=LOCATION,
      serving_container_image = SERVING_CONTAINER_IMAGE,
      model=train.outputs['model']))



# Compile and run the pipeline
compiler.Compiler().compile(pipeline_func=pipeline, package_path='01-fraud_detection_demo_custom_uscentral1.json')
 
from google.cloud import aiplatform
from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

aiplatform.init(project=PROJECT_ID, location=LOCATION)

aiplatform.PipelineJob(
    display_name='01-fraud-detection-demo-custom-uscentral1',
    template_path='01-fraud_detection_demo_custom_uscentral1.json',
    job_id="fraud-detection-demo-custom-uscentral1-{0}".format(TIMESTAMP),
    pipeline_root=PIPELINE_ROOT,
    enable_caching=False
).submit()