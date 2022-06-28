from kfp.v2 import components
from kfp.v2 import dsl
from kfp.v2 import compiler

# This pipeline uses a public dataset at:
# gs://financial_fraud_detection/fraud_data_kaggle.csv

generate_op = components.load_component_from_text("""
name: GenerateData
inputs:
- {name: project_id,          type: String, default: 'windy-site-254307', description: 'GCP project_id'}
- {name: temp_bucket,         type: String, default: 'caip-pipelines-xgb-demo-fraud-detection-uscentral1', description: 'Temporal bucket in GCS'}
- {name: gcs_dataset_filename,type: String, default: 'gs://financial_fraud_detection/fraud_data_kaggle.csv', description: 'dataset'}
- {name: model_output_bucket, type: String, default: 'caip-pipelines-xgb-demo-fraud-detection-uscentral1', description: 'Model output bucket in GCS'}
- {name: experiment_name,     type: String, default: 'exp01', description: 'Experiment name'}
outputs:
- {name: X_train, type: Dataset}
- {name: X_test, type: Dataset}
- {name: y_train, type: Dataset}
- {name: y_test, type: Dataset}
implementation:
  container:
    image: gcr.io/windy-site-254307/xgboost-fraud-detection-generatedata:fcee0b6-dirty
    command:
    - python3
    - /app/generatedata_script.py
    args:
    - --project_id
    - {inputValue: project_id}
    - --temp_bucket
    - {inputValue: temp_bucket}
    - --gcs_dataset_filename
    - {inputValue: gcs_dataset_filename}
    - --model_output_bucket
    - {inputValue: model_output_bucket}
    - --experiment_name
    - {inputValue: experiment_name}
    - --x_train_uri
    - {outputUri: X_train}
    - --x_test_uri
    - {outputUri: X_test}
    - --y_train_uri
    - {outputUri: y_train}
    - --y_test_uri
    - {outputUri: y_test}
""")

train_op = components.load_component_from_text("""
name: Train
inputs:
- {name: project_id,                  type: String,  default: 'windy-site-254307', description: 'GCP project_id'}
- {name: experiment_name,             type: String,  default: 'exp01', description: 'Experiment name'}
- {name: xgboost_param_max_depth,     type: Integer, default: '10', description: 'hyperparameter'}
- {name: xgboost_param_learning_rate, type: Float,   default: '0.2', description: 'hyperparameter'}
- {name: xgboost_param_n_estimators,  type: Integer, default: '200', description: 'hyperparameter'}
- {name: x_train, type: Dataset}
- {name: x_test, type: Dataset}
- {name: y_train, type: Dataset}
- {name: y_test, type: Dataset}
outputs:
- {name: model, type: Model}
- {name: metrics, type: Metrics}
implementation:
  container:
    image: gcr.io/windy-site-254307/xgboost-fraud-detection-trainer:fcee0b6-dirty
    command:
    - python3
    - /app/trainer_script.py
    args:
    - --project_id
    - {inputValue: project_id} 
    - --experiment_name
    - {inputValue: experiment_name}
    - --xgboost_param_max_depth
    - {inputValue: xgboost_param_max_depth}   
    - --xgboost_param_learning_rate
    - {inputValue: xgboost_param_learning_rate} 
    - --xgboost_param_n_estimators
    - {inputValue: xgboost_param_n_estimators}                    
    - --x_train_uri
    - {inputUri: x_train}
    - --x_test_uri
    - {inputUri: x_test}
    - --y_train_uri
    - {inputUri: y_train}
    - --y_test_uri
    - {inputUri: y_test}
    - --output_model_uri
    - {outputUri: model}
    - --output_metrics_uri
    - {outputUri: metrics}
""")

@dsl.pipeline(name='fraud-detection-demo-custom-uscentral1')
def pipeline():
  generate = generate_op()
  train = (train_op(x_train=generate.outputs['X_train'],
      x_test=generate.outputs['X_test'],
      y_train=generate.outputs['y_train'],
      y_test=generate.outputs['y_test']).
    set_cpu_limit('4').
    set_memory_limit('14Gi').
    add_node_selector_constraint(
      'cloud.google.com/gke-accelerator',
      'nvidia-tesla-k80').
    set_gpu_limit(1))

# Compile and run the pipeline
compiler.Compiler().compile(pipeline_func=pipeline, package_path='fraud_detection_demo_custom_uscentral1.json')
 
PIPELINE_ROOT='gs://caip-pipelines-xgb-demo-fraud-detection-uscentral1'

from google.cloud.aiplatform import pipeline_jobs
pipeline_jobs.PipelineJob(
    display_name='fraud-detection-demo-custom-uscentral1',
    template_path='fraud_detection_demo_custom_uscentral1.json',
    pipeline_root=PIPELINE_ROOT,
    enable_caching=True
).run(sync=False)


# IMPORTANT: create_schedule_from_job_spec will be DEPRECATED
# api_client.create_schedule_from_job_spec(
#     job_spec_path='fraud_detection_demo_custom_uscentral1.json',
#     schedule="* * * * *"
# )