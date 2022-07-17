""" TFX pipeline that uses TFX components to train and deploy in Vertex.It requires the following versions as a minimum:
TensorFlow version: 2.6.0
TFX version: 1.3.0
KFP version: 1.8.1
You need to upload these two files to GCS:
1. Dataset gs://download.tensorflow.org/data/palmer_penguins/penguins_processed.csv
2. Trainer file penguin_trainer.py
"""
import tensorflow as tf
print('TensorFlow version: {}'.format(tf.__version__))
from tfx import v1 as tfx
print('TFX version: {}'.format(tfx.__version__))
import kfp
print('KFP version: {}'.format(kfp.__version__))

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'   
REGION = 'europe-west4'   
GCS_BUCKET_NAME = 'argolis-vertex-europewest4'        
PIPELINE_NAME = 'penguin-vertex-pusher'

_trainer_module_file = 'penguin_trainer.py'

# Path to various pipeline artifact.
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' data.
DATA_ROOT = 'gs://{}/data/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Name of Vertex AI Endpoint.
ENDPOINT_NAME = 'prediction-' + PIPELINE_NAME



def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, endpoint_name: str, project_id: str,
                     region: str, use_gpu: bool) -> tfx.dsl.Pipeline:
  """Implements the penguin pipeline with TFX."""
  # Brings data into the pipeline or otherwise joins/converts training data.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # NEW: Configuration for Vertex AI Training.
  # This dictionary will be passed as `CustomJobSpec`.
  vertex_job_spec = {
      'project': project_id,
      'worker_pool_specs': [{
          'machine_spec': {
              'machine_type': 'n1-standard-4',
          },
          'replica_count': 1,
          'container_spec': {
              'image_uri': 'gcr.io/tfx-oss-public/tfx:{}'.format(tfx.__version__),
          },
      }],
  }
  if use_gpu:
    # See https://cloud.google.com/vertex-ai/docs/reference/rest/v1/MachineSpec#acceleratortype
    # for available machine types.
    vertex_job_spec['worker_pool_specs'][0]['machine_spec'].update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })

  # Trains a model using Vertex AI Training.
  # NEW: We need to specify a Trainer for GCP with related configs.
  trainer = tfx.extensions.google_cloud_ai_platform.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5),
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY:
              vertex_job_spec,
          'use_gpu':
              use_gpu,
      })

  # NEW: Configuration for pusher.
  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': endpoint_name,
      'model_name': 'try_pusher',  # '-' is not allowed.

      # Remaining argument is passed to aiplatform.Model.deploy()
      # See https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api#deploy_the_model
      # for the detail.
      #
      # Machine type is the compute resource to serve prediction requests.
      # See https://cloud.google.com/vertex-ai/docs/predictions/configure-compute#machine-types
      # for available machine types and acccerators.
      'machine_type': 'n1-standard-4',
  }

  # Vertex AI provides pre-built containers with various configurations for
  # serving.
  # See https://cloud.google.com/vertex-ai/docs/predictions/pre-built-containers
  # for available container images.
  serving_image = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-6:latest'
  if use_gpu:
    vertex_serving_spec.update({
        'accelerator_type': 'NVIDIA_TESLA_K80',
        'accelerator_count': 1
    })
    serving_image = 'europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest'

  # NEW: Pushes the model to Vertex AI.
  pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
              serving_image,
          tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
            vertex_serving_spec,
      })

  components = [
      example_gen,
      trainer,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components)


import os

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)
_ = runner.run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        module_file=os.path.join(MODULE_ROOT, _trainer_module_file),
        endpoint_name=ENDPOINT_NAME,
        project_id=PROJECT_ID,
        region=REGION,
        # We will use CPUs only for now.
        use_gpu=False))


from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
import logging
logging.getLogger().setLevel(logging.INFO)

aiplatform.init(project=PROJECT_ID, location=REGION)

job = pipeline_jobs.PipelineJob(template_path=PIPELINE_DEFINITION_FILE,
                                display_name=PIPELINE_NAME)
job.submit()

