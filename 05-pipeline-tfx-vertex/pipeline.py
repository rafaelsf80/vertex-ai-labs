""" TFX pipeline that uses TFX components to train and deploy in Vertex.It requires the following versions as a minimum:
TensorFlow version: 2.6.0
TFX version: 1.3.0
KFP version: 1.8.1
You need to upload these two files to GCS:
1. Dataset gs://download.tensorflow.org/data/palmer_penguins/penguins_processed.csv
2. Trainer file penguin_trainer.py
"""


import tensorflow as tf
from tfx import v1 as tfx
import kfp
import os

PROJECT_ID = 'windy-site-254307'   
REGION = 'us-central1'   
GCS_BUCKET_NAME = 'caip-prediction-custom-uscentral1'        
PIPELINE_NAME = 'penguin-vertex-pusher'

_trainer_module_file = 'penguin_trainer.py'

PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for users' Python module.
MODULE_ROOT = 'gs://{}/pipeline_module/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)

# Paths for input data.
DATA_ROOT = 'gs://{}/data/{}'.format(GCS_BUCKET_NAME, PIPELINE_NAME)


# Copied from https://www.tensorflow.org/tfx/tutorials/tfx/penguin_simple and
# slightly modified because we don't need `metadata_path` argument.

def _create_pipeline(pipeline_name: str, pipeline_root: str, data_root: str,
                     module_file: str, project_id: str, region: str
                     ) -> tfx.dsl.Pipeline:
  """Creates a three component penguin pipeline with TFX."""
  # Brings data into the pipeline.
  example_gen = tfx.components.CsvExampleGen(input_base=data_root)

  # Uses user-provided Python function that trains a model.
  trainer = tfx.components.Trainer(
      module_file=module_file,
      examples=example_gen.outputs['examples'],
      train_args=tfx.proto.TrainArgs(num_steps=100),
      eval_args=tfx.proto.EvalArgs(num_steps=5))

  # NEW: Configuration for pusher.
  vertex_serving_spec = {
      'project_id': project_id,
      'endpoint_name': 'my_endpoint_20211008',
      'model_name': 'my_model_20211008',
  }

  # Pushes the model to a filesystem destination.
  pusher = tfx.extensions.google_cloud_ai_platform.Pusher(
      model=trainer.outputs['model'],
      custom_config={
          tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY:
              True,
          tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY:
              region,
          tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY:
              'tensorflow/serving',
          tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY:
            vertex_serving_spec,
      })

  # Following three components will be included in the pipeline.
  components = [
      example_gen,
      trainer,
      pusher,
  ]

  return tfx.dsl.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=components)



print('TensorFlow version: {}'.format(tf.__version__))
print('TFX version: {}'.format(tfx.__version__))
print('KFP version: {}'.format(kfp.__version__))

PIPELINE_DEFINITION_FILE = PIPELINE_NAME + '_pipeline.json'

runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
    config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(),
    output_filename=PIPELINE_DEFINITION_FILE)
# Following function will write the pipeline definition to PIPELINE_DEFINITION_FILE.
_ = runner.run(
    _create_pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        data_root=DATA_ROOT,
        module_file=os.path.join(MODULE_ROOT, _trainer_module_file),
        project_id=GOOGLE_CLOUD_PROJECT,
        region=GOOGLE_CLOUD_REGION))

from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs

aiplatform.init(project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_REGION)

job = pipeline_jobs.PipelineJob(display_name="tfx-vertex", template_path=PIPELINE_DEFINITION_FILE)
job.run(sync=False)
