from kfp import dsl
from kfp.v2.dsl import (
    component,
    InputPath,
    OutputPath,
    Input,
    Output,
  #  InputArtifact,
  #  OutputArtifact,
    Artifact,
    Dataset,
    Model,
    ClassificationMetrics,
    Metrics,
)

PROJECT_ID = 'windy-site-254307'
MY_STAGING_BUCKET = 'caip-prediction-custom-uscentral1'
LOCATION = 'us-central1'
USER='rafaelsanchez'
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(MY_STAGING_BUCKET, USER)
BIGQUERY_URI = 'bq://windy-site-254307.public.ulb_'

FEATURES = ["Amount", "V1", "V2","V3", "V4",
          "V5", "V6", "V7", "V8", "V9", "V10", "V11",
          "V12", "V13", "V14", "V15", "V16", "V17",
          "V18", "V19", "V20", "V21", "V22", "V23",
          "V24", "V25", "V26", "V27", "V28"]

TARGET = 'Class'

##############
# Preprocess component
##############
@component(
    base_image='python:3.9', # Use a different base image.
    packages_to_install=['tensorflow', 'tensorflow_io']
)
def preprocess(
    # An input parameter of type string.
    bigquery_uri: str, # eg 'bq://project-demos.london_bikes_weather.bikes_weather',
    target_input: str,
    features_input: str,
    # Use OutputArtifact to get a metadata-rich handle to the output artifact of type `Dataset`.
    output_data_path: OutputPath("Dataset")):

    #output_dataset: OutputArtifact(Dataset)):

  import tensorflow as tf
  from tensorflow_io.bigquery import BigQueryClient
  from tensorflow.python.framework import dtypes

  import logging
  logging.getLogger().setLevel(logging.INFO)

  TARGET = 'Class'

  # Converts string to list
  features = list(features_input.split(","))
  target = target_input

  def caip_uri_to_fields(uri):
    uri = uri[5:]
    project, dataset, table = uri.split('.')
    return project, dataset, table

  def transform_row(row_dict):
    # Trim all string tensors
    trimmed_dict = { column:
                    (tf.strings.strip(tensor) if tensor.dtype == 'string' else tensor) 
                    for (column,tensor) in row_dict.items()
                    }
    target_tmp = trimmed_dict.pop(target)

    # esto no hace nada realmente
    target_int = tf.cond(tf.equal(target_tmp, 1), 
                  lambda: tf.constant(1,dtype=tf.int64),
                  lambda: tf.constant(0,dtype=tf.int64))
    return (trimmed_dict, target_int)

  def caip_uri_to_fields(uri):
      uri = uri[5:]
      project, dataset, table = uri.split('.')
      return project, dataset, table

  project, dataset, table = caip_uri_to_fields(bigquery_uri)
  tensorflow_io_bigquery_client = BigQueryClient()
  read_session = tensorflow_io_bigquery_client.read_session(
      "projects/" + project,
      project, table, dataset,
      features + [TARGET],
      [dtypes.float64] * 29 + [dtypes.int64],
      requested_streams=2)

  dataset = read_session.parallel_read_rows()
  transformed_ds = dataset.map(transform_row)

  # OutputArtifact supports rich metadata, let's add some: 
  #output_dataset.get().metadata['dataset_size'] = 10000
  #output_dataset.get().metadata['info'] = 'Info about dataset'

  # Save dataset in Metadata
  # Use OutputArtifact.path to access a local file path for writing.
  # One can also use OutputArtifact.uri to access the actual URI file path.
  tf.data.experimental.save(transformed_ds, output_data_path)

##############
# Train component
##############
@component(
    base_image='python:3.9', # Use a different base image.
    packages_to_install=['tensorflow']
)
def train(
    dataset: Input[Dataset],

    #dataset: InputArtifact(Dataset),
    features_input: str,
    # Output artifact of type Model.
    output_model: Output[Model],
    metrics: Output[Metrics],
    # An input parameter of type int with a default value.
    num_epochs: int = 3,
  ):        

  import tensorflow as tf
  from tensorflow import feature_column
  import time

  import logging
  logging.getLogger().setLevel(logging.INFO)

  # Converts string to list
  features = list(features_input.split(","))

  full_transformed_ds = tf.data.experimental.load(dataset.path)

  DATASET_FULL_SIZE = 100000
  BATCH_SIZE = 2048

  train_ds_size = int(0.64 * DATASET_FULL_SIZE)
  valid_ds_size = int(0.16 * DATASET_FULL_SIZE)

  dataset_train = full_transformed_ds.take(train_ds_size).shuffle(10).batch(BATCH_SIZE)
  remaining = full_transformed_ds.skip(train_ds_size)  
  dataset_eval = remaining.take(valid_ds_size).batch(BATCH_SIZE)
  dataset_test = remaining.skip(valid_ds_size).batch(BATCH_SIZE)

  logging.info('train_ds_size: %d', train_ds_size)

  feature_columns = []

  # numeric cols
  for header in features:
      feature_columns.append(feature_column.numeric_column(header))

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

  Dense = tf.keras.layers.Dense
  keras_model = tf.keras.Sequential(
    [
      feature_layer,
        Dense(16, activation=tf.nn.relu),
        Dense(8, activation=tf.nn.relu),
        Dense(4, activation=tf.nn.relu),
        Dense(1, activation=tf.nn.sigmoid),
    ])

  # Compile Keras model
  keras_model.compile(
      loss='binary_crossentropy', 
      metrics=['accuracy'],
      optimizer='adam')

  starttime = time.time()
  hist=keras_model.fit(dataset_train, epochs=num_epochs, validation_data=dataset_eval)
  endtime = time.time() - starttime
  logging.info('evaluate: ', keras_model.evaluate(dataset_test))

  # Save model in Metadata
  keras_model.save(output_model.path)
  logging.info('using model.uri: %s', output_model.uri)

  #metrics.log_metric("accuracy", hist.history['accuracy'])
  metrics.log_metric("framework", 'Tensorflow')
  metrics.log_metric("time_to_train_in_seconds", str(endtime-starttime))
  metrics.log_metric("dataset_size", DATASET_FULL_SIZE)
  metrics.log_metric('message', 'this is a message')


##############
# Upload and deploy model in Vertex
##############
@component(
    base_image='python:3.9', # Use a different base image.
    packages_to_install=['google-cloud-aiplatform']
)
def deploy(
    # Input model.
    model: Input[Model],
    vertex_endpoint: Output[Artifact],
    vertex_model: Output[Model]
    ):

  import logging
  logging.getLogger().setLevel(logging.INFO)

  from google.cloud import aiplatform
  aiplatform.init(project='windy-site-254307')

  # Upload model
  uploaded_model = aiplatform.Model.upload(
      display_name=f'pipeline-lw-tf',
      artifact_uri=model.uri,
      serving_container_image_uri='us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-3:latest'
  )
  logging.info('uploaded model: %s', uploaded_model.resource_name)

  # Deploy model
  endpoint = uploaded_model.deploy(
      machine_type='n1-standard-4'
  )
  logging.info('endpoint: %s', endpoint.resource_name)
  vertex_endpoint.uri = endpoint.resource_name
  vertex_model.uri = uploaded_model.resource_name


@dsl.pipeline(
    # A name for the pipeline. Use to determine the pipeline Context.
    name='pipeline-lw-tf',
    description='Demo with lightweight components.',
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT
)

def pipeline(message: str):
  preprocess_task = preprocess(
    bigquery_uri=BIGQUERY_URI, 
    target_input=TARGET, 
    features_input=','.join(FEATURES)) # Converts list to string

  train_task = train(
    #dataset=preprocess_task.outputs['output_dataset'],
    dataset=preprocess_task.output,
    features_input=','.join(FEATURES),
    #message=preprocess_task.outputs['output_parameter'],
    num_epochs=5)
  deploy_task = deploy(
    model = train_task.outputs['output_model']
  )

# Compile and submit
from kfp.v2 import compiler
from google.cloud.aiplatform import pipeline_jobs


compiler.Compiler().compile(pipeline_func=pipeline,                                                     
  package_path='3-pipeline-lwpython-tf/demo-lw-pipeline-tf.json')

from datetime import datetime
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

pipeline_jobs.PipelineJob(
    display_name="pipeline_lw_tf",
    template_path="3-pipeline-lwpython-tf/demo-lw-pipeline-tf.json",
    job_id="pipeline-lwpython-tf-uscentral1-{0}".format(TIMESTAMP),
    parameter_values={'message': "Hello, World"},
    enable_caching=True,
).run(sync=False)