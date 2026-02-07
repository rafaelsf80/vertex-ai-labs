from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow import feature_column
import os

training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
validation_data_uri = os.environ["AIP_VALIDATION_DATA_URI"]
test_data_uri = os.environ["AIP_TEST_DATA_URI"]
data_format = os.environ["AIP_DATA_FORMAT"]

def caip_uri_to_fields(uri):
    uri = uri[5:]
    project, dataset, table = uri.split('.')
    return project, dataset, table

feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

target_name = 'species'

def transform_row(row_dict):
  # Trim all string tensors
  trimmed_dict = { column:
                  (tf.strings.strip(tensor) if tensor.dtype == 'string' else tensor) 
                  for (column,tensor) in row_dict.items()
                  }
  target = trimmed_dict.pop(target_name)

  target_float = tf.cond(tf.equal(tf.strings.strip(target), 'versicolor'), 
                 lambda: tf.constant(1.0),
                 lambda: tf.constant(0.0))
  return (trimmed_dict, target_float)

def read_bigquery(project, dataset, table):
  tensorflow_io_bigquery_client = BigQueryClient()
  read_session = tensorflow_io_bigquery_client.read_session(
      "projects/" + project,
      project, table, dataset,
      feature_names + [target_name],
      [dtypes.float64] * 4 + [dtypes.string],
      requested_streams=2)

  dataset = read_session.parallel_read_rows()
  transformed_ds = dataset.map(transform_row)
  return transformed_ds

BATCH_SIZE = 16

training_ds = read_bigquery(*caip_uri_to_fields(training_data_uri)).shuffle(10).batch(BATCH_SIZE)
eval_ds = read_bigquery(*caip_uri_to_fields(validation_data_uri)).batch(BATCH_SIZE)
test_ds = read_bigquery(*caip_uri_to_fields(test_data_uri)).batch(BATCH_SIZE)


feature_columns = []

# numeric cols
for header in feature_names:
  feature_columns.append(feature_column.numeric_column(header))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

Dense = tf.keras.layers.Dense
model = tf.keras.Sequential(
  [
    feature_layer,
      Dense(16, activation=tf.nn.relu),
      Dense(8, activation=tf.nn.relu),
      Dense(4, activation=tf.nn.relu),
      Dense(1, activation=tf.nn.sigmoid),
  ])

# Compile Keras model
model.compile(
    loss='binary_crossentropy', 
    metrics=['accuracy'],
    optimizer='adam')

model.fit(training_ds, epochs=5, validation_data=eval_ds)

print(model.evaluate(test_ds))

tf.saved_model.save(model, os.environ["AIP_MODEL_DIR"])