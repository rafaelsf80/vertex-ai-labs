import logging
import os

from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
import tensorflow_io

logging.getLogger().setLevel(logging.INFO)

TRAINING_DATA_URI = 'bq://argolis-rafaelsanchez-ml-dev.dataset_1295149930729439232_tables_2022_06_29T09_57_20_534Z.validation'
VALIDATION_DATA_URI = 'bq://argolis-rafaelsanchez-ml-dev.dataset_1295149930729439232_tables_2022_06_29T09_57_20_534Z.validation'
TEST_DATA_URI = 'bq://argolis-rafaelsanchez-ml-dev.dataset_1295149930729439232_tables_2022_06_29T09_57_20_534Z.test'
BATCH_SIZE = 16

def uri_to_fields(uri):
    uri = uri[5:]
    project, dataset, table = uri.split('.')
    return project, dataset, table

FEATURES = ['Time', 'V1',  'V2',  'V3',  'V4',  'V5',  'V6',  'V7',  'V8',  'V9',
                  'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
                  'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

TARGET = 'Class'

def transform_row(row_dict):

  features = dict(row_dict)
  label = tf.cast(features.pop(TARGET), tf.float64)
  return (features, label)

def read_bigquery(project, dataset, table):
  tensorflow_io_bigquery_client = BigQueryClient()
  read_session = tensorflow_io_bigquery_client.read_session(
      "projects/" + project,
      project, table, dataset,
      FEATURES + [TARGET],
      [tf.int64] + [tf.float64] * (len(FEATURES)-1) + [tf.int64],
      requested_streams=2)

  dataset = read_session.parallel_read_rows()
  transformed_ds = dataset.map(transform_row)
  return transformed_ds

logging.info(f'Using tensorflow {tf.__version__} and tensorflow_io {tensorflow_io.__version__}')

training_ds = read_bigquery(*uri_to_fields(TRAINING_DATA_URI)).shuffle(10).batch(BATCH_SIZE)
eval_ds = read_bigquery(*uri_to_fields(VALIDATION_DATA_URI)).batch(BATCH_SIZE)
test_ds = read_bigquery(*uri_to_fields(TEST_DATA_URI)).batch(BATCH_SIZE)

logging.info(TRAINING_DATA_URI)
logging.info("first batch")
logging.info(next(iter(training_ds))) # Print first batch

def encode_numerical_feature(feature, name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = tf.keras.layers.Normalization()

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])
  feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  encoded_feature = normalizer(feature)
  return encoded_feature


all_inputs = []
encoded_features = []

# Numerical features.
for header in FEATURES:
  numeric_col = tf.keras.Input(shape=(1,), name=header)
  all_inputs.append(numeric_col)
  logging.info(header)

  encoded_numeric_col = encode_numerical_feature(numeric_col, header, training_ds)
  encoded_features.append(encoded_numeric_col)


all_features = tf.keras.layers.concatenate(encoded_features)
x = tf.keras.layers.Dense(64, activation="relu")(all_features)
x = tf.keras.layers.Dropout(0.5)(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(all_inputs, output)

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])

CLASS_WEIGHT = {
    0: 1,
    1: 100
}
EPOCHS = 1

model.fit(training_ds, epochs=EPOCHS, class_weight=CLASS_WEIGHT, validation_data=eval_ds)

tf.saved_model.save(model, '.')
test_instance={
    'Time':80422,
    'Amount':17.99,
    'V1':-0.24,
    'V2':-0.027,
    'V3':0.064,
    'V4':-0.16,
    'V5':-0.152,
    'V6':-0.3,
    'V7':-0.03,
    'V8':-0.01,
    'V9':-0.13,
    'V10':-0.18,
    'V11':-0.16,
    'V12':0.06,
    'V13':-0.11,
    'V14':2.1,
    'V15':-0.07,
    'V16':-0.033,
    'V17':-0.14,
    'V18':-0.08,
    'V19':-0.062,
    'V20':-0.08,
    'V21':-0.06,
    'V22':-0.088,
    'V23':-0.03,
    'V24':0.01,
    'V25':-0.04,
    'V26':-0.99,
    'V27':-0.13,
    'V28':0.003,
}

response = model.predict([test_instance])
print(response)


