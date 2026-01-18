import logging
import os

from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
import tensorflow_io

import functools

logging.getLogger().setLevel(logging.INFO)

TENSORBOARD_LOG_DIR = os.environ["AIP_TENSORBOARD_LOG_DIR"]
TRAINING_DATA_URI = os.environ["AIP_TRAINING_DATA_URI"]
VALIDATION_DATA_URI = os.environ["AIP_VALIDATION_DATA_URI"]
TEST_DATA_URI = os.environ["AIP_TEST_DATA_URI"]
DATA_FORMAT = os.environ["AIP_DATA_FORMAT"]
BATCH_SIZE = 32

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

# Preprocess data

MEANS = [94816.7387536405, 0.0011219465482001268, -0.0021445914636999603, -0.002317402958335562,
         -0.002525792169927835, -0.002136576923287782, -3.7586818983702984, 8.135919975738768E-4,
         -0.0015535579268265718, 0.001436137140461279, -0.0012193712736681508, -4.5364970422902533E-4,
         -4.6175444671576083E-4, 9.92177789685366E-4, 0.002366229151475428, 6.710217226762278E-4,
         0.0010325807119864225, 2.557260815835395E-4, -2.0804190062322664E-4, -5.057391100818653E-4,
         -3.452114767842334E-6, 1.0145936326270006E-4, 3.839214074518535E-4, 2.2061197469126577E-4,
         -1.5601580596677608E-4, -8.235017846415852E-4, -7.298316615408554E-4, -6.898459943652376E-5,
         4.724125688297753E-5, 88.73235686453587]

def norm_data(mean, data):
  data = tf.cast(data, tf.float32) * 1/(2*mean)
  return tf.reshape(data, [-1, 1])

numeric_columns = []

for i, feature in enumerate(FEATURES):
  num_col = tf.feature_column.numeric_column(feature, normalizer_fn=functools.partial(norm_data, MEANS[i]))
  numeric_columns.append(num_col)

numeric_columns

# Build model

model = tf.keras.Sequential([
    tf.keras.layers.DenseFeatures(numeric_columns),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])

CLASS_WEIGHT = {
    0: 1,
    1: 100
}
EPOCHS = 3

train_data = training_ds.shuffle(10000)
val_data = eval_ds
test_data = test_ds

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1, profile_batch = '50,100')

model.fit(training_ds, epochs=EPOCHS, class_weight=CLASS_WEIGHT, validation_data=eval_ds, callbacks = [tensorboard_callback])

logging.info(model.evaluate(test_ds))

tf.saved_model.save(model, os.environ["AIP_MODEL_DIR"])