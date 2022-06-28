import argparse
import logging
import os

from tensorflow_io.bigquery import BigQueryClient
import tensorflow as tf
import tensorflow_io

from tensorboard.plugins.hparams import api as hp
import hypertune

logging.getLogger().setLevel(logging.INFO)

TENSORBOARD_LOG_DIR = os.environ["AIP_TENSORBOARD_LOG_DIR"]
TRAINING_DATA_URI = os.environ["AIP_TRAINING_DATA_URI"]
VALIDATION_DATA_URI = os.environ["AIP_VALIDATION_DATA_URI"]
TEST_DATA_URI = os.environ["AIP_TEST_DATA_URI"]
DATA_FORMAT = os.environ["AIP_DATA_FORMAT"]
BATCH_SIZE = 16


def get_args():
  """Argument parser.
  Returns:
    Dictionary of arguments.
  """
  parser = argparse.ArgumentParser(description='ML in the cloud course - HT')
  parser.add_argument(
      '--tfds',
      default=None,
      help='The tfds URI from https://www.tensorflow.org/datasets/ to load the data from')

  parser.add_argument('--lr', type=float, default=0.01)
  parser.add_argument('--units', type=int, default=4)
  parser.add_argument('--activation', type=str, default='relu')
  parser.add_argument('--batch_size', type=int, default=128)

  args = parser.parse_args()
  return args

class HPTCallback(tf.keras.callbacks.Callback):
    
    def __init__(self):
        super().__init__()
        self._hp_tune_reporter = hypertune.HyperTune()

    def on_epoch_end(self, epoch, logs=None):
        self._hp_tune_reporter.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='accuracy',
            metric_value=logs['accuracy'],
            global_step=epoch)

# Training settings
args = get_args()

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

training_ds = read_bigquery(*uri_to_fields(TRAINING_DATA_URI)).shuffle(10).batch(args.batch_size)
eval_ds = read_bigquery(*uri_to_fields(VALIDATION_DATA_URI)).batch(args.batch_size)
test_ds = read_bigquery(*uri_to_fields(TEST_DATA_URI)).batch(args.batch_size)

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

strategy = tf.distribute.MirroredStrategy()    
with strategy.scope():
  all_features = tf.keras.layers.concatenate(encoded_features)
  x = tf.keras.layers.Dense(64, activation=args.activation)(all_features)
  x = tf.keras.layers.Dropout(0.5)(x)
  output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

  model = tf.keras.Model(all_inputs, output)

  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy', tf.keras.metrics.AUC(curve='PR')])

CLASS_WEIGHT = {
    0: 1,
    1: 100
}
EPOCHS = 3

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR, histogram_freq=1, profile_batch = '50,100')
vertex_hpt_callback = HPTCallback() # log to Vertex UI (hypertune)

model.fit(training_ds, epochs=EPOCHS, class_weight=CLASS_WEIGHT, validation_data=eval_ds, callbacks = [vertex_hpt_callback, tensorboard_callback])

logging.info(model.evaluate(test_ds))

tf.saved_model.save(model, os.environ["AIP_MODEL_DIR"])