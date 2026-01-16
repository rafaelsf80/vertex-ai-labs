from explainable_ai_sdk.metadata.tf.v2 import SavedModelMetadataBuilder

from tensorflow.python.client import device_lib
import tensorflow_datasets as tfds
import tensorflow as tf

import numpy as np
import tempfile
import argparse
import sys
import os

tfds.disable_progress_bar()

parser = argparse.ArgumentParser()
parser.add_argument('--model-dir', dest='model_dir',
                    default=os.getenv('AIP_MODEL_DIR'), type=str, help='Model dir.')
parser.add_argument('--lr', dest='lr',
                    default=0.001, type=float,
                    help='Learning rate.')
parser.add_argument('--epochs', dest='epochs',
                    default=20, type=int,
                    help='Number of epochs.')
parser.add_argument('--steps', dest='steps',
                    default=100, type=int,
                    help='Number of steps per epoch.')
parser.add_argument('--distribute', dest='distribute', type=str, default='single',
                    help='distributed training strategy')
args = parser.parse_args()

print('Python Version = {}'.format(sys.version))
print('TensorFlow Version = {}'.format(tf.__version__))
print('TF_CONFIG = {}'.format(os.environ.get('TF_CONFIG', 'Not found')))

# Single Machine, single compute device
if args.distribute == 'single':
    if tf.test.is_gpu_available():
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
# Single Machine, multiple compute device
elif args.distribute == 'mirror':
    strategy = tf.distribute.MirroredStrategy()
# Multiple Machine, multiple compute device
elif args.distribute == 'multi':
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

# Multi-worker configuration
print('num_replicas_in_sync = {}'.format(strategy.num_replicas_in_sync))

def make_dataset():
  # Scaling Boston Housing data features
  def scale(feature):
    max = np.max(feature)
    feature = (feature / max).astype(np.float)
    return feature

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
  )
  for _ in range(13):
    x_train[_] = scale(x_train[_])
    x_test[_] = scale(x_test[_])
  return (x_train, y_train), (x_test, y_test)

# Build the Keras model
def build_and_compile_dnn_model():
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(13,)),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(1, activation='linear')
  ])
  model.compile(
      loss='mse',
      optimizer=tf.keras.optimizers.RMSprop(learning_rate=args.lr))
  return model

# Train the model
NUM_WORKERS = strategy.num_replicas_in_sync
# Here the batch size scales up by number of workers since
# `tf.data.Dataset.batch` expects the global batch size.
BATCH_SIZE = 16
GLOBAL_BATCH_SIZE = BATCH_SIZE * NUM_WORKERS

with strategy.scope():
  # Creation of dataset, and model building/compiling need to be within
  # `strategy.scope()`.
  model = build_and_compile_dnn_model()

# Train the model
(x_train, y_train), (x_test, y_test) = make_dataset()
model.fit(x_train, y_train, epochs=args.epochs, batch_size=GLOBAL_BATCH_SIZE)

tmpdir = tempfile.mkdtemp()

model.save(tmpdir)

# Save TF Model with Explainable metadata to GCS
builder = SavedModelMetadataBuilder(tmpdir)
builder.save_model_with_metadata(args.model_dir)