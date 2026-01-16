import argparse
import logging
import os

from google.cloud import bigquery
from google.cloud import storage
import xgboost as xgb

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import hypertune

logging.getLogger().setLevel(logging.INFO)

TENSORBOARD_LOG_DIR = os.environ["AIP_TENSORBOARD_LOG_DIR"]
BQ_SOURCE = 'bq://argolis-rafaelsanchez-ml-dev.ml_datasets_europewest4.ulb_'


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
  parser.add_argument('--depth', type=int, default=6)
  parser.add_argument('--activation', type=str, default='relu')
  parser.add_argument('--batch_size', type=int, default=128)

  args = parser.parse_args()
  return args


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

def read_bigquery(uri):
    project, dataset, table = uri_to_fields(uri)
    client = bigquery.Client(project=project)
    # Using the bigquery storage client for faster downloads
    query = f"SELECT {','.join(FEATURES + [TARGET])} FROM `{project}.{dataset}.{table}`"
    logging.info(f"Reading data from {uri}...")
    df = client.query(query).to_dataframe()
    return df

# Load datasets
logging.info("Loading training data...")
df = read_bigquery(BQ_SOURCE)
logging.info("Data loaded: {} rows".format(len(df)))
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
eval_df = df.drop(train_df.index).drop(test_df.index)

X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_eval, y_eval = eval_df[FEATURES], eval_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]

class TensorBoardCallback(xgb.callback.TrainingCallback):
    def __init__(self, log_dir):
        # Initialize the TensorFlow SummaryWriter
        self.writer = tf.summary.create_file_writer(log_dir)

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration to log metrics."""
        with self.writer.as_default():
            for data_name, metrics in evals_log.items():
                for metric_name, log_values in metrics.items():
                    # Log the most recent value (last in the list)
                    tf.summary.scalar(f"{data_name}-{metric_name}", log_values[-1], step=epoch)
        self.writer.flush()
        return False  # Return True to stop training early

class HPTCallback(xgb.callback.TrainingCallback):
    def __init__(self, metric_tag='accuracy'):
        super().__init__()
        self._hp_tune_reporter = hypertune.HyperTune()
        self.metric_tag = metric_tag

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration to report metrics for hyperparameter tuning."""
        # evals_log format: {'train': {'accuracy': [...]}, 'eval': {'accuracy': [...]}}
        # We report the 'eval' accuracy
        if 'eval' in evals_log and self.metric_tag in evals_log['eval']:
            metric_value = evals_log['eval'][self.metric_tag][-1]
            self._hp_tune_reporter.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=self.metric_tag,
                metric_value=metric_value,
                global_step=epoch
            )
        return False

# Convert to DMatrix (Native XGBoost format)
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
deval = xgb.DMatrix(X_eval.values, label=y_eval.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

# Move parameters to a dictionary for the native API
params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': 100,
    'max_depth': args.depth,
    'learning_rate': args.lr,
    'eval_metric': 'error' # accuracy in xgb is 1-error, using 'error' is common for binary
}

logging.info("Starting training...")
log_dir = TENSORBOARD_LOG_DIR
tb_callback = TensorBoardCallback(log_dir)
# 'accuracy' is typically 1-error in binary:logistic if not explicitly added
vertex_hpt_callback = HPTCallback(metric_tag='error') 

# Use the native xgb.train instead of model.fit to ensure callback compatibility
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (deval, 'eval')],
    callbacks=[tb_callback, vertex_hpt_callback],
    verbose_eval=True
)

logging.info("Starting evaluation...")
# For the native API, we use predict and calculate the accuracy manually
preds = model.predict(dtest)
predictions = [1 if p > 0.5 else 0 for p in preds]
accuracy = sum(1 for i, j in zip(predictions, y_test.values) if i == j) / len(y_test)
logging.info(f"Test Accuracy: {accuracy}")

# Save model locally
local_model_path = 'model.bst'
model.save_model(local_model_path)
