import logging
import os
import pandas as pd
import xgboost as xgb
from google.cloud import bigquery
from google.cloud import storage
import tensorflow as tf

logging.getLogger().setLevel(logging.INFO)

# OPTIONAL: if you created a Managed Dataset
# TRAINING_DATA_URI = os.environ["AIP_TRAINING_DATA_URI"]
# VALIDATION_DATA_URI = os.environ["AIP_VALIDATION_DATA_URI"]
# TEST_DATA_URI = os.environ["AIP_TEST_DATA_URI"]
TENSORBOARD_LOG_DIR = os.environ["AIP_TENSORBOARD_LOG_DIR"]
DATA_URI=os.environ["bq_dataset"]
MODEL_DIR = os.environ["AIP_MODEL_DIR"]

def uri_to_fields(uri):
    # Expect bq://project.dataset.table
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
df = read_bigquery(DATA_URI)
logging.info("Data loaded: {} rows".format(len(df)))
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index).sample(frac=0.5, random_state=42)
eval_df = df.drop(train_df.index).drop(test_df.index)


# OPTIONAL: if you created a Managed dataset
# logging.info("Loading training data...")
# train_df = read_bigquery(TRAINING_DATA_URI)
# logging.info("Loading validation data...")
# eval_df = read_bigquery(VALIDATION_DATA_URI)
# logging.info("Loading test data...")
# test_df = read_bigquery(TEST_DATA_URI)

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

# Convert to DMatrix (Native XGBoost format)
dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
deval = xgb.DMatrix(X_eval.values, label=y_eval.values)
dtest = xgb.DMatrix(X_test.values, label=y_test.values)

# Move parameters to a dictionary for the native API
params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'eval_metric': 'aucpr' # accuracy
}

logging.info("Starting training...")
log_dir = TENSORBOARD_LOG_DIR
tb_callback = TensorBoardCallback(log_dir)

# Use the native xgb.train instead of model.fit to ensure callback compatibility
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (deval, 'eval')],
    callbacks=[tb_callback],
    verbose_eval=True
)

logging.info("Starting evaluation...")
# For the native API, we use predict and calculate the accuracy manually
preds = model.predict(dtest)
predictions = [1 if p > 0.5 else 0 for p in preds]
accuracy = sum(1 for i, j in zip(predictions, y_test.values) if i == j) / len(y_test)
logging.info(f"Test Accuracy: {accuracy}")

# Save model locally first
local_model_path = 'model.bst'
model.save_model(local_model_path)

# Upload to GCS if MODEL_DIR is a gs:// path
if MODEL_DIR.startswith("gs://"):
    logging.info(f"Uploading model to {MODEL_DIR}...")
    # Extract bucket and path
    path_parts = MODEL_DIR.replace("gs://", "").split('/')
    bucket_name = path_parts[0]
    blob_path = '/'.join(path_parts[1:]).rstrip('/')
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    # Vertex AI prediction expects the model file name to be specific (model.bst, model.joblib, or model.pkl)
    # We should upload it directly into the MODEL_DIR path
    blob = bucket.blob(f"{blob_path}/model.bst")
    blob.upload_from_filename(local_model_path)
    logging.info(f"Model successfully uploaded to {MODEL_DIR}/model.bst")
else:
    # Handle local directory case
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.save_model(os.path.join(MODEL_DIR, 'model.bst'))
    logging.info(f"Model saved locally to {os.path.join(MODEL_DIR, 'model.bst')}")

logging.info("Training script execution complete.")