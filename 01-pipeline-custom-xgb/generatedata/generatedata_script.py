# This pipeline uses a public dataset at:
# gs://financial_fraud_detection/fraud_data_kaggle.csv

import argparse
import json
import logging

from kfp.v2.components import executor
from kfp.v2.dsl import Output, Dataset

_logger = logging.getLogger(__name__)

def main(project_id: str,
          temp_bucket: str,
          gcs_dataset_filename: str,
          model_output_bucket: str,
          experiment_name: str,
          x_train_artifact: Output[Dataset],
          x_test_artifact: Output[Dataset],
          y_train_artifact: Output[Dataset],
          y_test_artifact: Output[Dataset]):

  import pandas as pd

  from sklearn.preprocessing import LabelEncoder
  from sklearn.model_selection import train_test_split
  from sklearn.utils import shuffle

  logging.info(f'project: {project_id} and gcs_dataset_filename {gcs_dataset_filename}')

  df = pd.DataFrame()
  df = df.append(pd.read_csv(gcs_dataset_filename, index_col=None, header=0))
  data = df

  # Split the data into two DataFrames, one for fraud and one for non-fraud 
  fraud = data[data['isFraud'] == 1]
  not_fraud = data[data['isFraud'] == 0]

  # Take a random sample of non fraud rows
  not_fraud_sample = not_fraud.sample(random_state=2, frac=.005)

  # Put it back together and shuffle
  df = pd.concat([not_fraud_sample,fraud])
  df = shuffle(df, random_state=2)

  # Remove a few columns (isFraud is the label column we'll use, not isFlaggedFraud)
  df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

  # Categorical column
  CATEGORICAL_COLUMNS = ['type']
  encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
  for col in CATEGORICAL_COLUMNS:
        df[col] = encoders[col].fit_transform(df[col]) 

  # Preview the updated dataset
  logging.info(df.head())

  # Split the data
  train_test_split = int(len(df) * .8)

  train_set = df[:train_test_split]
  test_set = df[train_test_split:]

  train_labels = train_set.pop('isFraud')
  test_labels = test_set.pop('isFraud')

  # Save artifacts
  train_set.to_csv(x_train_artifact.path, index=False)
  test_set.to_csv(x_test_artifact.path, index=False)
  train_labels.to_csv(y_train_artifact.path, index=False)
  test_labels.to_csv(y_test_artifact.path, index=False)


def executor_main():
    """Main executor."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--executor_input', type=str)
    parser.add_argument('--function_to_execute', type=str)

    args, _ = parser.parse_known_args()
    print(args)
    executor_input = json.loads(args.executor_input)
    function_to_execute = globals()[args.function_to_execute]

    executor.Executor(
      executor_input=executor_input,
      function_to_execute=function_to_execute).execute()

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    executor_main()
