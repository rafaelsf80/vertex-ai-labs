import datetime
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import tensorflow as tf

import pandas as pd
import xgboost as xgb
import json
import glob
from io import StringIO

from google.cloud import storage
from google.cloud import aiplatform

# This pipeline uses a public dataset at:
# gs://financial_fraud_detection/fraud_data_kaggle.csv

def main(params):
    generate_data(params.project_id,
                        params.temp_bucket,
                        params.gcs_dataset_filename,
                        params.model_output_bucket,
                        params.experiment_name,
                        params.x_train_uri,
                        params.x_test_uri,
                        params.y_train_uri,
                        params.y_test_uri
                     ) 


def caip_uri_to_fields(uri):
    uri = uri[5:]
    project, dataset, table = uri.split('.')
    return project, dataset, table


def generate_data(project_id,
                        temp_bucket,
                        gcs_dataset_filename,
                        model_output_bucket,
                        experiment_name,
                        x_train_uri, x_test_uri,
                        y_train_uri, y_test_uri):
    
    # training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
    # validation_data_uri = os.environ["AIP_VALIDATION_DATA_URI"]
    # test_data_uri = os.environ["AIP_TEST_DATA_URI"]
    # data_format = os.environ["AIP_DATA_FORMAT"]
    
    
    model_dir =  datetime.datetime.now().strftime('xgboost_%Y%m%d_%H%M%S')
    os.environ["AIP_MODEL_DIR"] = model_dir
    model_dir = os.environ["AIP_MODEL_DIR"]
    
    # train_data_project, train_data_dataset, train_data_table = caip_uri_to_fields(training_data_uri)
    # test_data_project, test_data_dataset, test_data_table = caip_uri_to_fields(test_data_uri)
    # val_data_project, val_data_dataset, val_dat_table = caip_uri_to_fields(validation_data_uri)
    
    #print("caip_train_data_uri:{}".format(training_data_uri))
    print("project:{}".format(project_id))
    print("gcs_dataset_filename:{}".format(gcs_dataset_filename))

    df = pd.DataFrame()
    df = df.append(pd.read_csv(gcs_dataset_filename, index_col=None, header=0))
    data = df


    # plit the data into two DataFrames, one for fraud and one for non-fraud 
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
    print(df.head())
    CATEGORICAL_COLUMNS = ['type']
    encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
         df[col] = encoders[col].fit_transform(df[col]) 

    # Preview the updated dataset
    print(df.head())

    # Split the data
    train_test_split = int(len(df) * .8)

    train_set = df[:train_test_split]
    test_set = df[train_test_split:]

    train_labels = train_set.pop('isFraud')
    test_labels = test_set.pop('isFraud')

    # Save artifacts
    print(train_set.dtypes)
    print(train_labels.dtypes)

    X_train = train_set.to_json()
    X_test = test_set.to_json()
    y_train = train_labels.to_json()
    y_test = test_labels.to_json()

    document_file_x_train = os.path.join(x_train_uri, "X_train")
    with tf.io.gfile.GFile(document_file_x_train, 'w') as f:
      f.write(json.dumps(X_train))
   
    document_file_x_test = os.path.join(x_test_uri, "X_test")
    with tf.io.gfile.GFile(document_file_x_test, 'w') as f:
      f.write(json.dumps(X_test))

    document_file_y_train = os.path.join(y_train_uri, "y_train")
    with tf.io.gfile.GFile(document_file_y_train, 'w') as f:
      f.write(json.dumps(y_train))
    
    document_file_y_test = os.path.join(y_test_uri, "y_test")
    with tf.io.gfile.GFile(document_file_y_test, 'w') as f:
      f.write(json.dumps(y_test))   
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load data from GCS')
    parser.add_argument('--project_id', type=str)
    parser.add_argument('--temp_bucket', type=str)
    parser.add_argument('--gcs_dataset_filename', type=str)
    parser.add_argument('--model_output_bucket', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--x_train_uri', type=str)
    parser.add_argument('--x_test_uri', type=str)
    parser.add_argument('--y_train_uri', type=str)
    parser.add_argument('--y_test_uri', type=str)
    params = parser.parse_args()
    main(params)
