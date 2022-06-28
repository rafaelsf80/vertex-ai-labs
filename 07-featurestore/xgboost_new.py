import datetime
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pathlib
import numpy as np
import pandas as pd
import xgboost as xgb

from google.cloud import storage
from google.cloud import bigquery

from google.cloud import aiplatform
from IPython.display import display, Image


# Set up project, location, featurestore ID and endpoints
PROJECT_ID = "windy-site-254307"  
LOCATION = "us-central1"  

# Connect with ML metadata
aiplatform.connect(project=PROJECT_ID, location=LOCATION)
aiplatform.set_experiment('xgb-fraudulent-operations')

@aiplatform.execution(name="Data Reader")
def read_data(project_id,
                        temp_bucket,
                        bq_dataset,
                        bq_table,
                        bq_sql_extract):
    # aiplatform.log_parameter('temp_bucket', bq_dataset)          
    # aiplatform.log_parameter('bq_dataset', bq_dataset)
    # aiplatform.log_parameter('bq_table', bq_table)
    # aiplatform.log_parameter('bq_sql_extract', bq_sql_extract)

    logging.info("Start exec ...")    
    client = bigquery.Client()
    destination_uri = "gs://{}/{}.csv".format(temp_bucket, bq_table)                    
    dataset_ref = bigquery.DatasetReference(project_id, bq_dataset)
    table_ref = dataset_ref.table(bq_table)

    extract_job = client.extract_table(table_ref,destination_uri,location=LOCATION)  
    extract_job.result()  
    
    CATEGORICAL_COLUMNS = (
    'cliente_presente',
    'tarjeta_presente',
    'operacion_cajero',
    'operacion_online',
    'des_pais_operacion',
    'is_oper_segura',
    'tipo_lectura_tarjeta',
    'ramo_comercio',
    'tipo_tarjeta')

    TARGET_VAR = 'is_fraude'

    print("Writing to: ", destination_uri)
    raw_training_data = pd.read_csv(destination_uri)
    train_features = raw_training_data[['hora_operacion',
    'imp_moneda_intercambio',
    'cliente_presente',
    'tarjeta_presente',
    'operacion_cajero',
    'operacion_online',
    'des_pais_operacion',
    'is_oper_segura',
    'tipo_lectura_tarjeta',
    'ramo_comercio',
    'tipo_tarjeta',
    'is_fraude']]
    train_labels = raw_training_data[TARGET_VAR]
    encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
    for col in CATEGORICAL_COLUMNS:
        train_features[col] = encoders[col].fit_transform(train_features[col])
 
    # aiplatform.log_dataset(train_features, "Train features")
    # aiplatform.log_dataset(train_labels, "Train labels")

    return train_features, train_labels


@aiplatform.execution(name="Data Splitter")
def train_and_test_split(train_features, train_labels, test_size=0.2, random_state=0):
    aiplatform.log_parameters(split_fraction=test_size,
                            random_state=random_state)
    
    X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=1)
    
    # aiplatform.log_dataset(X_train, "Train Data")
    # aiplatform.log_dataset(X_test, "Test Data")
    # aiplatform.log_dataset(y_train, "Train Labels")
    # aiplatform.log_dataset(y_test, "Test Labels")

    return X_train, X_test, y_train, y_test


@aiplatform.execution(name="Trainer")
def build_train_model(X_train, X_test, y_train, y_test, model_output_bucket):

    aiplatform.log_parameter('model_output_bucket', model_output_bucket)
    
    print("******Serializing ...")
    X_train_serialized = X_train.to_json()
    y_train_serialized = y_train.to_json()

    print("******Deserializing....")
    X_train = pd.read_json(X_train_serialized)
    y_train = pd.read_json(y_train_serialized, typ='series')
    y_train = y_train.to_frame('count')

    
    print('Training ...')
    clf = xgb.XGBClassifier(max_depth=7, learning_rate=0.2, n_estimators=200)
    clf.fit(X_train, y_train)
    print(clf)
    print(accuracy_score(y_test, clf.predict(X_test)))

    model = 'xgboost_model.bst'
    clf.save_model(model)

    bucket = storage.Client().bucket(model_output_bucket)
    blob = bucket.blob('{}/{}'.format(
         datetime.datetime.now().strftime('xgboost_%Y%m%d_%H%M%S'),model))
    blob.upload_from_filename(model)
    print("Model Exported {}".format(model_output_bucket))

    # aiplatform.log_model(clf, 'XGB Model')
    # aiplatform.log_metric('accuracy', accuracy_score(y_test, clf.predict(X_test)))


def main(params):

    # Read dataset from BigQuery
    train_features, train_labels = read_data(params.project_id,
                        params.temp_bucket,
                        params.bq_dataset,
                        params.bq_table,
                        params.bq_sql_extract)
    # Data Split
    train_dataset, test_dataset, train_labels, test_labels = train_and_test_split(train_features, train_labels)

    # Train and eval
    build_train_model(train_dataset, test_dataset, train_labels, test_labels, params.model_output_bucket)
  
    # Save png file
    display(aiplatform.graph_experiment())


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train XGBoost model with Feature Store and ML Metadata')
    parser.add_argument('--project_id', type=str)
    parser.add_argument('--temp_bucket', type=str)
    parser.add_argument('--bq_dataset', type=str)
    parser.add_argument('--bq_table', type=str)
    parser.add_argument('--bq_sql_extract', type=str)
    parser.add_argument('--model_output_bucket', type=str)
    params = parser.parse_args()
    main(params)
