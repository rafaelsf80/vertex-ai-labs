import datetime
import os
import subprocess
import argparse
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

import pandas as pd
import numpy as np
import xgboost as xgb
import json

from google.cloud import storage
from google.cloud import aiplatform

def main(params):
    train_model(params.xgboost_param_max_depth,
                        params.xgboost_param_learning_rate,
                        params.xgboost_param_n_estimators,
                        params.x_train_uri,
                        params.x_test_uri,
                        params.y_train_uri,
                        params.y_test_uri,
                        params.output_model_uri,
                        params.output_metrics_uri
                     ) 
                     
def train_model(xgboost_param_max_depth,
                        xgboost_param_learning_rate,
                        xgboost_param_n_estimators,
                        x_train_uri, x_test_uri,
                        y_train_uri, y_test_uri,
                        output_model_uri,
                        output_metrics_uri):
    

    
    # training_data_uri = os.environ["AIP_TRAINING_DATA_URI"]
    # validation_data_uri = os.environ["AIP_VALIDATION_DATA_URI"]
    # test_data_uri = os.environ["AIP_TEST_DATA_URI"]
    # data_format = os.environ["AIP_DATA_FORMAT"]
    
    
    # model_dir =  datetime.datetime.now().strftime('xgboost_%Y%m%d_%H%M%S')
    # os.environ["AIP_MODEL_DIR"] = model_dir
    # model_dir = os.environ["AIP_MODEL_DIR"]
    
    # train_data_project, train_data_dataset, train_data_table = caip_uri_to_fields(training_data_uri)
    # test_data_project, test_data_dataset, test_data_table = caip_uri_to_fields(test_data_uri)
    # val_data_project, val_data_dataset, val_dat_table = caip_uri_to_fields(validation_data_uri)
    
    # print("caip_train_data_uri:{}".format(training_data_uri))
    # print("project:{}".format(train_data_project))
    # print("dataset:{}".format(train_data_dataset))
    # print("table:{}".format(train_data_table))
    
    # client = bigquery.Client()
    # dataset_ref = bigquery.DatasetReference(train_data_project, train_data_dataset)
    # table_ref = dataset_ref.table(train_data_table)
    
    # destination_uri = "gs://{}/{}*.csv".format(temp_bucket, train_data_table)
    # print("destination_uri:{}".format(destination_uri))
    
    # extract_job = client.extract_table(table_ref,destination_uri,location="us-central1")  
    # extract_job.result()  
    # raw_training_data = pd.read_csv(destination_uri)
    
    
    # train_features = raw_training_data[
    # ['IMP_MONEDA_INTERCAMBIO',
    #      'CLIENTE_PRESENTE',
    #     'TARJETA_PRESENTE',
    #     'OPERACION_CAJERO',
    #     'OPERACION_ONLINE',
    #     'IS_OPER_SEGURA',
    #     'PAIS_COMERCIO', 
    #     'RAMO_COMERCIO',
    #     'TIPO_TARJETA',
    #     'IS_FRAUDE']]
    
    # TARGET_VAR = 'IS_FRAUDE'
    # train_labels = raw_training_data[TARGET_VAR]
    # train_features.pop(TARGET_VAR)
    
    # CATEGORICAL_COLUMNS = (
    # 'TIPO_TARJETA',
    # 'PAIS_COMERCIO')
    # encoders = {col:LabelEncoder() for col in CATEGORICAL_COLUMNS}
    # for col in CATEGORICAL_COLUMNS:
    #     train_features[col] = encoders[col].fit_transform(train_features[col]) 
        
    # X_train, X_test, y_train, y_test = train_test_split(train_features, train_labels, test_size=0.2, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    
    # aiplatform.log_dataset(X_train,"X_train")
    # aiplatform.log_dataset(X_train,"X_test")
    # aiplatform.log_dataset(X_train,"X_val")
    # aiplatform.log_dataset(X_train,"y_train")
    # aiplatform.log_dataset(X_train,"y_test")
    # aiplatform.log_dataset(X_train,"y_val")

    with tf.io.gfile.GFile(os.path.join(x_train_uri, 'X_train')) as f:
        train_set = json.loads(f.read())
    with tf.io.gfile.GFile(os.path.join(x_test_uri, 'X_test')) as f:
        test_set = json.loads(f.read())

    with tf.io.gfile.GFile(os.path.join(y_train_uri, 'y_train')) as f:
        train_labels = json.loads(f.read())
    with tf.io.gfile.GFile(os.path.join(y_test_uri, 'y_test')) as f:
        test_labels = json.loads(f.read())

    # Convert to dataframes
    X_train = pd.read_json(train_set)
    X_test = pd.read_json(test_set)
    # Convert to dataframes. This avoids the error:  “If using all scalar values, you must pass an index”
    y_train = pd.read_json(train_labels, typ='series')
    y_train = y_train.to_frame('count')   
    y_test = pd.read_json(test_labels, typ='series')
    y_test = y_test.to_frame('count')   


     # Step 1: Normalize the data
    #scaler = StandardScaler()
    #train_set = scaler.fit_transform(train_set) # Only normalize on the train set
    #test_set = scaler.transform(test_set)

    # clip() ensures all values fall within the range [-5,5]
    # useful if any outliers remain after normalizing
    #train_set = np.clip(train_set, -5, 5)
    #test_set = np.clip(test_set, -5, 5)

    # Step 2:Determine class weights
    # weight_for_non_fraud = 0.75;#1.0 / df['isFraud'].value_counts()[0]
    # weight_for_fraud = 0.25;#1.0 / df['isFraud'].value_counts()[1]

    # class_weight = {0: weight_for_non_fraud, 1: weight_for_fraud}

    # Step 3: training and evaluation
                               
 
    print(X_train.dtypes)
    print(y_train.dtypes)

          
    clf = xgb.XGBClassifier(max_depth=int(xgboost_param_max_depth), learning_rate=xgboost_param_learning_rate, n_estimators=int(xgboost_param_n_estimators))
    clf.fit(X_train, y_train)
    print(clf)
     
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
    print(cm)
    model = 'model.bst'
    clf.save_model(model)

    # metrics.get().log_confusion_matrix(
    #     ['amount', 'oldbalanceOrg', 'newbalanceDest'],
    #     confusion_matrix(y_pred, y_test).tolist() # .tolist() to convert np array to list.
    # )

    #model_json = model.to_json()
    with tf.io.gfile.GFile(os.path.join(output_model_uri, 'model.json'),
                         'w') as f:
        f.write(model)

    with tf.io.gfile.GFile(os.path.join(output_metrics_uri, 'metrics.json'),
                         'w') as f:
        f.write(json.dumps(acc))



    
    # bucket = storage.Client().bucket(model_output_bucket)
    # blob = bucket.blob('{}/{}'.format(model_dir,model))
    # blob.upload_from_filename(model)
    # print("Model Exported {}".format(model_output_bucket))
    # aiplatform.log_model(model=clf, name='xgboost_model', uri=model_output_bucket,framework='XGboost')
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train XGBoost model')
    parser.add_argument('--project_id', type=str)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--xgboost_param_max_depth', type=int)
    parser.add_argument('--xgboost_param_learning_rate', type=float)
    parser.add_argument('--xgboost_param_n_estimators', type=int)
    parser.add_argument('--x_train_uri', type=str)
    parser.add_argument('--x_test_uri', type=str)
    parser.add_argument('--y_train_uri', type=str)
    parser.add_argument('--y_test_uri', type=str)
    parser.add_argument('--output_model_uri', type=str)
    parser.add_argument('--output_metrics_uri', type=str)
    params = parser.parse_args()
    main(params)
