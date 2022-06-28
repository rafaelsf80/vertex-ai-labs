
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import (Artifact, Dataset, Input, InputPath, Model, Output,
                        OutputPath, ClassificationMetrics, Metrics, component)

from google.cloud.aiplatform import pipeline_jobs

import logging


# We'll use this beta library for metadata querying
from google.cloud import aiplatform_v1beta1

PROJECT_ID = 'windy-site-254307'
MY_STAGING_BUCKET = 'caip-prediction-custom-uscentral1'
REGION = 'us-central1'
USER='rafaelsanchez'
PIPELINE_ROOT = 'gs://{}/pipeline_root/{}'.format(MY_STAGING_BUCKET, USER)

#########################
### Download BigQuery and convert to CSV
#########################

@component(
    packages_to_install=["google-cloud-bigquery", "pandas", "pyarrow"],
    base_image="python:3.9",
    output_component_file="4-pipeline-lwpython-xgb/get_dataset.yaml"
)
def get_dataframe(
    bq_table: str,
    output_data_path: OutputPath("Dataset")
):
    from google.cloud import bigquery
    import pandas as pd

    bqclient = bigquery.Client()
    table = bigquery.TableReference.from_string(
        bq_table
    )
    rows = bqclient.list_rows(
        table
    )
    dataframe = rows.to_dataframe(
        create_bqstorage_client=True,
    )
    dataframe = dataframe.sample(frac=1, random_state=2)
    dataframe.to_csv(output_data_path)

#########################
### Training
#########################
@component(
    packages_to_install=["sklearn", "pandas", "joblib", "xgboost"],
    base_image="python:3.9",
    output_component_file="4-pipeline-lwpython-xgb/xgb_model_training.yaml"
)
def xgb_train(
    dataset: Input[Dataset],
    metrics: Output[Metrics],
    model: Output[Model],
    xgboost_param_max_depth: int,
    xgboost_param_learning_rate: float,
    xgboost_param_n_estimators: int,
    metricsc: Output[ClassificationMetrics],

):
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import roc_curve
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    from sklearn import model_selection

    from sklearn.model_selection import train_test_split, cross_val_predict
    from joblib import dump

    import pandas as pd
    import xgboost as xgb
    import numpy as np

    import logging

    df = pd.read_csv(dataset.path)
    labels = df.pop("Class").tolist()
    data = df.values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(data, labels)

    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)

    classifier = xgb.XGBClassifier(max_depth=int(xgboost_param_max_depth), learning_rate=xgboost_param_learning_rate, n_estimators=int(xgboost_param_n_estimators))
    classifier.fit(x_train,y_train)
    print(classifier)
    
    score = accuracy_score(y_test, classifier.predict(x_test))
   
    y_pred = classifier.predict(x_test)
    cm = confusion_matrix(y_pred, y_test, labels=['DERMASON','SEKER','CALI','SIRA','BOMBAY','BARBUNYA','HOROZ'])
    print(cm)
    #model = 'model.bst'
    #clf.save_model(model)

    # log metrics
    logging.info("accuracy is: %s", score)
    metrics.log_metric("accuracy",(score * 100.0))
    metrics.log_metric("framework", "XGBoost")
    metrics.log_metric("dataset_size", len(df))

    # log the ROC curve
    # fpr = []
    # tpr = []
    # thresholds = []
    # y_scores = cross_val_predict(classifier, x_train, y_train, cv=3, method='predict_proba')
    # y_predict = cross_val_predict(classifier, x_train, y_train, cv=3, method='predict')
    # fpr, tpr, thresholds = roc_curve(y_true=y_train, y_score=y_scores[:,1], pos_label=True)
    # metricsc.log_roc_curve(fpr, tpr, thresholds)

    # log the confusion matrix
    predictions = model_selection.cross_val_predict(classifier, x_train, y_train, cv=3)
    metricsc.log_confusion_matrix(
        ['DERMASON','SEKER','CALI','SIRA','BOMBAY','BARBUNYA','HOROZ'],
        confusion_matrix(y_train, predictions).tolist() #to convert np array to list.
    )
            
    dump(classifier, model.path + ".bst")


#########################
### Define pipeline
#########################

@dsl.pipeline(
    # Default pipeline root. You can override it when submitting the pipeline.
    pipeline_root=PIPELINE_ROOT,
    # A name for the pipeline.
    name="demo-lwtraining-xgboost",
)
def pipeline(
    bq_table: str = "",
    output_data_path: str = "data.csv",
    project: str = PROJECT_ID,
    region: str = REGION,
    xgboost_param_max_depth: int=10,
    xgboost_param_learning_rate: float=0.1,
    xgboost_param_n_estimators: int=200
):
    dataset_task = get_dataframe(bq_table)

    model_task = xgb_train(
        dataset_task.output,
        xgboost_param_max_depth,
        xgboost_param_learning_rate,
        xgboost_param_n_estimators
    )

#########################
### Compile and run pipeline on Vertex AI
#########################

logging.getLogger().setLevel(logging.INFO)

compiler.Compiler().compile(
    pipeline_func=pipeline, package_path="4-pipeline-lwpython-xgb/demo-lw-pipeline-xgb.json"
)

from datetime import datetime

TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

run1 = pipeline_jobs.PipelineJob(
    display_name="demo-lwtraining-xgb-small",
    template_path="4-pipeline-lwpython-xgb/demo-lw-pipeline-xgb.json",
    job_id="demo-lwtraining-xgb-small-{0}".format(TIMESTAMP),
    parameter_values={"bq_table": "sara-vertex-demos.beans_demo.small_dataset"},
    enable_caching=True,
)

run2 = pipeline_jobs.PipelineJob(
    display_name="demo-lwtraining-xgb-large",
    template_path="4-pipeline-lwpython-xgb/demo-lw-pipeline-xgb.json",
    job_id="demo-lwtraining-xgb-large-{0}".format(TIMESTAMP),
    parameter_values={"bq_table": "sara-vertex-demos.beans_demo.large_dataset"},
    enable_caching=True,
)

run1.run()
run2.run()