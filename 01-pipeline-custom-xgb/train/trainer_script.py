import argparse
import json
import logging

from kfp.v2.components import executor
from kfp.v2.dsl import Input, Output, Model, Metrics, Dataset, ClassificationMetrics

_logger = logging.getLogger(__name__)

def main(xgboost_param_max_depth: int,
            xgboost_param_learning_rate: float,
            xgboost_param_n_estimators: int,
            x_train_artifact: Input[Dataset],
            x_test_artifact: Input[Dataset],
            y_train_artifact: Input[Dataset],
            y_test_artifact: Input[Dataset],
            model: Output[Model],
            metrics: Output[Metrics],
            metricsc: Output[ClassificationMetrics]
):
    
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.preprocessing import StandardScaler

    import numpy as np
    import pandas as pd
    import xgboost as xgb

    X_train = pd.read_csv(x_train_artifact.path)
    X_test = pd.read_csv(x_test_artifact.path)
    y_train = pd.read_csv(y_train_artifact.path)
    y_test = pd.read_csv(y_test_artifact.path)

    # Step 1: Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train) # Only normalize on the train set
    X_test = scaler.transform(X_test)

    # clip() ensures all values fall within the range [-5,5]
    # useful if any outliers remain after normalizing
    X_train = np.clip(X_train, -5, 5)
    X_test = np.clip(X_test, -5, 5)

    # Step 2: Determine class weights
    weight_for_non_fraud = 0.75;#1.0 / df['isFraud'].value_counts()[0]
    weight_for_fraud = 0.25;#1.0 / df['isFraud'].value_counts()[1]

    class_weight = {0: weight_for_non_fraud, 1: weight_for_fraud}

    # Step 3: Training and evaluation       
    clf = xgb.XGBClassifier(max_depth=int(xgboost_param_max_depth), learning_rate=xgboost_param_learning_rate, n_estimators=int(xgboost_param_n_estimators))
    clf.fit(X_train, y_train)
     
    acc = accuracy_score(y_test, clf.predict(X_test))
    
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_pred, y_test)
 
    # If you want to set a custom path, use .uri method. Then access using .path method. Do not modify .path directly
    model.uri = f'{model.uri}.bst'
    logging.info(f'model.path is {model.path} and model.uri is {model.uri}')
    
    clf.save_model(model.path)
    #model = 'model.bst'
    #dump(model, model.path)

    metricsc.log_confusion_matrix(
        ["Non-Fraudulent", "Fraudulent"],
        cm.tolist(),  # .tolist() to convert np array to list.
    )

    metrics.log_metric("accuracy", acc)
    metrics.log_metric("framework", "XGBoost")
    metrics.log_metric("dataset_size", "pending")
    
    
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
