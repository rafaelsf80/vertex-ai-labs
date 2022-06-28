from google.cloud import bigquery
from google.cloud import aiplatform
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection

from sklearn.model_selection import train_test_split
from joblib import dump

import numpy as np
import pandas as pd
import xgboost as xgb

xgboost_param_max_depth=10
xgboost_param_learning_rate=0.1
xgboost_param_n_estimators=200

bqclient = bigquery.Client()
table = bigquery.TableReference.from_string(
    'sara-vertex-demos.beans_demo.small_dataset'
)
rows = bqclient.list_rows(
    table
)
dataframe = rows.to_dataframe(
    create_bqstorage_client=True,
)
dataframe = dataframe.sample(frac=1, random_state=2)
dataframe.to_csv('./data.csv')

df = pd.read_csv('./data.csv')
labels = df.pop("Class").tolist()
data = df.values.tolist()
x_train, x_test, y_train, y_test = train_test_split(data, labels)

# train_test_split() returns lists, we need to convert it to numpy to avoid
# 'list' object has no attribute 'shape' errors in xgb.fit()
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

print('accuracy is:',score)

print(y_train)


# aiplatform.log_metric("accuracy",(score * 100.0))
# aiplatform.log_metric("framework", "Scikit Learn")
# aiplatform.log_metric("dataset_size", len(df))
# #metrics.log_confusion_matrix(
# #        annotations,
# predictions = model_selection.cross_val_predict(classifier, x_train, y_train, cv=3)
# aiplatform.log_confusion_matrix(
#     ["Area", "Perimeter", "MajorAxisLength"],
#     confusion_matrix(
#         y_train, predictions
#     ).tolist(),  # .tolist() to convert np array to list.
# )

        
dump(classifier, "./classifier.bst")