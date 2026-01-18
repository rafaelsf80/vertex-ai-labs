""" Train a XGB model and show WIT widget """

""" IMPORTANT: You MUST run as a cell in a notebook (Vertex Workbench managed notebook).
You can only render HTML (WitWidget) in a browser or notebook and not in a Python console/editor environment.
It works in a browser, in Jupiter notebook, Jupyter Lab, etc."""

import pandas as pd
import xgboost as xgb
import numpy as np
import collections

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import shuffle
from witwidget.notebook.visualization import WitWidget, WitConfigBuilder


COLUMN_NAMES = collections.OrderedDict({
 'as_of_year': np.int16,
 'agency_code': 'category',
 'loan_type': 'category',
 'property_type': 'category',
 'loan_purpose': 'category',
 'occupancy': np.int8,
 'loan_amt_thousands': np.float64,
 'preapproval': 'category',
 'county_code': np.float64,
 'applicant_income_thousands': np.float64,
 'purchaser_type': 'category',
 'hoepa_status': 'category',
 'lien_status': 'category',
 'population': np.float64,
 'ffiec_median_fam_income': np.float64,
 'tract_to_msa_income_pct': np.float64,
 'num_owner_occupied_units': np.float64,
 'num_1_to_4_family_units': np.float64,
 'approved': np.int8
})

# Download the pre-processed dataset. Uncomment the following line
#!gsutil cp 'gs://mortgage_dataset_files/mortgage-small.csv' .


# Read the dataset with Pandas
data = pd.read_csv(
 'mortgage-small.csv',
 index_col=False,
 dtype=COLUMN_NAMES
)
data = data.dropna()
data = shuffle(data, random_state=2)
data.head()


# Class labels - 0: denied, 1: approved
print(data['approved'].value_counts())
labels = data['approved'].values
data = data.drop(columns=['approved'])

# Creating dummy column for categorical values
dummy_columns = list(data.dtypes[data.dtypes == 'category'].index)
data = pd.get_dummies(data, columns=dummy_columns)

data.head()

# Splitting data into train and test sets
x,y = data.values,labels
x_train,x_test,y_train,y_test = train_test_split(x,y)

print("Please, wait, training takes 2-3 minutes")
# Training
model = xgb.XGBClassifier(
    objective='reg:logistic'
)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred.round())
print(acc, '\n')

model.save_model('model.bst')

# IMPORTANT: You can only render HTML witwidget() in a browser and not in a Python console/editor environment.

num_wit_examples = 500
test_examples = np.hstack((x_test[:num_wit_examples],y_test[:num_wit_examples].reshape(-1,1)))

# create custom predict because xbg.predict_proba requires a numpy array and not a list
def custom_predict(examples_to_infer):
    model_ins = np.array(examples_to_infer)
    preds = model.predict_proba(model_ins)
    return preds

config_builder = (WitConfigBuilder(test_examples.tolist(), data.columns.tolist() + ['mortgage_status'])
  .set_custom_predict_fn(custom_predict)
  .set_target_feature('mortgage_status')
  .set_label_vocab(['denied', 'approved']))
a = WitWidget(config_builder, height=800)
a.render() # only works in a notebook