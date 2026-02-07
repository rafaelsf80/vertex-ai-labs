import xgboost as xgb
import numpy as np

# 1. Load the model (using load_model since you used save_model in training)
bst = xgb.Booster()
bst.load_model('../prediction/model/model.bst')

# 2. Prepare Iris data (4 features: sepal length, sepal width, petal length, petal width)
# Example data for 2 flowers
data = np.array([
    [5.1, 3.5, 1.4, 0.2],
    [4.9, 3.0, 1.4, 0.2]
])

# 3. Predict
dtest = xgb.DMatrix(data)
predictions = bst.predict(dtest)

print(f"Predictions (Iris classes): {predictions}")