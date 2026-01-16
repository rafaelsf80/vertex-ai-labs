import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import urllib.request
import os
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def read_data(uri):
    """
    Read data and process it for training.
    """
    dataset_path = "auto-mpg.data"
    if not os.path.exists(dataset_path):
        urllib.request.urlretrieve(uri, dataset_path)
        
    column_names = [
        "MPG", "Cylinders", "Displacement", "Horsepower", 
        "Weight", "Acceleration", "Model Year", "Origin"
    ]
    
    # Read the dataset
    dataset = pd.read_csv(
        dataset_path,
        names=column_names,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )
    
    # Map Origin
    dataset["Origin"] = dataset["Origin"].map(
        lambda x: {1: "USA", 2: "Europe", 3: "Japan"}.get(x)
    )
    
    # Drop any NaNs (including those from Horsepower '?' or failed mapping)
    dataset = dataset.dropna()
    
    # Create dummy variables with explicit float dtype
    dataset = pd.get_dummies(dataset, prefix="", prefix_sep="", dtype=float)
    
    # Ensure all columns are numeric
    dataset = dataset.apply(pd.to_numeric, errors='coerce')
    dataset = dataset.dropna()
    
    return dataset


def train_test_split(dataset, split_frac=0.8, random_state=0):
    """
    Split data into train and test sets.
    """
    train_dataset = dataset.sample(frac=split_frac, random_state=random_state)
    test_dataset = dataset.drop(train_dataset.index)
    
    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")

    return train_dataset, test_dataset, train_labels, test_labels


def normalize_dataset(train_dataset, test_dataset):
    """
    Normalize data using training set statistics.
    """
    train_stats = train_dataset.describe().transpose()

    def norm(x):
        # Prevent division by zero if std is 0 or NaN
        std = train_stats["std"].fillna(1.0).replace(0, 1.0)
        mean = train_stats["mean"].fillna(0)
        return (x - mean) / std

    normed_train_data = norm(train_dataset).fillna(0)
    normed_test_data = norm(test_dataset).fillna(0)

    return normed_train_data, normed_test_data


def build_model(num_units, dropout_rate):
    """
    Build Scikit-learn MLPRegressor.
    """
    # Note: Scikit-learn MLPRegressor doesn't have a direct dropout parameter.
    # We use alpha for regularization which is a similar concept.
    model = MLPRegressor(
        hidden_layer_sizes=(num_units, num_units),
        activation="relu",
        solver="adam",
        max_iter=1000,
        random_state=42,
        alpha=dropout_rate # Using dropout_rate as alpha for regularization
    )
    return model


def train(
    train_data,
    train_labels,
    num_units=16,
    activation="relu",
    epochs=10,
    dropout_rate=0.1,
):
    """
    Train the model and return it with a Keras-like history object.
    Matches the call signature in experiments-simple.py.
    """
    model = build_model(num_units, dropout_rate)
    
    # Scikit-learn uses max_iter instead of epochs for iteration count
    model.set_params(max_iter=epochs)
    
    # Ensure data is clean of NaNs
    train_data = train_data.fillna(0)
    train_labels = train_labels.fillna(0)
    
    model.fit(train_data, train_labels)
    
    # Calculate metrics for logging
    train_preds = model.predict(train_data)
    mae = mean_absolute_error(train_labels, train_preds)
    mse = mean_squared_error(train_labels, train_preds)
    
    # Define a simple class to mimic Keras history
    class History:
        def __init__(self, mae, mse):
            self.history = {
                "mae": [mae],
                "mse": [mse],
                "loss": [mse]
            }
            
    return model, History(mae, mse)