import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.python.keras import Sequential, layers
from tensorflow.python.keras.utils import data_utils

def read_data(uri):
    """
    Read data
    Args:
        uri: path to data
    Returns:
        pandas dataframe
    """
    dataset_path = data_utils.get_file("auto-mpg.data", uri)
    column_names = [
        "MPG",
        "Cylinders",
        "Displacement",
        "Horsepower",
        "Weight",
        "Acceleration",
        "Model Year",
        "Origin",
    ]
    raw_dataset = pd.read_csv(
        dataset_path,
        names=column_names,
        na_values="?",
        comment="\t",
        sep=" ",
        skipinitialspace=True,
    )
    dataset = raw_dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        lambda x: {1: "USA", 2: "Europe", 3: "Japan"}.get(x)
    )
    dataset = pd.get_dummies(dataset, prefix="", prefix_sep="")
    return dataset


def train_test_split(dataset, split_frac=0.8, random_state=0):
    """
    Split data into train and test
    Args:
        dataset: pandas dataframe
        split_frac: fraction of data to use for training
        random_state: random seed
    Returns:
        train and test dataframes
    """
    train_dataset = dataset.sample(frac=split_frac, random_state=random_state)
    test_dataset = dataset.drop(train_dataset.index)
    train_labels = train_dataset.pop("MPG")
    test_labels = test_dataset.pop("MPG")

    return train_dataset, test_dataset, train_labels, test_labels


def normalize_dataset(train_dataset, test_dataset):
    """
    Normalize data
    Args:
        train_dataset: pandas dataframe
        test_dataset: pandas dataframe

    Returns:

    """
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()

    def norm(x):
        return (x - train_stats["mean"]) / train_stats["std"]

    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    return normed_train_data, normed_test_data


def build_model(num_units, dropout_rate):
    """
    Build model
    Args:
        num_units: number of units in hidden layer
        dropout_rate: dropout rate
    Returns:
        compiled model
    """
    model = Sequential(
        [
            layers.Dense(
                num_units,
                activation="relu",
                input_shape=[9],
            ),
            layers.Dropout(rate=dropout_rate),
            layers.Dense(num_units, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(loss="mse", optimizer="adam", metrics=["mae", "mse"])
    return model


def train(
    model,
    train_data,
    train_labels,
    validation_split=0.2,
    epochs=10,
):
    """
    Train model
    Args:
        train_data: pandas dataframe
        train_labels: pandas dataframe
        model: compiled model
        validation_split: fraction of data to use for validation
        epochs: number of epochs to train for
    Returns:
        history
    """
    history = model.fit(
        train_data, train_labels, epochs=epochs, validation_split=validation_split
    )

    return history