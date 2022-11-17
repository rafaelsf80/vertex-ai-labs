
import joblib
import pandas as pd
import tempfile
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import string

TARGET_LABELS = ["b", "t", "e", "m"]

def preprocess(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    Preprocess text
    Args:
        df: The DataFrame to preprocesss
        text_col: The text column name
    Returns:
        preprocessed_df: The datafrane with text in lowercase and without punctuation
    """
    preprocessed_df = df.copy()
    preprocessed_df[text_col] = preprocessed_df[text_col].apply(lambda x: x.lower())
    preprocessed_df[text_col] = preprocessed_df[text_col].apply(
        lambda x: x.translate(str.maketrans("", "", string.punctuation))
    )
    return preprocessed_df


def get_training_split(
    x: pd.DataFrame, y: pd.Series, test_size: float, random_state: int
) -> (pd.DataFrame, pd.Series, pd.DataFrame, pd.Series):
    """
    Splits data into training and testing sets
    Args:
        x: The data to be split
        y: The labels to be split
        test_size: The proportion of the data to be reserved for testing
        random_state: The seed used by the random number generator
    Returns:
        x_train: The training data
        x_test: The testing data
        y_train: The training labels
        y_test: The testing labels
    """

    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )
    return x_train, x_val, y_train, y_val


def get_pipeline():
    """
    Get the model
    Args:
        None
    Returns:
        model: The model
    """
    model = Pipeline(
        [
            ("vect", CountVectorizer()),
            ("tfidf", TfidfTransformer()),
            ("clf", MultinomialNB()),
        ]
    )
    return model


def train_pipeline(model: Pipeline, X_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """
    Train the model
    Args:
        model: The model to train
        X_train: The training data
        y_train: The training labels
    Returns:
        model: The trained model
    """
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: Pipeline, X_test: pd.Series, y_test: pd.Series) -> float:
    """
    Evaluate the model
    Args:
        model: The model to evaluate
        X_test: The testing data
        y_test: The testing labels
    Returns:
        score: The accuracy of the model
    """
    # Evaluate model
    y_pred = model.predict(X_test)

    # Store evaluation metrics
    summary_metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 5),
        "precision": round(precision_score(y_test, y_pred, average="weighted"), 5),
        "recall": round(recall_score(y_test, y_pred, average="weighted"), 5),
    }
    classification_metrics = {
        "matrix": confusion_matrix(y_test, y_pred, labels=TARGET_LABELS).tolist(),
        "labels": TARGET_LABELS,
    }

    return summary_metrics, classification_metrics


def save_model(model: Pipeline, save_path: str) -> int:
    try:
        with tempfile.NamedTemporaryFile() as tmp:
            joblib.dump(model, filename=save_path)
            print("Stop here and upload the model to GCS")
            #! gsutil cp {tmp.name} {save_path}/model.joblib
    except RuntimeError as error:
        print(error)
    return 1