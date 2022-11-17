import logging

logger = logging.getLogger("logger")
logging.basicConfig(level=logging.INFO)

import collections
import pandas as pd
import time
from json import dumps

collections.Iterable = collections.abc.Iterable

from google.cloud import aiplatform as vertex_ai

from helpers import preprocess, get_pipeline, train_pipeline, evaluate_model, get_training_split, save_model

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
REGION = 'europe-west4'
EXPERIMENT_NAME = 'exp-lineage'
EXPERIMENT_RUN_NAME = 'run-8'

TARGET = "category"
TARGET_LABELS = ["b", "t", "e", "m"]
PREPROCESS_EXECUTION_NAME = "preprocess"
COLUMN_NAMES = [
    "id",
    "title",
    "url",
    "publisher",
    "category",
    "story",
    "hostname",
    "timestamp",
]
DELIMITER = "	"
INDEX_COL = 0

TRAIN_EXECUTION_NAME = "train"
FEATURES = "title"
TEST_SIZE = 0.2
SEED = 8

vertex_ai.init(
    project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME
)
run = vertex_ai.start_run(EXPERIMENT_RUN_NAME)

# Create Dataset artifact
# Raw dataset: wget https://archive.ics.uci.edu/ml/machine-learning-databases/00359/NewsAggregatorDataset.zip
DATASET_NAME = "news_corpora"
DATASET_URI = "gs://argolis-rafaelsanchez-ml-dev/experiments/raw/newsCorpora.csv"
raw_dataset_artifact = vertex_ai.Artifact.create(
    schema_title="system.Dataset", display_name=DATASET_NAME, uri=DATASET_URI
)


##### PART 1: PREPROCESSING

PREPROCESSED_DATASET_NAME = f"preprocessed_{DATASET_NAME}"
PREPROCESSED_DATASET_URI = (
    f"gs://argolis-rafaelsanchez-ml-dev/experiments/preprocess/{PREPROCESSED_DATASET_NAME}.csv"
)


with vertex_ai.start_execution(
    schema_title="system.ContainerExecution", display_name=PREPROCESS_EXECUTION_NAME
) as exc:
    logging.info(f"Start {PREPROCESS_EXECUTION_NAME} execution.")
    exc.assign_input_artifacts([raw_dataset_artifact])

    # Log preprocessing params --------------------------------------------------
    logging.info("Log preprocessing params.")
    vertex_ai.log_params(
        {
            "delimiter": DELIMITER,
            "features": dumps(COLUMN_NAMES),
            "index_col": INDEX_COL,
        }
    )

    # Preprocessing ------------------------------------------------------------
    logging.info("Preprocessing.")
    raw_df = pd.read_csv(
        raw_dataset_artifact.uri,
        delimiter=DELIMITER,
        names=COLUMN_NAMES,
        index_col=INDEX_COL,
    )
    preprocessed_df = preprocess(raw_df, "title")
    preprocessed_df.to_csv(PREPROCESSED_DATASET_URI, sep=",")

    # Log preprocessing metrics and store dataset artifact ---------------------
    logging.info(f"Log preprocessing metrics and {PREPROCESSED_DATASET_NAME} dataset.")
    vertex_ai.log_metrics(
        {
            "n_records": preprocessed_df.shape[0],
            "n_columns": preprocessed_df.shape[1],
        },
    )

    preprocessed_dataset_metadata = vertex_ai.Artifact.create(
        schema_title="system.Dataset",
        display_name=PREPROCESSED_DATASET_NAME,
        uri=PREPROCESSED_DATASET_URI,
    )
    exc.assign_output_artifacts([preprocessed_dataset_metadata])




##### PART 2: TRAINING

TRAINED_MODEL_URI = f"gs://argolis-rafaelsanchez-ml-dev/experiments/deliverables/models"
MODEL_NAME =f"{EXPERIMENT_NAME}-model"

SERVE_IMAGE = vertex_ai.helpers.get_prebuilt_prediction_container_uri(
    framework="sklearn", framework_version="1.0", accelerator="cpu"
)

with vertex_ai.start_execution(
    schema_title="system.ContainerExecution", display_name=TRAIN_EXECUTION_NAME
) as exc:

    exc.assign_input_artifacts([preprocessed_dataset_metadata])

    # Get training and testing data
    logging.info("Get training and testing data.")
    x_train, x_val, y_train, y_val = get_training_split(
        preprocessed_df[FEATURES],
        preprocessed_df[TARGET],
        test_size=TEST_SIZE,
        random_state=SEED,
    )
    # Get model pipeline
    logging.info("Get model pipeline.")
    pipeline = get_pipeline()

    # Log training param -------------------------------------------------------

    # Log data parameters
    logging.info("Log data parameters.")
    vertex_ai.log_params(
        {
            "target": TARGET,
            "features": FEATURES,
            "test_size": TEST_SIZE,
            "random_state": SEED,
        }
    )

    # Log pipeline parameters
    logging.info("Log pipeline parameters.")
    vertex_ai.log_params(
        {
            "pipeline_steps": dumps(
                {step[0]: str(step[1].__class__.__name__) for step in pipeline.steps}
            )
        }
    )

    # Training -----------------------------------------------------------------

    # Train model pipeline
    logging.info("Train model pipeline.")
    train_start = time.time()
    trained_pipeline = train_pipeline(pipeline, x_train, y_train)
    train_end = time.time()

    # Evaluate model
    logging.info("Evaluate model.")
    summary_metrics, classification_metrics = evaluate_model(
        trained_pipeline, x_val, y_val
    )

    # Log training metrics and store model artifact ----------------------------

    # Log training metrics
    logging.info("Log training metrics.")
    vertex_ai.log_metrics(summary_metrics)
    vertex_ai.log_classification_metrics(
        labels=classification_metrics["labels"],
        matrix=classification_metrics["matrix"],
        display_name="my-confusion-matrix",
    )

    # Generate first ten predictions
    logging.info("Generate prediction sample.")
    prediction_sample = trained_pipeline.predict(x_val)[:10]
    print("prediction sample:", prediction_sample)

    # Upload Model on Vertex AI
    logging.info("Upload Model on Vertex AI.")
    loaded = save_model(trained_pipeline, "model.joblib")
    if loaded:
        model = vertex_ai.Model.upload(
            serving_container_image_uri=SERVE_IMAGE,
            artifact_uri=TRAINED_MODEL_URI,
            display_name=MODEL_NAME,
        )

    exc.assign_output_artifacts([model])