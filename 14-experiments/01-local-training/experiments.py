import matplotlib.pyplot as plt
import pandas as pd
from google.cloud import aiplatform as vertex_ai

from helpers import read_data, train_test_split, normalize_dataset, build_model, train

PROJECT_ID = 'argolis-rafaelsanchez-ml-dev'
REGION = 'europe-west4'
EXPERIMENT_NAME = 'exp-local-training'

vertex_ai_tb = vertex_ai.Tensorboard.create(location=REGION)
vertex_ai.init(project=PROJECT_ID, location=REGION, experiment=EXPERIMENT_NAME, experiment_tensorboard=vertex_ai_tb)

# Define experiment parameters
parameters = [
    {"num_units": 16, "dropout_rate": 0.1, "epochs": 3},
    {"num_units": 16, "dropout_rate": 0.1, "epochs": 10},
    {"num_units": 16, "dropout_rate": 0.2, "epochs": 10},
    {"num_units": 32, "dropout_rate": 0.1, "epochs": 10},
    {"num_units": 32, "dropout_rate": 0.2, "epochs": 10},
]

# Read data
dataset = read_data(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# Split data
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset)

# Normalize data
normed_train_data, normed_test_data = normalize_dataset(train_dataset, test_dataset)

# Run experiments
for i, params in enumerate(parameters):

    # Initialize Vertex AI Experiment run
    vertex_ai.start_run(run=f"auto-mpg-local-run-{i}")

    # Log training parameters
    vertex_ai.log_params(params)

    # Build model
    model = build_model(
        num_units=params["num_units"], dropout_rate=params["dropout_rate"]
    )

    # Train model
    history = train(
        model,
        normed_train_data,
        train_labels,
        epochs=params["epochs"],
    )

    # Log additional parameters
    vertex_ai.log_params(history.params)

    # Log metrics per epochs
    for idx in range(0, history.params["epochs"]):
        vertex_ai.log_time_series_metrics(
            {
                "train_mae": history.history["mae"][idx],
                "train_mse": history.history["mse"][idx],
            }
        )

    # Log final metrics
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    vertex_ai.log_metrics({"eval_loss": loss, "eval_mae": mae, "eval_mse": mse})

    vertex_ai.end_run()

experiment_df = vertex_ai.get_experiment_df()
print(experiment_df.T)

plt.rcParams["figure.figsize"] = [15, 5]

ax = pd.plotting.parallel_coordinates(
    experiment_df.reset_index(level=0),
    "run_name",
    cols=[
        "param.num_units",
        "param.dropout_rate",
        "param.epochs",
        "metric.eval_loss",
        "metric.eval_mse",
        "metric.eval_mae",
    ],
    color=["blue", "green", "pink", "red"],
)
ax.set_yscale("symlog")
ax.legend(bbox_to_anchor=(1.0, 0.5))

ax.sh