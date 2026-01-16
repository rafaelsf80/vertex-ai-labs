from google.cloud import aiplatform

from helpers import read_data, train_test_split, normalize_dataset, build_model, train


MY_PROJECT = 'argolis-rafaelsanchez-ml-dev'
EXPERIMENT_NAME = 'rafa-experiment-simple'
REGION = 'europe-west4'

aiplatform.init(project=MY_PROJECT, location=REGION, experiment=EXPERIMENT_NAME)

parameters = [
    {"num_units": 16, "epochs": 3, "dropout_rate": 0.1},
    {"num_units": 16, "epochs": 10, "dropout_rate": 0.1},
    {"num_units": 16, "epochs": 10, "dropout_rate": 0.2},
    {"num_units": 32, "epochs": 10, "dropout_rate": 0.1},
    {"num_units": 32, "epochs": 10, "dropout_rate": 0.2},
]

# Read data
# https://www.kaggle.com/datasets/yasserh/auto-mpg-dataset
dataset = read_data(
    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

# Split data
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset)

# Normalize data
normed_train_data, normed_test_data = normalize_dataset(train_dataset, test_dataset)

for i, params in enumerate(parameters):
    aiplatform.start_run(run=f"auto-mpg-local-run4-{i}")
    aiplatform.log_params(params)
    model, history = train(
        normed_train_data,
        train_labels,
        num_units=params["num_units"],
        activation="relu",
        epochs=params["epochs"],
        dropout_rate=params["dropout_rate"],
    )
    aiplatform.log_metrics(
        {metric: values[-1] for metric, values in history.history.items()}
    )

    # Evaluate using Scikit-learn
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    test_preds = model.predict(normed_test_data)
    eval_mae = mean_absolute_error(test_labels, test_preds)
    eval_mse = mean_squared_error(test_labels, test_preds)
    
    aiplatform.log_metrics({
        "eval_loss": eval_mse, 
        "eval_mae": eval_mae, 
        "eval_mse": eval_mse
    })

# recovers dataframe
experiment_df = aiplatform.get_experiment_df(EXPERIMENT_NAME)
print(experiment_df)