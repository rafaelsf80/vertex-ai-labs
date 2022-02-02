from google.cloud import aiplatform

MY_PROJECT = 'windy-site-254307'
EXPERIMENT_NAME = 'rafa-experiment-test'
REGION = 'europe-west4'


aiplatform.init(project=MY_PROJECT, location=REGION, experiment=EXPERIMENT_NAME)

parameters = [
    {"num_units": 16, "epochs": 3, "dropout_rate": 0.1},
    {"num_units": 16, "epochs": 10, "dropout_rate": 0.1},
    {"num_units": 16, "epochs": 10, "dropout_rate": 0.2},
    {"num_units": 32, "epochs": 10, "dropout_rate": 0.1},
    {"num_units": 32, "epochs": 10, "dropout_rate": 0.2},
]

for i, params in enumerate(parameters):
    aiplatform.start_run(run=f"auto-mpg-local-run-{i}")
    aiplatform.log_params(params)
    # model, history = train(
    #     normed_train_data,
    #     train_labels,
    #     num_units=params["num_units"],
    #     activation="relu",
    #     epochs=params["epochs"],
    #     dropout_rate=params["dropout_rate"],
    # )
    # aiplatform.log_metrics(
    #     {metric: values[-1] for metric, values in history.history.items()}
    # )

    #loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    aiplatform.log_metrics({"eval_loss": 23, "eval_mae": 23, "eval_mse": 23})


experiment_df = aiplatform.get_experiment_df()
experiment_df