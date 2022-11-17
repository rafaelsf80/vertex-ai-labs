# NEW: https://cloud.google.com/vertex-ai/docs/experiments/tensorboard-profiler
# https://www.machinelearningnuggets.com/tensorboard-tutorial/
# https://neptune.ai/blog/tensorboard-tutorial
# 
# #!/usr/bin/env python

import tensorflow as tf
import argparse
import os
from google.cloud.aiplatform.training_utils import cloud_profiler
import time

"""Train an mnist model and use cloud_profiler for profiling."""

def _create_model():
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )
    return model


def main(args):
    strategy = None
    if args.distributed:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if args.distributed:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            model = _create_model()
            model.compile(
                optimizer="adam",
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=["accuracy"],
            )
    else:
        model = _create_model()
        model.compile(
            optimizer="adam",
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=["accuracy"],
        )

    # Initialize the profiler.
    cloud_profiler.init()

    # Use AIP_TENSORBOARD_LOG_DIR to update where logs are written to.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=os.environ["AIP_TENSORBOARD_LOG_DIR"], histogram_freq=1
    )

    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        verbose=0,
        callbacks=[tensorboard_callback],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epochs to run model."
    )
    parser.add_argument(
        "--distributed", action="store_true", help="Use MultiWorkerMirroredStrategy"
    )
    args = parser.parse_args()
    main(args)