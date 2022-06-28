from explainable_ai_sdk.metadata.tf.v2 import SavedModelMetadataBuilder

import os
import tempfile
import argparse
from datetime import datetime

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from google.cloud import aiplatform

parser = argparse.ArgumentParser()
parser.add_argument('--project', dest='project',
                    default='my-project', type=str,
                    help='Project to report experiments to.')
parser.add_argument('--experiment', dest='experiment',
                    default='my-experiment', type=str,
                    help='Experiment to document metrics in AI platform.')
parser.add_argument('--epochs', dest='epochs',
                    default=10, type=int,
                    help='Number of epochs.')
parser.add_argument('--data-source', dest='csv_path',
                    default='data.csv', type=str,
                    help='CSV file in GCS to read from.')
args = parser.parse_args()

#Init Vertex AI experiment
aiplatform.init(project=args.project, experiment=args.experiment)
aiplatform.start_run(run=f'{args.experiment}-{str(int(datetime.utcnow().timestamp()))}')

#Log parameters
aiplatform.log_params({"epochs": args.epochs, "data-source": args.csv_path})


sample_df = pd.read_csv(args.csv_path)

target = sample_df['churned']
features = sample_df.drop(['churned'], axis=1).select_dtypes(include=['int64'])

normalizer = keras.layers.experimental.preprocessing.Normalization()
normalizer.adapt(tf.convert_to_tensor(features))

dataset = tf.data.Dataset.from_tensor_slices((features, target))

train_ds = (dataset.skip(1000)
            .batch(10, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))
val_ds = (dataset.take(1000)
          .batch(10, drop_remainder=True)
          .cache()
          .prefetch(tf.data.experimental.AUTOTUNE))

model = keras.Sequential([
        normalizer,
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['binary_accuracy', tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]
             )

history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, 
          callbacks=[keras.callbacks.TensorBoard(
              log_dir=os.getenv("AIP_TENSORBOARD_LOG_DIR"), 
              histogram_freq=1, 
              write_graph=True, write_images=True,
              profile_batch=[2,10])])

evaluation = model.evaluate(val_ds)
model.summary()

#Log metrics
aiplatform.log_metrics({"val_loss": evaluation[0], "val_accuracy": evaluation[1]})

tmpdir = tempfile.mkdtemp()
model.save(tmpdir)

# Save TF Model with Explainable metadata to GCS
builder = SavedModelMetadataBuilder(tmpdir)
builder.save_model_with_metadata(os.getenv("AIP_MODEL_DIR"))

# WARNING during BATCH PREDICTION WITH XAI:tensorflow:SavedModel saved prior to TF 2.5 detected when loading Keras model. 
# Please ensure that you are saving the model with model.save() or tf.keras.models.save_model(), *NOT* tf.saved_model.save(). 
# To confirm, there should be a file named "keras_metadata.pb" in the SavedModel directory.

# model.save(os.getenv("AIP_MODEL_DIR"))