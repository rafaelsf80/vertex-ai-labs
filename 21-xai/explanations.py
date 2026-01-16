""" Train a Keras model with explainability in Vertex AI """

MY_PROJECT = "argolis-rafaelsanchez-ml-dev"
MY_STAGING_BUCKET = "gs://argolis-vertex-europewest4"  
LOCATION='europe-west4'

import uuid

import tensorflow as tf
from google.cloud import aiplatform
from tabulate import tabulate

aiplatform.init(project=MY_PROJECT, staging_bucket=MY_STAGING_BUCKET, location=LOCATION)

job = aiplatform.CustomTrainingJob(
    display_name=f"explainable-ai-custom-tabular-nb-{uuid.uuid4()}",
    script_path="training_script.py",
    container_uri="europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-6:latest",
    requirements=[
        "tensorflow_datasets",
        "explainable-ai-sdk",
    ],
    model_serving_container_image_uri="europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-6:latest",
)

model = job.run(
    model_display_name="mbsdk-explainable-tabular-model",
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    args=["--epochs=50", "--distribute=single"],
)

# Get info about the Custom Job
print(
    f"Display Name:\t{job.display_name}\n"
    f"Resource Name:\t{job.resource_name}\n"
    f"Current State:\t{job.state.name}\n"
)

#model = aiplatform.Model('projects/655797269815/locations/us-central1/models/2025784203479613440')
#Get path to saved model in GCS
output_dir = model._gca_resource.artifact_uri

# #################################
# # Build the Explanation Metadata and Parameters. This is needed to deploy
# #################################
loaded = tf.keras.models.load_model(output_dir)

serving_input = list(
    loaded.signatures["serving_default"].structured_input_signature[1].keys()
)[0]
serving_output = list(loaded.signatures["serving_default"].structured_outputs.keys())[0]
feature_names = [
     "crim",
     "zn",
     "indus",
     "chas",
     "nox",
     "rm",
     "age",
     "dis",
     "rad",
     "tax",
     "ptratio",
     "b",
     "lstat",
]

XAI = "shapley"
if XAI == "shapley":
    PARAMETERS = {"sampled_shapley_attribution": {"path_count": 10}}
if XAI == "ig":
    PARAMETERS = {"integrated_gradients_attribution": {"step_count": 50}}
if XAI == "xrai":
    PARAMETERS = {"xrai_attribution": {"step_count": 50}}

explain_params = aiplatform.explain.ExplanationParameters(PARAMETERS)


input_metadata = {
    "input_tensor_name": serving_input,
    "encoding": "BAG_OF_FEATURES",
    "modality": "numeric",
    "index_feature_mapping": feature_names,
}
output_metadata = {"output_tensor_name": serving_output}

input_metadata = aiplatform.explain.ExplanationMetadata.InputMetadata(input_metadata)
output_metadata = aiplatform.explain.ExplanationMetadata.OutputMetadata(output_metadata)

explain_metadata = aiplatform.explain.ExplanationMetadata(
    inputs={"features": input_metadata}, outputs={"medv": output_metadata}
)
#################################
# END Build explanations metadata
#################################


# Deploy the model with model explanations enabled
endpoint = model.deploy(
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
    explanation_metadata=explain_metadata,
    explanation_parameters=explain_params,
)

print(f"Endpoint resource name: {endpoint.resource_name}")
print(
    f"\nTo use this endpoint in the future:\nendpoint = aiplatform.Endpoint('{endpoint.resource_name}')"
)

#############################
## PREDICT WITH EXPLANATIONS
#############################
#endpoint = aiplatform.Endpoint('projects/655797269815/locations/us-central1/endpoints/1753360406488809472')

import numpy as np
from tensorflow.keras.datasets import boston_housing

(_, _), (x_test, y_test) = boston_housing.load_data(
    path="boston_housing.npz", test_split=0.2, seed=113
)


def scale(feature):
    max = np.max(feature)
    feature = (feature / max).astype(np.float32)
    return feature


for _ in range(13):
    x_test[_] = scale(x_test[_])
x_test = x_test.astype(np.float32)

print(x_test)
print(y_test)

print(x_test.shape, x_test.dtype, y_test.shape)


print(x_test[0])
# Get predictions
response = endpoint.explain(
    instances=[{"dense_input": s.tolist()} for s in [x_test[0]]]
)

# Checkout feature attributions
import numpy
test_data = x_test[0]
test_data = numpy.array([0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0.1, 0.2])
attributions = response.explanations[0].attributions[0].feature_attributions

rows = []
for i, val in enumerate(feature_names):
    rows.append([val, test_data[i], attributions["dense_input"][i]])
print(tabulate(rows, headers=["Feature name", "Feature value", "Attribution value"]))

