# Vertex AI LABS

This repository contains sample codes for Vertex AI, including pipelines, metadata and more.


## Lab 1: Pipeline with custom containers

This pipeline uses a public dataset at:
[gs://financial_fraud_detection/fraud_data_kaggle.csv](gs://financial_fraud_detection/fraud_data_kaggle.csv)

Output of pipeline with custom containers:
![Vertex pipelines result](1-pipeline-custom-xgb/pipeline_custom.png)


## Lab 2: Pipeline with GCP operators

Training a tabular detaset (fraud detection) with AutoML.

Output of pipeline with GCP components:
![Vertex pipelines result](2-pipeline-gcp-operators/pipeline_gccaip.png)


## Lab 3: 3-step pipeline with lightweight Python components and TensorFlow

Demo code of a production pipeline using Vertex Pipelines, and training a model using lightweight python components:

* **Preprocess component:** Load from BigQuery using tensorflow_io
* **Train component:** custom train using a Keras model with 4 layers. 
* **Upload and deploy component:** upload and deploy into an endpoint in Vertex.

To install the proper libraries:

```bash
gsutil cp gs://cloud-aiplatform-pipelines/releases/latest/kfp-1.5.0rc5.tar.gz .
gsutil cp gs://cloud-aiplatform-pipelines/releases/latest/aiplatform_pipelines_client-0.1.0.caip20210415-py3-none-any.whl 

python3 -m pip install google-cloud-aiplatform
python3 -m pip install kfp-1.5.0rc5.tar.gz --upgrade
python3 -m pip install aiplatform_pipelines_client-0.1.0.caip20210415-py3-none-any.whl  --upgrade
```

![Vertex pipelines result](3-pipeline-lwpython-tf/pipeline_lwpython.png)


## Lab 4: 2-step pipeline with lightweight Python components and XGB

Demo code of a production pipeline with the following services:

* Vertex Pipelines
* Vertex ML Metadata

![Vertex pipelines result](4-pipeline-lwpython-xgb/pipeline_lwpython_xgb.png)


## Lab 5: simple Vertex custom training job (Iris dataset)

Simple Vertex custom training job, usingTensorFlow pre-built custom containers (for training and serving) and the [tabular iris dataset](https://archive.ics.uci.edu/ml/datasets/iris).

For more information about custom training in Vertex, visit the [official documentation](https://cloud.google.com/vertex-ai/docs/training/custom-training) and [this github repo](https://github.com/rafaelsf80/vertex-custom-training)


## Lab 6: Feature Store

Create a Managed Feature Store, importaing data and batch serving.
