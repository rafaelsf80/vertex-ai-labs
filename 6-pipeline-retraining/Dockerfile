FROM gcr.io/deeplearning-platform-release/sklearn-cpu.0-23
WORKDIR /

# Copies the trainer code to the docker image.
COPY trainer /trainer

RUN pip install sklearn google-cloud-bigquery joblib pandas google-cloud-storage google-cloud-aiplatform==1.4.0 explainable-ai-sdk

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "trainer.train"]