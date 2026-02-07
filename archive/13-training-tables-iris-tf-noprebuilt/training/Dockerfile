# gcloud builds submit --tag gcr.io/argolis-rafaelsanchez-ml-dev/caip-training 
# gcloud artifacts repositories create ml-pipelines-repo --repository-format=docker --location=europe-west4 --description="ML pipelines repository"
# gcloud auth configure-docker europe-west4-docker.pkg.dev
# gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/vertex-iris-demo

# Specifies base image and tag
FROM gcr.io/google-appengine/python
WORKDIR /root

# Installs additional packages
RUN pip3 install tensorflow-gpu tensorflow-io pyarrow

# Copies the trainer code to the docker image.
COPY train_script.py /root/train_script.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "train_script.py"]