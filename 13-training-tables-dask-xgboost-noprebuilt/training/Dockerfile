# gcloud artifacts repositories create ml-workloads-qatar-training --repository-format=docker --location=me-central1 --description="ML pipelines repository"
# gcloud auth configure-docker me-central1-docker.pkg.dev
# gcloud builds submit --tag me-central1-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-workloads-qatar-training/13-training-tables-xgboost-noprebuilt

FROM python:3.10-slim-bullseye
WORKDIR /root

# Install sudo and curl
RUN apt-get update && apt-get -y install sudo curl gnupg

# Update the keyring in order to run apt-get update.
RUN rm -rf /usr/share/keyrings/cloud.google.gpg
RUN rm -rf /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Install packages (without sudo)
RUN apt-get update && apt-get install -y telnet netcat iputils-ping net-tools

# Determine the default Python version
RUN echo python3 --version

# Install Python packages required for training
RUN python3 -m pip install 'numpy<2.0.0' 'pandas<2.0.0' 'xgboost>=1.4.2' 'dask-ml[complete]==2022.5.27' 'dask[complete]==2022.01.0' --upgrade
RUN python3 -m pip install dask==2022.01.0 distributed==2022.01.0 bokeh==2.4.3 dask-cuda==22.2.0 click==8.0.1 --upgrade
RUN python3 -m pip install gcsfs --upgrade


# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code
RUN mkdir /root/trainer
COPY train_script.py /root/trainer/train.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "trainer/train.py"]