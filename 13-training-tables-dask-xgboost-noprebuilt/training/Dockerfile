# gcloud artifacts repositories create ml-pipelines-repo --repository-format=docker --location=europe-west4 --description="ML pipelines repository"
# gcloud auth configure-docker europe-west4-docker.pkg.dev
# gcloud builds submit --tag europe-west4-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-pipelines-repo/13-training-tables-xgboost-noprebuilt

FROM us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-9:latest
WORKDIR /root

# Install sudo
RUN apt-get update && apt-get -y install sudo

# Update the keyring in order to run apt-get update.
RUN rm -rf /usr/share/keyrings/cloud.google.gpg
RUN rm -rf /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt cloud-sdk main" | sudo tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Install packages (without sudo)
RUN apt-get update && apt-get install -y telnet netcat iputils-ping net-tools

# Determine the default Python version
RUN echo python3 --version

# Install Python packages using the identified version
RUN python3 -m pip install 'xgboost>=1.4.2' 'dask-ml[complete]==2022.5.27' 'dask[complete]==2022.01.0' --upgrade
RUN python3 -m pip install dask==2022.01.0 distributed==2022.01.0 bokeh==2.4.3 dask-cuda==22.2.0 click==8.0.1 --upgrade
RUN python3 -m pip install gcsfs --upgrade


# Make sure gsutil will use the default service account
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

# Copies the trainer code
RUN mkdir /root/trainer
COPY train_script.py /root/trainer/train.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "trainer/train.py"]