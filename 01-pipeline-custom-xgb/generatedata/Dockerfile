FROM google/cloud-sdk:latest
WORKDIR /app

RUN pip install pandas
RUN pip install xgboost
RUN pip install scikit-learn
RUN pip install google-cloud-storage
RUN pip install google-cloud-bigquery
RUN pip install fsspec
RUN pip install gcsfs
RUN pip install tensorflow
RUN pip install google-cloud-aiplatform

# Copies the code and credentials to the docker image.
COPY generatedata_script.py /app/generatedata_script.py
COPY keyfile.json /app/keyfile.json
RUN gcloud auth activate-service-account --key-file=/app/keyfile.json
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/keyfile.json"
