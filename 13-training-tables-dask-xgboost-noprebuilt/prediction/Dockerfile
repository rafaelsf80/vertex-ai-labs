# gcloud artifacts repositories create ml-workloads-qatar-prediction --repository-format=docker --location=me-central1 --description="ML pipelines repository"
# gcloud auth configure-docker me-central1-docker.pkg.dev
# gcloud builds submit --tag me-central1-docker.pkg.dev/argolis-rafaelsanchez-ml-dev/ml-workloads-qatar-prediction/13-xgboost_dask

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.8

# Install libgomp1 for XGBoost
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./app /app
# TODO: download the model from Model Registry
# Temporally load the model from local
COPY ./model /model
COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 7080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7080"]