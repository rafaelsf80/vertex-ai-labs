FROM python:3.7-slim AS builder

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY deploy_script.py /app/deploy_script.py