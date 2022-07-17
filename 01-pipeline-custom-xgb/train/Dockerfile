FROM python:3.7-slim AS builder

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY trainer_script.py /app/trainer_script.py