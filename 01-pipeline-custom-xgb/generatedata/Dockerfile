FROM python:3.7-slim AS builder

COPY requirements.txt .

RUN pip install -r requirements.txt

WORKDIR /app

COPY generatedata_script.py /app/generatedata_script.py
