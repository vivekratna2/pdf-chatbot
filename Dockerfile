FROM python:3.12-slim

WORKDIR /app

COPY setup.py .

RUN pip3 install --upgrade pip && pip3 install --no-cache-dir --compile --editable .

COPY src/ ./src/

