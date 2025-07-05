FROM python:3.12-slim

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy setup.py first to leverage Docker cache
COPY setup.py .

# Install dependencies from setup.py
RUN pip install --no-cache-dir -e .

# Copy the rest of the application
COPY src ./src
COPY .env* ./

# Expose the port the app will run on
EXPOSE 8001

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser
RUN chown -R appuser:appuser /app
USER appuser
