# Use Python 3.11.13 slim base image (x86_64 for Railway)
FROM --platform=linux/amd64 python:3.11.13-slim-bookworm

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies (including git!)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    libpq-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project
COPY . .

# Expose FastAPI port
EXPOSE 8000

# IMPORTANT: your main file is app.py
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
