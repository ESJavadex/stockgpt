# Dockerfile for StockGPT Flask app
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements if present, else install inline
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt || true

# Copy the .env file
COPY .env ./

# Copy the rest of the application
COPY . .

EXPOSE 5003

CMD ["python", "scrap.py"]
