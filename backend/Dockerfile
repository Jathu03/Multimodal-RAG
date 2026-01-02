# Use a lightweight Python image
FROM python:3.10-slim

# Install system dependencies
# FIX: Replaced 'libgl1-mesa-glx' with 'libgl1' and added 'libglib2.0-0'
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and backend code
COPY src/ src/
COPY backend/ backend/
#COPY .env .env

# Expose the port
EXPOSE 8000

# Set Python path so 'src' is importable
ENV PYTHONPATH=/app

# Command to run the API
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
