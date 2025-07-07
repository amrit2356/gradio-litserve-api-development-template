# Dockerfile.litserve
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY .docker/requirements/litserve.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r litserve.txt

# Copy application code
COPY src/yolov11_service.py .

# Create directory for models
RUN mkdir -p /app/models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "src/yolov11_service.py", "--host", "0.0.0.0", "--port", "8000"]