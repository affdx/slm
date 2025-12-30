# Dockerfile for FastAPI + PyTorch + MediaPipe backend
# Optimized for Sevalla deployment

# Use Python 3.11 slim image based on Debian Bookworm for better package availability
FROM python:3.11-slim-bookworm AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .

# Install all dependencies including PyTorch
# Note: On ARM64/Apple Silicon, PyTorch defaults to CPU-only
# On x86_64 (Sevalla servers), we use the CPU index for smaller images
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY configs/ ./configs/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
# Use 2 workers for better performance, adjust based on container resources
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
