# ===============================================================
# Stage 1: BUILDER
# Compile and install all Python dependencies (NumPy/SciPy/rdtools)
# We keep build tools only in this stage to reduce final image size
# ===============================================================
FROM python:3.11-slim AS builder

# Prevent .pyc files and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required to compile scientific packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        libblas3 \
        liblapack3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy dependency file
COPY requirements.txt .

# Install Python packages
# --no-cache-dir avoids pip cache which can add hundreds of MB
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt


# ===============================================================
# Stage 2: RUNTIME
# Minimal runtime image
# Only copy installed packages and app code
# ===============================================================
FROM python:3.11-slim

WORKDIR /app

# Install runtime libraries required by NumPy/SciPy
# (Important: otherwise BLAS fallback may slow down computation)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libblas3 \
        liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
# NOTE: .dockerignore should exclude large datasets
COPY . .

# Expose port (optional for documentation)
EXPOSE 8000

# Use gthread worker for Dash concurrency
# 1 worker recommended for Heroku 1X dyno (512MB RAM)
CMD gunicorn index:server \
    -k gthread \
    --workers 3 \
    --threads 4 \
    --timeout 120 \
    --bind 0.0.0.0:${PORT:-8000}
