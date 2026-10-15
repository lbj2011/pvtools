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
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        libopenblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

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

RUN echo "deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/20250301T000000Z bookworm main" > /etc/apt/sources.list && \
    apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgomp1 \
        libopenblas-dev \
        liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application source code
# NOTE: .dockerignore should exclude large datasets
COPY . .

# Expose port (optional for documentation)
EXPOSE 8000

# Gunicorn config for a Heroku 1X dyno (512 MB RAM total).
#
# IMPORTANT -- PVPRO compute is RAM-heavy.  Each Python worker process
# carries its own copy of numpy / pandas / scipy / statsmodels / sklearn,
# which is ~150-200 MB before any data is loaded.  Running 3 workers on a
# 512 MB dyno exceeds the limit during a PVPRO run, the OOM killer
# SIGKILLs the worker mid-computation, and the polling UI sees the job
# stuck at "phase=fitting" forever because the worker thread never gets a
# chance to write its "done" update.
#
# 1 worker + 4 gthread threads is the recommended Dash topology on a
# small dyno:
#   - One Python process keeps memory usage well under 512 MB.
#   - 4 threads handle concurrent HTTP requests (Dash callbacks return
#     quickly except for PVPRO, which runs in its own background thread).
#   - With diskcache (PVPRO_DISKCACHE_DIR set), the background-job state
#     persists across requests even with the single-worker setup.
#
# --timeout 600 covers PVPRO's worst-case ~6 minute runtime on large
# datasets with non-trivial array geometry (e.g. 14 modules/string,
# 37 parallel strings).  Each window fit can take 1-3 seconds when
# p0 lands far from the converged optimum, and 100+ windows is normal
# for 8-year datasets.  Lower timeouts cause gunicorn to SIGKILL the
# worker mid-run, which manifests as a "stuck" UI to the user.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

CMD gunicorn index:server \
    -k gthread \
    --workers 1 \
    --threads 4 \
    --timeout 600 \
    --bind 0.0.0.0:${PORT:-8000}
    