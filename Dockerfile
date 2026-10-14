# ===============================================================
# Stage 1: BUILDER
# Compile and install all Python dependencies (NumPy/SciPy/rdtools)
# We keep build tools only in this stage to reduce final image size
# ===============================================================
FROM python:3.11-slim-bookworm AS builder

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
COPY requirements.txt requirements_pkg.txt ./

# Install Python packages
# --no-cache-dir avoids pip cache which can add hundreds of MB
# Both files in ONE resolve, so the extra packages cannot pull a newer numpy / scipy over the
# pins in requirements.txt (two separate `--prefix` installs do not see each other and mixed
# scipy builds end in "cannot import name '_promote'").
# requirements_pkg.txt = PV-Copilot package-backed analysis (analysis_utils_pkg.py); rdtools 2.1.8
# goes in with --no-deps so it does not pull the 190 MB xgboost wheel (xgboost-cpu instead).
RUN pip install --prefix=/install --no-cache-dir -r requirements.txt -r requirements_pkg.txt && \
    pip install --prefix=/install --no-cache-dir --no-deps rdtools==2.1.8


# ===============================================================
# Stage 2: RUNTIME
# Minimal runtime image
# Only copy installed packages and app code
# ===============================================================
FROM python:3.11-slim-bookworm

WORKDIR /app

# Runtime needs only the shared libraries, not compilers or -dev headers
# (those live in the builder stage). Default Debian mirror with retries:
# the old snapshot.debian.org source was slow / rate-limited and broke builds,
# and its bookworm packages no longer matched the newer (trixie) base image.
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=5 \
        libgomp1 \
        libopenblas0 \
        liblapack3 && \
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
# 1 worker + 16 gthread threads on a small dyno (was 4: a load test with
# 10 concurrent users showed requests queueing behind slow LLM calls, which
# hold a thread while they wait on the network but use no CPU):
#   - One Python process keeps memory usage well under 512 MB.
#   - 16 threads handle concurrent HTTP requests (Dash callbacks return
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

# Exec form (JSON) so gunicorn is PID 1 and receives Heroku's SIGTERM directly
# (lets usage_log flush its last events on shutdown). sh -c expands $PORT.
# Shared on-disk store for background-job state (PVPRO, code export, analyze
# status), so a poll that lands on the other worker process still finds the
# job. Required for --workers > 1 (see the note at the top of pvcopilot.py).
ENV PVPRO_DISKCACHE_DIR=/tmp/pvpro-jobs

# 2 worker processes x 8 threads: two CPU-bound analyses (filtering, YoY,
# PVPRO) can run truly in parallel (one Python process is limited to ~1 core
# by the GIL); Performance-M's 2.5 GB RAM holds both comfortably.
CMD ["sh", "-c", "exec gunicorn index:server -k gthread --workers 2 --threads 8 --timeout 600 --bind 0.0.0.0:${PORT:-8000}"]
