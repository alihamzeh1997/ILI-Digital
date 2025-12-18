FROM python:3.12-slim

# Keep Python predictable + avoid writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (minimal). build-essential can be needed for some python wheels.
RUN apt-get update \
  && apt-get install -y --no-install-recommends build-essential curl \
  && rm -rf /var/lib/apt/lists/*

# Install uv for reproducible installs from uv.lock
RUN pip install --no-cache-dir uv==0.5.7

# Copy dependency manifests first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies into the system site-packages (so CLI entrypoints like `streamlit` are on PATH)
# `uv==0.5.7` doesn't support `uv sync --system`, so we export locked requirements and install them.
RUN uv export --frozen --no-dev --format requirements-txt > requirements.txt \
  && uv pip install --system -r requirements.txt \
  && rm -f requirements.txt

# Copy the rest of the repo
COPY . .

# Streamlit defaults
EXPOSE 8501

# Streamlit reads OPENAI_API_KEY from env; you can pass it via docker-compose/.env
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]


