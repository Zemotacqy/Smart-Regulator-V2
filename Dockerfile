FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml requirements.txt uv.lock ./

# Install dependencies using uv directly into the system python environment
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Copy source code and files
COPY backend/ ./backend
COPY documents/ ./documents
COPY migrations/ ./migrations
COPY modelfiles/ ./modelfiles
COPY scripts/ ./scripts
COPY tests/ ./tests

# Expose backend port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run uvicorn server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
