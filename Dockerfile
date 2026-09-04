FROM python:3.13-slim-bookworm AS builder

### ENVIRONMENT VARIABLES ###
ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

### SYSTEM SETUP ###
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        build-essential \
        libmagic1 \
        libmagic-dev \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        mime-support && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### PREPARE BUILD WITH NECESSARY FILES AND FOLDERS ###
WORKDIR /app

COPY ./pyproject.toml ./uv.lock ./LICENSE ./
COPY ./cat/core_plugins ./cat/core_plugins

### INSTALL DEPENDENCIES (CORE + CORE PLUGINS) ###
RUN pip install -U pip uv && \
    uv sync --frozen --no-install-project --no-upgrade --no-cache --no-dev --python /usr/local/bin/python3.13 && \
    find ./cat/core_plugins -name requirements.txt | sed 's/^/-r /' | xargs uv pip install --no-cache --no-upgrade && \
    rm -rf *.egg-info /root/.cache/pip /tmp/* /var/tmp/* && \
    uv cache clean && \
    find ./ -type d -name __pycache__ -exec rm -rf {} +

# ──────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS final

ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libmagic1 \
        poppler-utils \
        tesseract-ocr \
        libgl1 \
        mime-support && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

COPY ./cat ./cat
COPY ./data ./data
COPY ./migrations ./migrations

### SETUP ENTRYPOINT ###
COPY ./docker/entrypoint.sh ./docker/entrypoint.sh
RUN chmod +x ./docker/entrypoint.sh
ENTRYPOINT ["./docker/entrypoint.sh"]

CMD ["python", "-m", "cat.main"]