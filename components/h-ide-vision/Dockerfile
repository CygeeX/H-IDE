FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
RUN pip install --no-cache-dir uv==0.9.15
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY . .
RUN uv sync --frozen --no-dev && \
    useradd --create-home --uid 10001 vision && \
    mkdir -p /data/evidence && \
    chown -R vision:vision /app /data

USER vision
EXPOSE 8092
CMD ["h-ide-vision", "serve"]
