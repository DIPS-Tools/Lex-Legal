FROM python:3.11-slim-bookworm

# Keep the virtual environment outside /app. In development, Docker Compose
# bind-mounts the repository at /app; an /app/.venv would otherwise be hidden.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PROJECT_ROOT=/app \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

# libgomp is required by common PyTorch/FlagEmbedding wheels. libgl and glib
# cover image/PDF dependencies that may be loaded by the import pipeline.
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

# Install locked dependencies before copying application code so dependency
# layers remain cached when only source files change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app ./app
COPY keycloak_auth ./keycloak_auth
COPY configs ./configs
COPY prompts ./prompts
COPY lex-legal_frontend/templates ./lex-legal_frontend/templates
COPY main.py ./main.py

EXPOSE 8008

CMD ["python", "-m", "uvicorn", "app.api.lex_legal_api:app", "--host", "0.0.0.0", "--port", "8008"]
