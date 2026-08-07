# Docker development setup

The repository provides:

- `Dockerfile` for the unified Python API service;
- `lex-legal_frontend/Dockerfile` for the Vue/Vite development server;
- `docker-compose.yml` for the complete two-container application.

## Services and ports

| Compose service | Application | Host port | Container command |
|---|---|---:|---|
| `lexlegal-api` | Compliance query, SSE, authentication, and document import | 8008 | `uvicorn app.api.lex_legal_api:app --host 0.0.0.0 --port 8008 --reload` |
| `lexlegal-web` | Vue/Vite frontend | 5176 | `npm run dev -- --host 0.0.0.0 --port 5176` |

The repository is bind-mounted into the development containers. The Python
virtual environment is installed at `/opt/venv`, outside the `/app` mount, and
the frontend uses an anonymous `/app/node_modules` volume. This prevents either
dependency directory from being hidden by the source-code bind mounts.

## Start the application

The Compose file connects to an external Docker network named `dips-local` so
existing Elasticsearch, Milvus, MongoDB, MinIO, model, and related containers
can share DNS names with Lex-Legal.

Create the network once if it does not already exist:

```bash
docker network inspect dips-local >/dev/null 2>&1 \
  || docker network create dips-local
```

Validate and start the services:

```bash
docker compose config --quiet
docker compose up --build
```

Open:

- frontend: `http://localhost:5176/`
- unified API documentation: `http://localhost:8008/docs`

Stop the application:

```bash
docker compose down
```

## Environment endpoints

Compose loads the repository `.env` into the backend container. Any service
running in another container should be addressed by its Docker Compose service
name on `dips-local`, for example:

```dotenv
ES_URL=http://elasticsearch:9200
MILVUS_URL=http://milvus:19530
MONGO_BASE_URL=mongodb://mongodb:27017
MINIO_ENDPOINT=minio:9000
EMBEDDING_HOST=tei-embedding
```

The exact names must match the other services attached to `dips-local`.
`localhost` and `127.0.0.1` inside a container refer to that same container, not
the Docker host or a neighboring service.

When a required dependency runs directly on the Docker host rather than in a
container, use `host.docker.internal` where supported. Linux installations may
also need this service setting:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

Do not copy secrets into a Dockerfile or commit `.env`. The backend
`.dockerignore` explicitly excludes `.env` from the image build context.

## Production note

The supplied Compose file is a development configuration: it uses bind mounts,
Vite's development server, Uvicorn `--reload`, and in-memory task/SSE state.
For production, remove `--reload`, build static frontend assets, serve them from
an HTTP server, use explicit CORS origins, and replace process-local background
tasks/SSE queues with durable shared infrastructure.
