# AI Legal Compliance RAG Assistant

A legal Retrieval-Augmented Generation (RAG) application for assessing user
policies against EU legislation, including the GDPR, EU AI Act, EU Data Act,
Data Governance Act, Open Data Directive, and related digital regulations.

## What It Does

- Splits legal Markdown into traceable Articles, paragraphs, points, recitals,
  and parent-child records.
- Extracts legal key terms and generates BGE dense embeddings.
- Stores authoritative legal text in Elasticsearch and child vectors in Milvus.
- Combines Elasticsearch BM25 and Milvus semantic retrieval using RRF and a
  BGE cross-encoder reranker.
- Applies a retrieval-quality gate and performs one Tavily MCP web fallback
  when local evidence is insufficient.
- Produces source-grounded compliance assessments with legal citations, raw
  retrieved text, corrective suggestions, and warnings for open-web evidence.
- Streams workflow progress and answers from FastAPI to a Vue frontend using
  Server-Sent Events (SSE).

## Architecture


![Lex-Legal_Architecture.jpg](docs/Lex-Legal_Architecture.jpg)


## Run the Application

Start the unified API:

```bash
uvicorn app.api.lex_legal_api:app --host 0.0.0.0 --port 8008
```

Swagger UI at `http://localhost:8008/docs` groups the compliance and document
import routes separately.

Start the Vue frontend:

```bash
cd lex-legal_frontend
npm ci
npm run dev -- --host 0.0.0.0 --port 5176
```

Build the legal knowledge base from a directory or one Markdown file:

```bash
python -m app.scripts.build_rag_knowledge_base ./doc/laws \
  --output-dir ./output/laws_build
```

## Run with Docker

The development Compose stack starts the unified API on port 8008 and the
Vue/Vite frontend on port 5176:

```bash
docker network inspect dips-local >/dev/null 2>&1 \
  || docker network create dips-local
docker compose up --build
```

See [docs/docker.md](docs/docker.md) for endpoint configuration, service URLs,
and production considerations.
