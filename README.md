# AI Legal Compliance RAG Assistant

A legal Retrieval-Augmented Generation (RAG) application for assessing user
policies against EU legislation, including the GDPR, EU AI Act, EU Data Act,
Data Governance Act, Open Data Directive, and related digital regulations.

## What It Does

- Splits legal Markdown into traceable Articles, paragraphs, points, recitals,
  and parent-child records.
- Extracts legal key terms and generates BGE dense embeddings.
- Stores authoritative legal text in Elasticsearch and child vectors in
  Milvus.
- Combines Elasticsearch BM25 and Milvus semantic retrieval using RRF and a
  BGE cross-encoder reranker.
- Applies a retrieval-quality gate and performs one Tavily MCP web fallback
  when local evidence is insufficient.
- Produces source-grounded compliance assessments with legal citations, raw
  retrieved text, corrective suggestions, and warnings for open-web evidence.
- Streams workflow progress and answers from FastAPI to a Vue frontend using
  Server-Sent Events (SSE).

## Architecture

```text
Legal documents
  -> structure-aware splitting
  -> key-term extraction
  -> Elasticsearch text/BM25 index
  -> BGE embeddings
  -> Milvus vector index

User policy
  -> intent and query rewriting
  -> BM25 + semantic retrieval
  -> RRF and cross-encoder reranking
  -> evidence-quality check
  -> grounded answer with sources
```

## Run the Application

Start the import API:

```bash
uvicorn app.api.import_server:app --host 0.0.0.0 --port 8007
```

Start the query API:

```bash
uvicorn app.api.query_server:app --host 0.0.0.0 --port 8008
```

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

