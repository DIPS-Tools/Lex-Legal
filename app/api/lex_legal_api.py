"""Unified Lex-Legal FastAPI application."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.authentication import auth_router
from app.api.import_server import import_router
from app.api.query_server import query_router
from app.clients.es_utils import es_client_manager
from app.clients.milvus_utils import milvus_client_manager
from app.core.logger import logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Release shared asynchronous clients when the API shuts down."""
    yield
    await es_client_manager.close()
    await milvus_client_manager.close()


openapi_tags = [
    {
        "name": "Authentication APIs",
        "description": "Keycloak login and authenticated-user operations.",
    },
    {
        "name": "Legal Compliance APIs",
        "description": "Legal questions, compliance analysis, streaming, and history.",
    },
    {
        "name": "Legal Document Import APIs",
        "description": "Administrator-only legal-document ingestion and task status.",
    },
    {
        "name": "Service",
        "description": "Service availability endpoints.",
    },
]

app = FastAPI(
    title="Lex-Legal API",
    description=(
        "Unified API for legal compliance analysis and legal-document "
        "knowledge-base import."
    ),
    lifespan=lifespan,
    openapi_tags=openapi_tags,
)

configured_origins = os.getenv(
    "LEXLEGAL_CORS_ORIGINS",
    os.getenv("QUERY_CORS_ORIGINS", os.getenv("IMPORT_CORS_ORIGINS", "*")),
)
cors_origins = [
    origin.strip()
    for origin in configured_origins.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(
    auth_router,
    prefix="/api",
    tags=["Authentication APIs"],
)
app.include_router(
    query_router,
    prefix="/api/query",
    tags=["Legal Compliance APIs"],
)
app.include_router(
    import_router,
    prefix="/api/import",
    tags=["Legal Document Import APIs"],
)


@app.get("/health", tags=["Service"], summary="Service health")
async def health() -> dict[str, str]:
    """Return a lightweight response without contacting dependencies."""
    logger.info("Unified Lex-Legal API health check completed successfully.")
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.api.lex_legal_api:app",
        host="0.0.0.0",
        port=8008,
    )
