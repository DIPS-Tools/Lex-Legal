from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class EmbeddingConfig:
    """Configuration for a Hugging Face TEI embedding endpoint."""

    host: str
    port: int
    model: str
    dimension: int
    batch_size: int
    normalize: bool
    truncate: bool
    timeout: float
    max_retries: int
    retry_base_delay: float
    retry_max_delay: float
    max_concurrent_requests: int
    query_prefix: str = ""

    def __post_init__(self) -> None:
        if not self.host:
            raise ValueError("EMBEDDING_HOST must not be empty.")
        if not 1 <= self.port <= 65535:
            raise ValueError("EMBEDDING_PORT must be between 1 and 65535.")
        if self.dimension <= 0:
            raise ValueError("EMBEDDING_DIMENSION must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("EMBEDDING_BATCH_SIZE must be greater than zero.")
        if self.timeout <= 0:
            raise ValueError("EMBEDDING_TIMEOUT must be greater than zero.")
        if self.max_retries < 0:
            raise ValueError("EMBEDDING_MAX_RETRIES must not be negative.")
        if self.retry_base_delay < 0:
            raise ValueError(
                "EMBEDDING_RETRY_BASE_DELAY must not be negative."
            )
        if self.retry_max_delay < self.retry_base_delay:
            raise ValueError(
                "EMBEDDING_RETRY_MAX_DELAY must be greater than or "
                "equal to EMBEDDING_RETRY_BASE_DELAY."
            )
        if self.max_concurrent_requests <= 0:
            raise ValueError(
                "EMBEDDING_MAX_CONCURRENT_REQUESTS must be greater than zero."
            )

    @property
    def endpoint_url(self) -> str:
        """Return the base URL exposed by the TEI container."""
        return f"http://{self.host}:{self.port}"


embedding_config = EmbeddingConfig(
    host=os.getenv("EMBEDDING_HOST", "127.0.0.1"),
    port=int(os.getenv("EMBEDDING_PORT", "8083")),
    model=os.getenv("EMBEDDING_MODEL", "bge-base-en-v1.5"),
    dimension=int(os.getenv("EMBEDDING_DIMENSION", "768")),
    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "8")),
    normalize=_env_bool("EMBEDDING_NORMALIZE", True),
    truncate=_env_bool("EMBEDDING_TRUNCATE", True),
    timeout=float(os.getenv("EMBEDDING_TIMEOUT", "60")),
    max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "5")),
    retry_base_delay=float(
        os.getenv("EMBEDDING_RETRY_BASE_DELAY", "1")
    ),
    retry_max_delay=float(
        os.getenv("EMBEDDING_RETRY_MAX_DELAY", "15")
    ),
    max_concurrent_requests=int(
        os.getenv("EMBEDDING_MAX_CONCURRENT_REQUESTS", "2")
    ),
    query_prefix=os.getenv(
        "EMBEDDING_QUERY_PREFIX",
        "Represent this sentence for searching relevant passages: ",
    ),
)
