from __future__ import annotations

import math
import threading
import time
from typing import Any
import httpx

from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError
from langchain_core.embeddings import Embeddings

from app.config.embedding_config import (
    EmbeddingConfig,
    embedding_config,
)
from app.core.logger import logger


class TEIEmbeddings(Embeddings):
    """LangChain embedding adapter for Hugging Face Text Embeddings
    Inference."""

    def __init__(
        self,
        endpoint_url: str,
        *,
        expected_dimension: int,
        normalize: bool = True,
        truncate: bool = True,
        timeout: float = 60,
        max_retries: int = 5,
        retry_base_delay: float = 1,
        retry_max_delay: float = 15,
        max_concurrent_requests: int = 2,
        query_prefix: str = "",
        inference_client: Any | None = None,
    ):
        self.endpoint_url = endpoint_url.rstrip("/")
        self.expected_dimension = expected_dimension
        self.normalize = normalize
        self.truncate = truncate
        self.query_prefix = query_prefix
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.retry_max_delay = retry_max_delay
        self._request_slots = threading.BoundedSemaphore(
            max_concurrent_requests
        )
        self.client = inference_client or InferenceClient(
            model=self.endpoint_url,
            timeout=timeout,
        )

        self.http_client = httpx.Client(
            timeout=timeout,
            headers={"Content-Type": "application/json"},
        )


    def _feature_extraction_with_retry(
        self,
        texts: list[str],
    ) -> Any:
        """Call TEI with bounded concurrency and overload backoff."""
        for attempt in range(self.max_retries + 1):
            try:
                with self._request_slots:
                    return self.client.feature_extraction(
                        text=texts,
                        normalize=self.normalize,
                        truncate=self.truncate,
                    )
            except HfHubHTTPError as exc:
                status_code = (
                    exc.response.status_code
                    if exc.response is not None
                    else None
                )
                retryable = status_code in {429, 503}
                if not retryable or attempt >= self.max_retries:
                    raise

                retry_after = (
                    exc.response.headers.get("Retry-After")
                    if exc.response is not None
                    else None
                )
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = self.retry_base_delay * (2 ** attempt)
                delay = min(delay, self.retry_max_delay)

                logger.warning(
                    f"TEI is overloaded (HTTP {status_code}); "
                    f"retrying request in {delay:g}s "
                    f"(attempt {attempt + 1}/{self.max_retries})."
                )
                time.sleep(delay)

        raise RuntimeError("TEI retry loop ended unexpectedly.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed one TEI batch and validate count, numbers, and dimensions."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Every embedding input must be a non-empty string.")

        result = self._feature_extraction_with_retry(texts)
        vectors = result.tolist() if hasattr(result, "tolist") else result

        if (
            len(texts) == 1
            and isinstance(vectors, list)
            and vectors
            and isinstance(vectors[0], (int, float))
        ):
            vectors = [vectors]

        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise ValueError(
                "TEI returned an unexpected number of embedding vectors."
            )

        validated_vectors: list[list[float]] = []
        for vector in vectors:
            if not isinstance(vector, list):
                raise TypeError("TEI embedding vector must be a list.")
            if len(vector) != self.expected_dimension:
                raise ValueError(
                    f"TEI embedding dimension mismatch: expected "
                    f"{self.expected_dimension}, got {len(vector)}."
                )

            float_vector = [float(value) for value in vector]
            if any(not math.isfinite(value) for value in float_vector):
                raise ValueError("TEI embedding contains a non-finite value.")
            validated_vectors.append(float_vector)

        return validated_vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the configured BGE retrieval instruction."""
        query_text = f"{self.query_prefix}{text}" if self.query_prefix else text
        return self.embed_documents([query_text])[0]

    def _tei_request(
            self,
            method: str,
            path: str,
            *,
            json: dict[str, Any] | None = None,
            return_text: bool = False,
    ) -> Any:
        """Make a direct request to a TEI HTTP endpoint."""
        url = f"{self.endpoint_url}{path}"

        for attempt in range(self.max_retries + 1):
            try:
                with self._request_slots:
                    response = self.http_client.request(
                        method=method,
                        url=url,
                        json=json,
                    )

                if (
                        response.status_code not in {429, 503}
                        or attempt >= self.max_retries
                ):
                    response.raise_for_status()
                    return response.text if return_text else response.json()

                retry_after = response.headers.get("Retry-After")
                try:
                    delay = float(retry_after)
                except (TypeError, ValueError):
                    delay = self.retry_base_delay * (2 ** attempt)

                delay = min(delay, self.retry_max_delay)

                logger.warning(
                    f"TEI returned HTTP {response.status_code}; "
                    f"retrying in {delay:g}s "
                    f"(attempt {attempt + 1}/{self.max_retries})."
                )
                time.sleep(delay)

            except httpx.HTTPStatusError:
                raise
            except httpx.RequestError as exc:
                if attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Could not connect to TEI at {url}"
                    ) from exc

                delay = min(
                    self.retry_base_delay * (2 ** attempt),
                    self.retry_max_delay,
                )
                logger.warning(
                    f"TEI connection failed; retrying in {delay:g}s."
                )
                time.sleep(delay)

        raise RuntimeError("TEI request retry loop ended unexpectedly.")

    def info(self) -> dict[str, Any]:
        """GET /info: return information about the loaded model."""
        return self._tei_request("GET", "/info")

    def health(self) -> bool:
        """GET /health: return True when TEI is healthy."""
        self._tei_request("GET", "/health", return_text=True)
        return True

    def metrics(self) -> str:
        """GET /metrics: return Prometheus metrics as text."""
        return self._tei_request(
            "GET",
            "/metrics",
            return_text=True,
        )

    def embed_api(
            self,
            texts: str | list[str],
            *,
            normalize: bool | None = None,
            truncate: bool | None = None,
    ) -> list[float] | list[list[float]]:
        """POST /embed: generate pooled embeddings."""
        payload = {
            "inputs": texts,
            "normalize": (
                self.normalize if normalize is None else normalize
            ),
            "truncate": (
                self.truncate if truncate is None else truncate
            ),
        }
        return self._tei_request("POST", "/embed", json=payload)

    def openai_embeddings(
            self,
            texts: str | list[str],
            *,
            model: str = "bge-large-en-v1.5",
    ) -> dict[str, Any]:
        """POST /v1/embeddings: use the OpenAI-compatible endpoint."""
        payload = {
            "input": texts,
            "model": model,
            "encoding_format": "float",
        }
        return self._tei_request(
            "POST",
            "/v1/embeddings",
            json=payload,
        )

    def similarity(
            self,
            source_sentence: str,
            sentences: list[str],
            *,
            truncate: bool | None = None,
    ) -> list[float]:
        """POST /similarity: compare one sentence with several sentences."""
        payload = {
            "inputs": {
                "source_sentence": source_sentence,
                "sentences": sentences,
            },
            "parameters": {
                "truncate": (
                    self.truncate if truncate is None else truncate
                ),
            },
        }
        return self._tei_request(
            "POST",
            "/similarity",
            json=payload,
        )

    def tokenize(
            self,
            texts: str | list[str],
            *,
            add_special_tokens: bool = True,
    ) -> Any:
        """POST /tokenize: tokenize one or more strings."""
        payload = {
            "inputs": texts,
            "add_special_tokens": add_special_tokens,
        }
        return self._tei_request(
            "POST",
            "/tokenize",
            json=payload,
        )

    def predict(
            self,
            texts: str | list[str],
            *,
            truncate: bool | None = None,
            raw_scores: bool = False,
    ) -> Any:
        """
        POST /predict.

        Requires TEI to be running a sequence-classification model.
        It will not work with bge-large-en-v1.5.
        """
        payload = {
            "inputs": texts,
            "truncate": (
                self.truncate if truncate is None else truncate
            ),
            "raw_scores": raw_scores,
        }
        return self._tei_request(
            "POST",
            "/predict",
            json=payload,
        )

    def rerank(
            self,
            query: str,
            texts: list[str],
            *,
            truncate: bool | None = None,
            raw_scores: bool = False,
            return_text: bool = True,
    ) -> list[dict[str, Any]]:
        """
        POST /rerank.

        Requires a reranker model such as BAAI/bge-reranker-large.
        It will not work with bge-large-en-v1.5.
        """
        payload = {
            "query": query,
            "texts": texts,
            "truncate": (
                self.truncate if truncate is None else truncate
            ),
            "raw_scores": raw_scores,
            "return_text": return_text,
        }
        return self._tei_request(
            "POST",
            "/rerank",
            json=payload,
        )

    def close(self) -> None:
        """Close pooled HTTP connections."""
        self.http_client.close()



class EmbeddingClientManager:
    """Create and reuse one TEI embedding client."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.client: TEIEmbeddings | None = None

    def init(self) -> None:
        if self.client is not None:
            return

        logger.info(
            f"Initializing TEI embedding client: "
            f"endpoint={self.config.endpoint_url}, "
            f"model={self.config.model}, "
            f"dimension={self.config.dimension}"
        )
        self.client = TEIEmbeddings(
            endpoint_url=self.config.endpoint_url,
            expected_dimension=self.config.dimension,
            normalize=self.config.normalize,
            truncate=self.config.truncate,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            retry_base_delay=self.config.retry_base_delay,
            retry_max_delay=self.config.retry_max_delay,
            max_concurrent_requests=self.config.max_concurrent_requests,
            query_prefix=self.config.query_prefix,
        )

    def get_client(self) -> TEIEmbeddings:
        if self.client is None:
            raise RuntimeError(
                "Embedding client has not been initialized. "
                "Call embedding_client_manager.init() first."
            )
        return self.client


embedding_client_manager = EmbeddingClientManager(embedding_config)


if __name__ == "__main__":
    import asyncio

    async def test() -> None:
        embedding_client_manager.init()
        client = embedding_client_manager.get_client()
        # vector = await client.aembed_query("What rights does a data subject have?")
        # texts = "What rights does a data subject have"
        # vector = await asyncio.to_thread(client.embed_api, texts)
        #
        # print("Dimension:", len(vector))
        # print("First five values:", vector[:5])


        #
        print(client.info())
        print(client.health())
        print(client.metrics())

        vectors = client.embed_api([
            "The first document",
            "The second document",
        ])

        scores = client.similarity(
            "What rights does a data subject have?",
            [
                "A data subject has the right of access.",
                "The weather is sunny today.",
            ],
        )

        tokens = client.tokenize(
            "What rights does a data subject have?"
        )

        openai_response = client.openai_embeddings([
            "The first document",
            "The second document",
        ])

        print("Vectors:", len(vectors))
        print("Similarity:", scores)
        print("Tokens:", tokens)
        print("OpenAI response:", openai_response)



    asyncio.run(test())
