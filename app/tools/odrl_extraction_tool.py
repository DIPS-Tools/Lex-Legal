"""HTTP interface for the external ODRL policy-component extractor.

The module name intentionally retains the existing ``oder`` spelling because
callers may already import it. Public functions use the correct ODRL spelling.
The synchronous interface is useful for scripts, while the asynchronous wrapper
keeps FastAPI and LangGraph event loops from blocking on ``requests`` I/O.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ODRL_ENDPOINT = "http://odrl-extraction:5001/odrl_generation"
DEFAULT_ODRL_TIMEOUT_SECONDS = 30.0
DEFAULT_ODRL_MAX_RETRIES = 2
DEFAULT_TEST_POLICY = (
    "The document may be downloaded once and retained for 30 days "
    "for academic research only."
)
DEFAULT_HOST_TEST_ENDPOINT = "http://127.0.0.1:5001/odrl_generation"


class ODRLExtractionError(RuntimeError):
    """Raised when the endpoint cannot return a valid extraction response."""


def _normalise_endpoint_url(value: str | None) -> str:
    """Return a complete endpoint URL, tolerating the legacy Docker value."""
    endpoint = str(value or DEFAULT_ODRL_ENDPOINT).strip()
    if not endpoint:
        endpoint = DEFAULT_ODRL_ENDPOINT
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"

    parsed = urlsplit(endpoint)
    hostname = parsed.hostname or ""
    netloc = parsed.netloc
    if hostname == "odrl-extraction" and parsed.port is None:
        netloc = f"{netloc}:5001"
    path = parsed.path.rstrip("/") or "/odrl_generation"
    return urlunsplit((parsed.scheme, netloc, path, parsed.query, ""))


def get_odrl_endpoint(endpoint: str | None = None) -> str:
    """Return the effective endpoint from an override, environment, or default."""
    return _normalise_endpoint_url(
        endpoint or os.getenv("ODRL_EXTRACTION_ENDPOINT")
    )


def _validate_response(payload: Any) -> dict[str, list[dict[str, str]]]:
    """Validate and normalize the endpoint's public ``rules`` response."""
    if not isinstance(payload, dict):
        raise ODRLExtractionError("ODRL endpoint response must be a JSON object.")
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise ODRLExtractionError(
            "ODRL endpoint response is missing the 'rules' list."
        )

    fields = (
        "source_text",
        "assignee",
        "action",
        "asset",
        "rule_type",
        "constraint",
        "purpose",
    )
    rules: list[dict[str, str]] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ODRLExtractionError(
                f"ODRL rule at index {index} must be a JSON object."
            )
        rules.append({
            field: " ".join(str(item.get(field) or "").split())
            for field in fields
        })
    return {"rules": rules}


def extract_odrl_components(
    policy_text: str,
    *,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Call the configured ODRL endpoint and return validated rule objects."""
    text = str(policy_text or "").strip()
    if not text:
        raise ValueError("ODRL extraction requires non-empty policy text.")

    endpoint_url = get_odrl_endpoint(endpoint)
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(
            os.getenv(
                "ODRL_EXTRACTION_TIMEOUT_SECONDS",
                str(DEFAULT_ODRL_TIMEOUT_SECONDS),
            )
        )
    )
    retries = (
        int(max_retries)
        if max_retries is not None
        else int(
            os.getenv(
                "ODRL_EXTRACTION_MAX_RETRIES",
                str(DEFAULT_ODRL_MAX_RETRIES),
            )
        )
    )
    if timeout <= 0:
        raise ValueError("ODRL extraction timeout must be greater than zero.")
    if retries < 0:
        raise ValueError("ODRL extraction retries cannot be negative.")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                endpoint_url,
                json={"text": text},
                timeout=timeout,
            )
            response.raise_for_status()
            return _validate_response(response.json())
        except (
            requests.RequestException,
            ValueError,
            ODRLExtractionError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(0.5 * (2 ** attempt), 2.0))

    raise ODRLExtractionError(
        f"ODRL extraction failed after {retries + 1} attempt(s): {last_error}"
    ) from last_error


async def aextract_odrl_components(
    policy_text: str,
    *,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> dict[str, list[dict[str, str]]]:
    """Run :func:`extract_odrl_components` without blocking the event loop."""
    return await asyncio.to_thread(
        extract_odrl_components,
        policy_text,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def test_odrl_extraction_endpoint(
    policy_text: str = DEFAULT_TEST_POLICY,
    *,
    endpoint: str | None = None,
    timeout_seconds: float = DEFAULT_ODRL_TIMEOUT_SECONDS,
    max_retries: int = 0,
) -> dict[str, list[dict[str, str]]]:
    """Run one real endpoint smoke test and return its validated response."""
    return extract_odrl_components(
        policy_text,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def main(argv: list[str] | None = None) -> int:
    """Test the configured ODRL endpoint from the command line."""
    parser = argparse.ArgumentParser(
        description="Send policy text to the ODRL extraction endpoint."
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEST_POLICY,
        help="Policy text to extract (defaults to the built-in example).",
    )
    parser.add_argument(
        "--endpoint",
        default=(
            os.getenv("ODRL_EXTRACTION_ENDPOINT")
            or DEFAULT_HOST_TEST_ENDPOINT
        ),
        help=(
            "Endpoint URL. The CLI defaults to localhost when the environment "
            "does not define ODRL_EXTRACTION_ENDPOINT."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_ODRL_TIMEOUT_SECONDS,
        help="Request timeout in seconds (default: 30).",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Retries after the first request (default: 0 for a quick test).",
    )
    args = parser.parse_args(argv)
    endpoint_url = get_odrl_endpoint(args.endpoint)
    print(f"Testing ODRL endpoint: {endpoint_url}", file=sys.stderr)
    try:
        result = test_odrl_extraction_endpoint(
            args.text,
            endpoint=endpoint_url,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
        )
    except (ODRLExtractionError, ValueError) as exc:
        print(f"ODRL endpoint test failed: {exc}", file=sys.stderr)
        if urlsplit(endpoint_url).hostname == "odrl-extraction":
            print(
                "The hostname 'odrl-extraction' is normally available only "
                "inside the Docker Compose network. From the host, use "
                "--endpoint http://127.0.0.1:5001/odrl_generation.",
                file=sys.stderr,
            )
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        f"ODRL endpoint test passed with {len(result['rules'])} rule(s).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
