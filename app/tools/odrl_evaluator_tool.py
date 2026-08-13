"""HTTP client for evaluating an ODRL policy against a State of the World.

The ODRL Engine accepts an RDF policy string (JSON-LD, Turtle, RDF/XML, and
other RDFLib-supported formats) plus State-of-the-World data encoded as CSV.
This module is deliberately independent from the query graph so it can be
tested now and later called from ``node_odrl_processing``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any, TypedDict
from urllib.parse import urlsplit, urlunsplit

import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ODRL_EVALUATOR_ENDPOINT = (
    "http://odrl-engine/api/evaluate_policy_on_sotw"
)
DEFAULT_HOST_TEST_ENDPOINT = (
    "http://10.22.12.250:8031/api/evaluate_policy_on_sotw"
)
DEFAULT_ODRL_EVALUATOR_TIMEOUT_SECONDS = 60.0
DEFAULT_ODRL_EVALUATOR_MAX_RETRIES = 2

DEFAULT_TEST_POLICY = """@prefix odrl: <http://www.w3.org/ns/odrl/2/> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

<http://example.com/policy:6161>
  a odrl:Offer ;
  odrl:permission [
    odrl:action [
      rdf:value odrl:print ;
      odrl:refinement [
        odrl:leftOperand odrl:resolution ;
        odrl:operator odrl:lteq ;
        odrl:rightOperand 1200 ;
        odrl:unit "http://dbpedia.org/resource/Dots_per_inch"^^xsd:string
      ]
    ] ;
    odrl:assignee <http://example.com/org:John> ;
    odrl:target <http://example.com/document:1234>
  ] .
"""

DEFAULT_TEST_SOTW = """http://www.w3.org/ns/odrl/2/dateTime,http://www.w3.org/ns/odrl/2/Party,http://www.w3.org/ns/odrl/2/Action,http://www.w3.org/ns/odrl/2/Asset,http://www.w3.org/ns/odrl/2/Action http://www.w3.org/ns/odrl/2/resolution
2026-01-11T11:33:10.665638,http://example.com/org:John,http://www.w3.org/ns/odrl/2/print,http://example.com/document:1234,1190
"""


class ODRLEvaluationResult(TypedDict):
    evaluation_state: Any
    valid: bool
    rows_violating_permissions: list[int]
    rows_violating_prohibitions: list[int]
    obligations_not_satisfied: list[Any]
    unfulfilled_duties: list[Any]
    unfulfilled_consequences: list[Any]
    unfulfilled_remedies: list[Any]


class ODRLEvaluationError(RuntimeError):
    """Raised when the ODRL Engine cannot return a valid evaluation result."""


def _normalise_endpoint_url(value: str | None) -> str:
    """Return the canonical evaluator URL, accepting the proposed legacy URL."""
    endpoint = str(value or DEFAULT_ODRL_EVALUATOR_ENDPOINT).strip()
    if not endpoint:
        endpoint = DEFAULT_ODRL_EVALUATOR_ENDPOINT
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"

    parsed = urlsplit(endpoint)
    netloc = parsed.netloc
    path = parsed.path.rstrip("/")
    if path.endswith("/evalute_policy_on_sotw"):
        path = path.removesuffix("/evalute_policy_on_sotw")
        path += "/evaluate_policy_on_sotw"
    # The current Docker image exposes Nginx on port 80, which proxies FastAPI
    # under /api/. Accept the originally proposed :8083 URL as a compatibility
    # alias, but send the request to the service's real Compose address.
    if parsed.hostname == "odrl-engine" and parsed.port == 8083:
        netloc = "odrl-engine"
    if path == "/evaluate_policy_on_sotw":
        path = f"/api{path}"
    if not path or path == "/":
        path = "/api/evaluate_policy_on_sotw"
    return urlunsplit((parsed.scheme, netloc, path, parsed.query, ""))


def get_odrl_evaluator_endpoint(endpoint: str | None = None) -> str:
    """Return the endpoint override, environment value, or Docker default."""
    return _normalise_endpoint_url(
        endpoint or os.getenv("ODRL_EVALUATOR_ENDPOINT")
    )


def _as_list(payload: dict[str, Any], field: str) -> list[Any]:
    value = payload.get(field)
    if not isinstance(value, list):
        raise ODRLEvaluationError(
            f"ODRL evaluator response field {field!r} must be a list."
        )
    return value


def _validate_response(payload: Any) -> ODRLEvaluationResult:
    """Validate the response contract published by ODRL Engine."""
    if not isinstance(payload, dict):
        raise ODRLEvaluationError(
            "ODRL evaluator response must be a JSON object."
        )
    if not isinstance(payload.get("valid"), bool):
        raise ODRLEvaluationError(
            "ODRL evaluator response field 'valid' must be a boolean."
        )

    permission_rows = _as_list(payload, "rows_violating_permissions")
    prohibition_rows = _as_list(payload, "rows_violating_prohibitions")
    if not all(isinstance(item, int) and not isinstance(item, bool)
               for item in (*permission_rows, *prohibition_rows)):
        raise ODRLEvaluationError(
            "ODRL evaluator violating-row fields must contain integers."
        )

    return ODRLEvaluationResult(
        evaluation_state=payload.get("evaluation_state"),
        valid=payload["valid"],
        rows_violating_permissions=permission_rows,
        rows_violating_prohibitions=prohibition_rows,
        obligations_not_satisfied=_as_list(
            payload, "obligations_not_satisfied"
        ),
        unfulfilled_duties=_as_list(payload, "unfulfilled_duties"),
        unfulfilled_consequences=_as_list(
            payload, "unfulfilled_consequences"
        ),
        unfulfilled_remedies=_as_list(payload, "unfulfilled_remedies"),
    )


def evaluate_policy_on_sotw(
    policy: str | dict[str, Any],
    sotw: str,
    *,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> ODRLEvaluationResult:
    """Evaluate an RDF policy string or JSON-LD object against SotW CSV."""
    if isinstance(policy, dict):
        policy_text = json.dumps(policy, ensure_ascii=False)
    else:
        policy_text = str(policy or "").strip()

    sotw_csv = str(sotw or "").strip()

    if not policy_text:
        raise ValueError("ODRL evaluation requires a non-empty policy.")
    if not sotw_csv:
        raise ValueError("ODRL evaluation requires non-empty SotW CSV data.")

    endpoint_url = get_odrl_evaluator_endpoint(endpoint)

    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(os.getenv(
            "ODRL_EVALUATOR_TIMEOUT_SECONDS",
            str(DEFAULT_ODRL_EVALUATOR_TIMEOUT_SECONDS),
        ))
    )
    retries = (
        int(max_retries)
        if max_retries is not None
        else int(os.getenv(
            "ODRL_EVALUATOR_MAX_RETRIES",
            str(DEFAULT_ODRL_EVALUATOR_MAX_RETRIES),
        ))
    )
    if timeout <= 0:
        raise ValueError("ODRL evaluator timeout must be greater than zero.")
    if retries < 0:
        raise ValueError("ODRL evaluator retries cannot be negative.")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                endpoint_url,
                json={"policy": policy_text, "sotw": sotw_csv},
                timeout=timeout,
            )
            response.raise_for_status()
            return _validate_response(response.json())
        except (
            requests.RequestException,
            ValueError,
            ODRLEvaluationError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(0.5 * (2**attempt), 2.0))

    raise ODRLEvaluationError(
        f"ODRL evaluation failed after {retries + 1} attempt(s): "
        f"{last_error}"
    ) from last_error


async def aevaluate_policy_on_sotw(
    policy: str | dict[str, Any],
    sotw: str,
    **kwargs: Any,
) -> ODRLEvaluationResult:
    """Run the synchronous HTTP client without blocking an async event loop."""
    return await asyncio.to_thread(
        evaluate_policy_on_sotw,
        policy,
        sotw,
        **kwargs,
    )


def test_odrl_evaluator_endpoint(
    policy: str = DEFAULT_TEST_POLICY,
    sotw: str = DEFAULT_TEST_SOTW,
    *,
    endpoint: str | None = None,
    timeout_seconds: float = DEFAULT_ODRL_EVALUATOR_TIMEOUT_SECONDS,
    max_retries: int = 0,
) -> ODRLEvaluationResult:
    """Run one real endpoint smoke test and return its validated response."""
    return evaluate_policy_on_sotw(
        policy,
        sotw,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def main(argv: list[str] | None = None) -> int:
    """Exercise the evaluator endpoint with built-in policy and SotW data."""
    parser = argparse.ArgumentParser(
        description="Evaluate a sample ODRL policy against sample SotW CSV."
    )
    parser.add_argument(
        "--endpoint",
        default=os.getenv("ODRL_EVALUATOR_ENDPOINT") or DEFAULT_HOST_TEST_ENDPOINT,
        help="Evaluator endpoint URL (defaults to the host-published service).",
    )
    parser.add_argument("--policy-file", help="Optional RDF policy file.")
    parser.add_argument("--sotw-file", help="Optional SotW CSV file.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        policy = (
            open(args.policy_file, encoding="utf-8").read()
            if args.policy_file
            else DEFAULT_TEST_POLICY
        )
        sotw = (
            open(args.sotw_file, encoding="utf-8").read()
            if args.sotw_file
            else DEFAULT_TEST_SOTW
        )
        endpoint_url = get_odrl_evaluator_endpoint(args.endpoint)
        print(f"Testing ODRL evaluator: {endpoint_url}", file=sys.stderr)
        result = test_odrl_evaluator_endpoint(
            policy,
            sotw,
            endpoint=endpoint_url,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
        )
    except (OSError, ODRLEvaluationError, ValueError) as exc:
        print(f"ODRL evaluator test failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
