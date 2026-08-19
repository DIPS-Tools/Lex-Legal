"""HTTP client for validating ODRL policies with the ODRL Engine."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict
import requests
from dotenv import load_dotenv


load_dotenv()

DEFAULT_ODRL_VALIDATOR_ENDPOINT = (
    "http://odrl-engine/api/validate_ODRL"
)
DEFAULT_HOST_TEST_ENDPOINT = (
    "http://10.22.12.250:8031/api/validate_ODRL"
)
DEFAULT_ODRL_VALIDATOR_TIMEOUT_SECONDS = 60.0
DEFAULT_ODRL_VALIDATOR_MAX_RETRIES = 2

DEFAULT_TEST_POLICY: dict[str, Any] = {
    "@context": "http://www.w3.org/ns/odrl.jsonld",
    "@type": "Policy",
    "uid": "urn:lexlegal:policy:validation-test",
    "permission": [
        {
            "uid": "urn:lexlegal:policy:validation-test:rule:1",
            "action": "reproduce",
            "target": "urn:lexlegal:asset:test-document",
            "constraint": [
                {
                    "leftOperand": "count",
                    "operator": "lteq",
                    "rightOperand": {
                        "@value": 1,
                        "@type": "xsd:integer",
                    },
                },
                {
                    "leftOperand": "purpose",
                    "operator": "eq",
                    "rightOperand": {
                        "@value": "academic research",
                        "@type": "xsd:string",
                    },
                },
            ],
        }
    ],
}


class ODRLValidationResult(TypedDict):
    valid: bool
    diagnostic_report: str
    engine_result: dict[str, Any]


class ODRLValidationError(RuntimeError):
    """Raised when the service cannot return a valid validation response."""


def _normalise_endpoint_url(value: str | None) -> str:
    """Return the configured ODRL validation endpoint URL."""
    endpoint = str(value or DEFAULT_ODRL_VALIDATOR_ENDPOINT).strip()
    if not endpoint:
        endpoint = DEFAULT_ODRL_VALIDATOR_ENDPOINT
    if "://" not in endpoint:
        endpoint = f"http://{endpoint}"
    return endpoint.rstrip("/")


def get_odrl_validator_endpoint(endpoint: str | None = None) -> str:
    """Resolve an override, canonical env var, requested legacy env var, or default."""
    configured = (
        endpoint
        or os.getenv("ODRL_VALIDATOR_ENDPOINT")
        or os.getenv("ODRL_validator_ENDPOINT")
    )
    return _normalise_endpoint_url(configured)


def _validate_response(payload: Any) -> ODRLValidationResult:
    """Normalize the new structured ODRL Engine response for graph callers."""
    if not isinstance(payload, dict):
        raise ODRLValidationError(
            "ODRL validator response must be a JSON object."
        )
    engine_result = payload.get("result")
    if not isinstance(engine_result, dict):
        raise ODRLValidationError(
            "ODRL validator response field 'result' must be an object."
        )

    is_valid_rdf = engine_result.get("is_valid_RDF")
    if not isinstance(is_valid_rdf, bool):
        raise ODRLValidationError(
            "ODRL validator result field 'is_valid_RDF' must be a boolean."
        )
    is_valid_odrl = engine_result.get("is_valid_ODRL", False)
    if not isinstance(is_valid_odrl, bool):
        raise ODRLValidationError(
            "ODRL validator result field 'is_valid_ODRL' must be a boolean."
        )

    return {
        "valid": is_valid_rdf and is_valid_odrl,
        "diagnostic_report": _format_diagnostic_report(engine_result),
        "engine_result": dict(engine_result),
    }


def _string_list(value: Any) -> list[str]:
    """Return readable entries from an engine errors/warnings/info field."""
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _format_diagnostic_report(engine_result: dict[str, Any]) -> str:
    """Build the legacy readable report while retaining structured data."""
    lines = [
        f"RDF valid: {engine_result.get('is_valid_RDF') is True}",
        f"ODRL valid: {engine_result.get('is_valid_ODRL') is True}",
    ]
    file_format = str(engine_result.get("file_format") or "").strip()
    if file_format:
        lines.append(f"Detected RDF format: {file_format}")

    graph_size = engine_result.get("ODRL_graph_size")
    if isinstance(graph_size, int) and not isinstance(graph_size, bool):
        lines.append(f"ODRL graph size: {graph_size} RDF triples")

    info = _string_list(engine_result.get("info"))
    warnings = _string_list(engine_result.get("warnings"))
    errors = _string_list(engine_result.get("errors"))
    if info:
        lines.extend(["Information:", *(f"- {item}" for item in info)])
    if warnings:
        lines.extend(["Warnings:", *(f"- {item}" for item in warnings)])
    if errors:
        lines.extend(["Errors:", *(f"- {item}" for item in errors)])

    stats_text = str(engine_result.get("odrl_stats_text") or "").strip()
    if stats_text:
        lines.append(stats_text)
    shacl_report = str(
        engine_result.get("shacl_validation_report") or ""
    ).strip()
    if shacl_report:
        lines.extend(["SHACL validation report:", shacl_report])
    return "\n".join(lines)


def validate_odrl(
    policy: dict[str, Any] | str,
    *,
    endpoint: str | None = None,
    timeout_seconds: float | None = None,
    max_retries: int | None = None,
) -> ODRLValidationResult:
    """Validate a JSON-LD object or serialized RDF string through ODRL Engine."""
    if isinstance(policy, dict):
        if not policy:
            raise ValueError("ODRL validation requires a non-empty policy.")
        # convert the dict to string
        odrl_string = json.dumps(policy, ensure_ascii=False)
    elif isinstance(policy, str):

        odrl_string = policy.strip()
        if not odrl_string:
            raise ValueError("ODRL validation requires a non-empty policy.")
    else:
        raise TypeError("ODRL policy must be a JSON object or RDF string.")

    endpoint_url = get_odrl_validator_endpoint(endpoint)
    timeout = (
        float(timeout_seconds)
        if timeout_seconds is not None
        else float(os.getenv(
            "ODRL_VALIDATOR_TIMEOUT_SECONDS",
            str(DEFAULT_ODRL_VALIDATOR_TIMEOUT_SECONDS),
        ))
    )
    retries = (
        int(max_retries)
        if max_retries is not None
        else int(os.getenv(
            "ODRL_VALIDATOR_MAX_RETRIES",
            str(DEFAULT_ODRL_VALIDATOR_MAX_RETRIES),
        ))
    )
    if timeout <= 0:
        raise ValueError("ODRL validator timeout must be greater than zero.")
    if retries < 0:
        raise ValueError("ODRL validator retries cannot be negative.")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                endpoint_url,
                json={"odrl": odrl_string},
                timeout=timeout,
            )
            response.raise_for_status()
            return _validate_response(response.json())
        except (
            requests.RequestException,
            ValueError,
            ODRLValidationError,
        ) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(0.5 * (2**attempt), 2.0))

    raise ODRLValidationError(
        f"ODRL validation failed after {retries + 1} attempt(s): "
        f"{last_error}"
    ) from last_error

async def avalidate_odrl(
    policy: dict[str, Any] | str,
    **kwargs: Any,
) -> ODRLValidationResult:
    """Run validation in a worker thread without blocking an async event loop."""
    return await asyncio.to_thread(validate_odrl, policy, **kwargs)


# Compatibility aliases for callers using the previous client function names.
validate_odrl_policy = validate_odrl
avalidate_odrl_policy = avalidate_odrl


def test_odrl_validation_endpoint(
    policy: dict[str, Any] | str = DEFAULT_TEST_POLICY,
    *,
    endpoint: str | None = None,
    timeout_seconds: float = DEFAULT_ODRL_VALIDATOR_TIMEOUT_SECONDS,
    max_retries: int = 0,
) -> ODRLValidationResult:
    """Run one real endpoint smoke test and return its validated response."""
    return validate_odrl(
        policy,
        endpoint=endpoint,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )


def _load_policy(path: str | None) -> dict[str, Any] | str:
    if not path:
        return DEFAULT_TEST_POLICY
    policy_text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in {".json", ".jsonld"}:
        parsed = json.loads(policy_text)
        if not isinstance(parsed, dict):
            raise ValueError("A JSON-LD policy file must contain a JSON object.")
        return parsed
    return policy_text


def main(argv: list[str] | None = None) -> int:
    """Validate the built-in policy or a supplied RDF policy file."""
    parser = argparse.ArgumentParser(
        description="Validate an ODRL JSON-LD or RDF policy."
    )
    parser.add_argument("--policy-file", help="Optional JSON-LD or RDF policy file.")
    parser.add_argument(
        "--endpoint",
        default=(
            os.getenv("ODRL_VALIDATOR_ENDPOINT")
            or os.getenv("ODRL_validator_ENDPOINT")
            or DEFAULT_HOST_TEST_ENDPOINT
        ),
        help="Validation endpoint URL (defaults to the host-published service).",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=0)
    args = parser.parse_args(argv)

    try:
        endpoint_url = get_odrl_validator_endpoint(args.endpoint)
        print(f"Testing ODRL validator: {endpoint_url}", file=sys.stderr)
        result = test_odrl_validation_endpoint(
            _load_policy(args.policy_file),
            endpoint=endpoint_url,
            timeout_seconds=args.timeout,
            max_retries=args.retries,
        )
    except (
        OSError,
        json.JSONDecodeError,
        ODRLValidationError,
        TypeError,
        ValueError,
    ) as exc:
        print(f"ODRL validator test failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
