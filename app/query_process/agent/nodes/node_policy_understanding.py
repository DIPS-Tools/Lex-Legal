"""Understand policy text through independent ODRL and AI extraction.

The two extractors receive the authoritative policy independently. Their raw
outputs are retained, aligned, compared, and written to a dedicated JSON Lines
audit log. A second AI call is used only when both extractors succeed and their
rule sets materially disagree.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Awaitable
from uuid import uuid4
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage

from app.core.logger import logger
from app.config.llm_config import llm_config
from app.llm.llm_utils import get_llm_client
from app.prompt.load_prompt import load_prompt
from app.query_process.agent.state import QueryGraphState
from app.tools.oder_extraction_tool import (
    aextract_odrl_components,
    get_odrl_endpoint,
)
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import add_done_task, add_running_task


RULE_FIELDS = (
    "source_text",
    "assignee",
    "action",
    "asset",
    "rule_type",
    "constraint",
    "purpose",
)

CONTEXT_FIELDS = (
    "purposes",
    "organisations",
    "legal_roles",
    "business_activities",
    "systems",
    "data_categories",
    "people",
    "jurisdictions",
    "procedures",
)

RULE_TYPE_ALIASES = {
    "permission": "permission",
    "permit": "permission",
    "permitted": "permission",
    "obligation": "obligation",
    "duty": "obligation",
    "required": "obligation",
    "prohibition": "prohibition",
    "prohibited": "prohibition",
    "forbidden": "prohibition",
}

MAX_POLICY_CHARACTERS = 50_000
RULE_ALIGNMENT_THRESHOLD = 0.45
AUDIT_LOG_LOCK = threading.Lock()


def _response_content_to_text(content: Any) -> str:
    """Normalize common LangChain response formats into plain text."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _parse_json_object(content: Any, source_name: str) -> dict[str, Any]:
    """Parse one JSON object and tolerate an accidental outer code fence."""
    text = _response_content_to_text(content)
    if not text:
        raise ValueError(f"{source_name} returned empty content.")
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source_name} returned invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{source_name} response must be a JSON object.")
    return parsed


def _clean_text(value: Any) -> str:
    """Convert a scalar policy component to normalized single-line text."""
    if not isinstance(value, (str, int, float)):
        return ""
    return " ".join(str(value).split())


def _normalize_rule_type(value: Any) -> str:
    """Map common deontic labels to the three supported rule types."""
    normalized = _clean_text(value).casefold()
    return RULE_TYPE_ALIASES.get(normalized, "")


def normalize_policy_rules(value: Any) -> list[dict[str, str]]:
    """Validate, normalize, and deduplicate a policy-rule collection."""
    if not isinstance(value, list):
        raise ValueError("Policy extraction response requires a 'rules' list.")

    rules: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        rule = {field: _clean_text(item.get(field)) for field in RULE_FIELDS}
        rule["rule_type"] = _normalize_rule_type(item.get("rule_type"))
        if not any(rule[field] for field in RULE_FIELDS if field != "source_text"):
            continue
        signature = tuple(rule[field].casefold() for field in RULE_FIELDS)
        if signature in seen:
            continue
        seen.add(signature)
        rules.append(rule)
    return rules


def _normalize_context_item(value: Any) -> dict[str, Any] | None:
    """Normalize one explicit or inferred policy-context value."""
    if isinstance(value, str):
        text = _clean_text(value)
        if not text:
            return None
        return {
            "value": text,
            "evidence": text,
            "provenance": "explicit",
            "confidence": 1.0,
        }
    if not isinstance(value, dict):
        return None
    text = _clean_text(value.get("value"))
    if not text:
        return None
    evidence = _clean_text(value.get("evidence"))
    provenance = _clean_text(value.get("provenance")).casefold()
    if provenance not in {"explicit", "inferred"}:
        provenance = "explicit" if evidence == text else "inferred"
    try:
        confidence = float(value.get("confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "value": text,
        "evidence": evidence,
        "provenance": provenance,
        "confidence": min(max(confidence, 0.0), 1.0),
    }


def normalize_policy_context(value: Any) -> dict[str, Any]:
    """Return a complete, predictable policy-context dictionary."""
    raw_context = value if isinstance(value, dict) else {}
    context: dict[str, Any] = {}
    for field in CONTEXT_FIELDS:
        raw_items = raw_context.get(field, [])
        if not isinstance(raw_items, list):
            raw_items = []
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_item in raw_items:
            item = _normalize_context_item(raw_item)
            if item is None:
                continue
            key = item["value"].casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append(item)
        context[field] = items

    raw_missing = raw_context.get("missing_context", [])
    missing: list[str] = []
    seen_missing: set[str] = set()
    if isinstance(raw_missing, list):
        for value_item in raw_missing:
            text = _clean_text(value_item)
            key = text.casefold()
            if text and key not in seen_missing:
                seen_missing.add(key)
                missing.append(text)
    context["missing_context"] = missing
    return context


def _source_contains(policy_text: str, evidence: str) -> bool:
    """

    Compare evidence after whitespace normalization and case folding.

    For each rule, it asks:

      Can this rule's source_text be found in the user's original policy?

      ### Valid example

      Original policy:

      The document may be downloaded once and retained for 30 days.

      Extracted rule:

      {
          "source_text": "The document may be downloaded once",
          "action": "download",
      }

      The source text appears in the original policy, so no issue is produced:

      []

      ### Invalid example

      Original policy:

      The document may be downloaded once.

      AI-extracted rule:

      {
          "source_text": "The document may be shared with commercial partners.",
          "action": "share",
      }

      That sentence does not appear in the policy. The result is:

      [
          "independent_ai.rules[0].source_text is not supported by policy text"
      ]

      This may indicate that the AI invented or substantially paraphrased the evidence.




    """


    normalized_policy = _clean_text(policy_text).casefold()
    normalized_evidence = _clean_text(evidence).casefold()
    return bool(
        normalized_evidence and normalized_evidence in normalized_policy
    )


def validate_policy_context_evidence(
    context: dict[str, Any],
    policy_text: str,
) -> tuple[dict[str, Any], list[str]]:
    """Remove context values whose claimed evidence is absent from the policy."""
    validated: dict[str, Any] = {}
    warnings: list[str] = []
    for field in CONTEXT_FIELDS:
        validated_items: list[dict[str, Any]] = []
        for index, item in enumerate(context.get(field, [])):
            evidence = _clean_text(item.get("evidence"))
            if not _source_contains(policy_text, evidence):
                warnings.append(
                    f"policy_context.{field}[{index}] has unsupported evidence"
                )
                continue
            validated_items.append(item)
        validated[field] = validated_items
    validated["missing_context"] = list(context.get("missing_context", []))
    return validated, warnings


def _rule_source_issues(
    rules: list[dict[str, str]],
    policy_text: str,
    source_name: str,
) -> list[str]:
    """Identify rules whose source quotation cannot be found in the policy."""
    return [
        f"{source_name}.rules[{index}].source_text is not supported by policy text"
        for index, rule in enumerate(rules)
        if not _source_contains(policy_text, rule.get("source_text", ""))
    ]


def _source_grounded_rules(
    rules: list[dict[str, str]],
    policy_text: str,
) -> list[dict[str, str]]:
    """Retain only rules backed by an exact quotation from the policy."""
    return [
        rule
        for rule in rules
        if _source_contains(policy_text, rule.get("source_text", ""))
    ]


async def extract_policy_independently_with_ai(
    policy_text: str,
) -> dict[str, Any]:
    """Extract rules and context without exposing the ODRL prediction."""
    prompt = load_prompt("policy_understanding", policy_text=policy_text)
    model = get_llm_client(json_mode=True)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    parsed = _parse_json_object(response.content, "Independent policy AI")
    return {
        "rules": normalize_policy_rules(parsed.get("rules")),
        "policy_context": normalize_policy_context(
            parsed.get("policy_context")
        ),
    }


def _comparison_text(value: Any) -> str:
    """Normalize text for extraction matching without changing raw results."""
    text = _clean_text(value).casefold()
    return re.sub(r"[^\w]+", " ", text).strip()


def _text_similarity(left: Any, right: Any) -> float:
    """Return a deterministic similarity score between two component values."""
    left_text = _comparison_text(left)
    right_text = _comparison_text(right)
    if not left_text and not right_text:
        return 1.0
    if not left_text or not right_text:
        return 0.0
    if left_text == right_text:
        return 1.0
    return SequenceMatcher(None, left_text, right_text).ratio()


def _rule_similarity(left: dict[str, str], right: dict[str, str]) -> float:
    """Score rule similarity with extra weight on source and deontic type."""
    weights = {
        "source_text": 0.30,
        "rule_type": 0.20,
        "action": 0.18,
        "asset": 0.12,
        "assignee": 0.08,
        "constraint": 0.06,
        "purpose": 0.06,
    }
    return sum(
        weights[field] * _text_similarity(left[field], right[field])
        for field in weights
    )


def compare_policy_extractions(
    odrl_rules: list[dict[str, str]],
    llm_rules: list[dict[str, str]],
    *,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Greedily align independent rules and describe every disagreement."""
    # Compare every ODRL rule with every independently extracted AI rule.
    # Each tuple contains: (similarity score, ODRL index, AI index).

    # candidates = sorted(
    #     (
    #         (_rule_similarity(odrl_rule, llm_rule), odrl_index, llm_index)
    #         for odrl_index, odrl_rule in enumerate(odrl_rules)
    #         for llm_index, llm_rule in enumerate(llm_rules)
    #     ),
    #     reverse=True,
    # )
    # Each result is stored as:
    #
    #   (similarity_score, odrl_index, llm_index)
    #
    #   For example:
    #
    #   [
    #       (0.82, 0, 0),
    #       (0.35, 0, 1),
    #       (0.41, 1, 0),
    #       (0.91, 1, 1),
    #   ]
    #
    #   After descending sorting:
    #
    #   [
    #       (0.91, 1, 1),
    #       (0.82, 0, 0),
    #       (0.41, 1, 0),
    #       (0.35, 0, 1),
    #   ]


    candidates: list[tuple[float, int, int]] = []
    for odrl_index, odrl_rule in enumerate(odrl_rules):
        for llm_index, llm_rule in enumerate(llm_rules):
            score = _rule_similarity(odrl_rule, llm_rule)
            candidates.append((score, odrl_index, llm_index))

    # Process the most similar rule pairs first during greedy alignment.
    candidates.sort(reverse=True)
    used_odrl: set[int] = set()
    used_llm: set[int] = set()
    matches: list[dict[str, Any]] = []

    for score, odrl_index, llm_index in candidates:
        if score < RULE_ALIGNMENT_THRESHOLD:
            break

        if odrl_index in used_odrl or llm_index in used_llm:
            continue

        used_odrl.add(odrl_index)
        used_llm.add(llm_index)

        odrl_rule = odrl_rules[odrl_index]
        llm_rule = llm_rules[llm_index]

        differences = {
            field: {
                "odrl": odrl_rule[field],
                "llm": llm_rule[field],
            }
            for field in RULE_FIELDS

            if _comparison_text(odrl_rule[field]) != _comparison_text(llm_rule[field])
        }
        matches.append({
            "odrl_index": odrl_index,
            "llm_index": llm_index,
            "score": round(score, 4),
            "status": "exact_match" if not differences else "partial_match",
            "differences": differences,
        })

    odrl_only = [
        index for index in range(len(odrl_rules)) if index not in used_odrl
    ]
    llm_only = [
        index for index in range(len(llm_rules)) if index not in used_llm
    ]
    extraction_errors = list(errors or [])
    material_difference = bool(
        odrl_only
        or llm_only
        or any(match["differences"] for match in matches)
    )
    return {
        "matches": matches,
        "odrl_only_rule_indices": odrl_only,
        "llm_only_rule_indices": llm_only,
        "odrl_rule_count": len(odrl_rules),
        "llm_rule_count": len(llm_rules),
        "errors": extraction_errors,
        "requires_reconciliation": (
            material_difference and not extraction_errors
        ),
        "review_required": material_difference or bool(extraction_errors),
    }

    """
        for example:
        {
            "matches": [
                  {
                    "odrl_index": 0,
                    "llm_index": 0,
                    "score": 0.6228,
                    "status": "partial_match",
                    "differences": {
                      "source_text": {
                        "odrl": "The document may be downloaded once and retained for 30 days for academic research only.",
                        "llm": "The document may be downloaded once"
                      },
                      "action": {
                        "odrl": "may be downloaded",
                        "llm": "download"
                      },
                      "asset": {
                        "odrl": "The document",
                        "llm": ""
                      },
                      "constraint": {
                        "odrl": "for 30 days",
                        "llm": "once"
                      }
                    }
                  }
            ],
            "odrl_only_rule_indices": [],
            "llm_only_rule_indices": [1],
            "odrl_rule_count": 1,
            "llm_rule_count": 2,
            "errors": [],
            "requires_reconciliation": true,
            "review_required": true
        }


    """


def reconciliation_quality_issues(
    reconciled_rules: list[dict[str, str]],
    odrl_rules: list[dict[str, str]],
    llm_rules: list[dict[str, str]],
) -> list[str]:

    """Detect merged actions and information loss in reconciled output.

     This checks three important properties.

          ### Atomic actions

          The following is rejected:

          {
            "action": "download; retain"
          }

          Actions containing separators such as these are considered non-atomic:

          ;
          and
          or
          then

          ### No lost AI actions

          If the independent AI identified:

          download
          retain

          the reconciliation result must preserve both actions.

          If it returns only download, an issue is generated:

          reconciled rules lost independent_ai.rules[1] action 'retain'

          ### No complete loss of important fields

          If either input contains an asset, but every reconciled rule has an empty asset, the result is rejected.

          This is checked for:

          assignee
          asset
          constraint
          purpose

          For example:

          reconciled rules lost every supported 'purpose' value

          ## Successful reconciliation

          If there are no quality issues:

          validated_rules = reconciled
          selected_from = "ai_reconciliation"
          reconciliation_method = "ai_reconciliation"

          Later, IDs and provenance are added:

          [
            {
              "policy_rule_id": "policy-rule-1",
              "source_text": "The document may be downloaded once and retained for 30 days for academic research only.",
              "assignee": "",
              "action": "download",
              "asset": "The document",
              "rule_type": "permission",
              "constraint": "once",
              "purpose": "academic research only",
              "selected_from": "ai_reconciliation"
            },
            {
              "policy_rule_id": "policy-rule-2",
              "source_text": "The document may be downloaded once and retained for 30 days for academic research only.",
              "assignee": "",
              "action": "retain",
              "asset": "The document",
              "rule_type": "permission",
              "constraint": "30 days",
              "purpose": "academic research only",
              "selected_from": "ai_reconciliation"
            }
          ]

          These become:

          state["validated_policy_rules"]


    """

    issues: list[str] = []
    for index, rule in enumerate(reconciled_rules):
        action = _comparison_text(rule.get("action"))
        if ";" in rule.get("action", "") or re.search(r"\b(?:and|or|then)\b", action):
            issues.append(
                f"reconciled.rules[{index}].action is not atomic: "
                f"{rule.get('action', '')!r}"
            )

    # The independent prompt explicitly produces atomic actions. Reconciliation
    # must retain each distinct action even when the endpoint combines them.
    reconciled_actions = [rule["action"] for rule in reconciled_rules]
    for index, rule in enumerate(llm_rules):
        action = rule.get("action", "")
        if action and not any( _text_similarity(action, candidate) >= 0.75 for candidate in reconciled_actions):
            issues.append(
                f"reconciled rules lost independent_ai.rules[{index}] "
                f"action {action!r}"
            )

    source_rules = [*odrl_rules, *llm_rules]
    for field in ("assignee", "asset", "constraint", "purpose"):
        if any(rule.get(field) for rule in source_rules) and not any(
            rule.get(field) for rule in reconciled_rules
        ):
            issues.append(
                f"reconciled rules lost every supported {field!r} value"
            )
    return issues


def enrich_atomic_rules_from_odrl(
    atomic_rules: list[dict[str, str]],
    odrl_rules: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Fill safe actor/asset gaps without copying action-specific constraints."""
    enriched: list[dict[str, str]] = []
    for rule in atomic_rules:
        result = dict(rule)
        if odrl_rules:
            score, candidate = max(
                (
                    (_rule_similarity(rule, odrl_rule), odrl_rule)
                    for odrl_rule in odrl_rules
                ),
                key=lambda item: item[0],
            )
            if score >= 0.35:
                # These fields normally govern the coordinated clause as a
                # whole. Constraints and purposes can modify only one action,
                # so they must remain from the atomic extraction.
                for field in ("assignee", "asset", "rule_type"):
                    if not result[field] and candidate[field]:
                        result[field] = candidate[field]
        enriched.append(result)
    return enriched


async def reconcile_policy_extractions(
    policy_text: str,
    odrl_rules: list[dict[str, str]],
    llm_rules: list[dict[str, str]],
    comparison: dict[str, Any],
) -> list[dict[str, str]]:
    """

    Use a source-grounded AI pass to resolve material disagreement.
    It is a third AI pass. It does not decide whether the policy complies with law. It decides what the policy itself says and produces one  reliable, atomic set of policy rules.

     ## Why reconciliation is necessary

      Consider this policy:
      The document may be downloaded once and retained for 30 days for academic research only.
      The ODRL endpoint might return one combined rule:
      {
        "action": "may be downloaded",
        "asset": "The document",
        "constraint": "for 30 days",
        "purpose": "for academic research only"
      }

      This result has problems:
      - downloading and retaining are different actions;
      - once belongs to downloading;
      - 30 days belongs to retention;
      - one combined rule cannot represent those constraints accurately.

      The independent AI might produce two atomic rules:

          [
            {
              "action": "download",
              "constraint": "once"
            },
            {
              "action": "retain",
              "constraint": "30 days"
            }
          ]

      But it might omit the asset or format the purpose differently.

      Reconciliation uses the strengths of both results while checking everything against the original policy.

      ## Inputs

      async def reconcile_policy_extractions(
          policy_text,
          odrl_rules,
          llm_rules,
          comparison,
      )

      It receives four pieces of information.

      ### policy_text
      The authoritative source:
      The document may be downloaded once and retained for 30 days for academic research only.
      This has priority over both extraction results.
      ### odrl_rules
      Rules returned by the ODRL endpoint.
      ### llm_rules
      Rules produced independently by the AI model.
      ### comparison
      The deterministic comparison result, including:

      {
        "matches": [...],
        "odrl_only_rule_indices": [],
        "llm_only_rule_indices": [1],
        "requires_reconciliation": true,
        "review_required": true
      }
      This helps the reconciliation model understand exactly where the two extractors disagree.


     For example, it should return something similar to:
      [
          {
              "source_text": (
                  "The document may be downloaded once and retained "
                  "for 30 days for academic research only."
              ),
              "assignee": "",
              "action": "download",
              "asset": "The document",
              "rule_type": "permission",
              "constraint": "once",
              "purpose": "academic research only",
          },
          {
              "source_text": (
                  "The document may be downloaded once and retained "
                  "for 30 days for academic research only."
              ),
              "assignee": "",
              "action": "retain",
              "asset": "The document",
              "rule_type": "permission",
              "constraint": "30 days",
              "purpose": "academic research only",
          },
      ]




    """

    prompt = load_prompt(
        "policy_reconciliation",
        policy_text=policy_text,
        odrl_result=json.dumps(
            {"rules": odrl_rules}, ensure_ascii=False, indent=2
        ),
        llm_result=json.dumps(
            {"rules": llm_rules}, ensure_ascii=False, indent=2
        ),
        comparison=json.dumps(comparison, ensure_ascii=False, indent=2),
    )

    """
    
    """

    model = get_llm_client(json_mode=True)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    parsed = _parse_json_object(response.content, "Policy reconciliation AI")
    reconciled = normalize_policy_rules(parsed.get("rules"))
    if not reconciled and (odrl_rules or llm_rules):
        raise ValueError(
            "Policy reconciliation removed every previously extracted rule."
        )
    return reconciled


async def _timed_call(
    awaitable: Awaitable[dict[str, Any]],
) -> tuple[dict[str, Any] | None, Exception | None, int]:
    """Run one extraction and retain its latency even when it fails."""
    started = time.perf_counter()
    try:
        result = await awaitable
        return result, None, int((time.perf_counter() - started) * 1000)
    except Exception as exc:  # The caller records each external failure.
        return None, exc, int((time.perf_counter() - started) * 1000)


def _audit_log_path() -> Path:
    """Resolve the dedicated policy-extraction audit file."""
    configured = os.getenv("ODRL_AUDIT_LOG_PATH", "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path(PROJECT_ROOT) / "logs" / "odrl_extraction.log"
    )


def write_odrl_audit_record(record: dict[str, Any]) -> str:
    """Append one complete JSON record and return the audit file path."""
    if os.getenv("ODRL_AUDIT_LOG_ENABLED", "True").lower() != "true":
        return ""
    log_path = _audit_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    with AUDIT_LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"{line}\n")
    return str(log_path)


def _assign_policy_rule_ids(
    rules: list[dict[str, str]],
    source: str,
) -> list[dict[str, str]]:
    """Add stable request-local identifiers and reconciliation provenance."""
    return [
        {
            "policy_rule_id": f"policy-rule-{index}",
            **rule,
            "selected_from": source,
        }
        for index, rule in enumerate(rules, start=1)
    ]


async def node_policy_understanding(
    state: QueryGraphState,
) -> dict[str, Any]:

    """Extract, compare, reconcile, and audit structured policy information."""

    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))

    policy_text = str(
        state.get("policy_text") or state.get("original_query") or ""
    ).strip()

    if not session_id:
        raise ValueError("node_policy_understanding requires a session_id.")
    if not policy_text:
        raise ValueError("node_policy_understanding requires policy text.")
    if len(policy_text) > MAX_POLICY_CHARACTERS:
        raise ValueError(
            "Policy text exceeds the policy-understanding limit of "
            f"{MAX_POLICY_CHARACTERS} characters."
        )

    add_running_task(session_id, function_name, is_stream)
    extraction_id = str(uuid4())
    logger.info(
        f">>> [{function_name}] started: session_id={session_id}, "
        f"extraction_id={extraction_id}, characters={len(policy_text)}"
    )

    try:
        # extract odrl components using api
        odrl_call = _timed_call(aextract_odrl_components(policy_text))

        # using AI to extract odrl components using api
        llm_call = _timed_call(extract_policy_independently_with_ai(policy_text))
        # parallel
        odrl_timed, llm_timed = await asyncio.gather(odrl_call, llm_call)

        odrl_result, odrl_error, odrl_latency = odrl_timed
        llm_result, llm_error, llm_latency = llm_timed

        errors: list[str] = []
        if odrl_error is not None:
            errors.append(
                f"odrl_endpoint: {type(odrl_error).__name__}: {odrl_error}"
            )
        if llm_error is not None:
            errors.append(
                f"independent_ai: {type(llm_error).__name__}: {llm_error}"
            )

        odrl_rules = normalize_policy_rules(
            (odrl_result or {"rules": []}).get("rules", [])
        )

        llm_rules = normalize_policy_rules(
            (llm_result or {"rules": []}).get("rules", [])
        )

        raw_policy_context = normalize_policy_context(
            (llm_result or {}).get("policy_context")
        )

        policy_context, context_warnings = validate_policy_context_evidence(
            raw_policy_context,
            policy_text,
        )

        comparison = compare_policy_extractions(
            odrl_rules,
            llm_rules,
            errors=errors,
        )

        source_validation_issues = [
            *_rule_source_issues(odrl_rules, policy_text, "odrl"),
            *_rule_source_issues(llm_rules, policy_text, "independent_ai"),
        ]
        """
        
         It is equivalent to the more readable version:

          odrl_source_issues = _rule_source_issues(
              odrl_rules,
              policy_text,
              "odrl",
          )
        
          ai_source_issues = _rule_source_issues(
              llm_rules,
              policy_text,
              "independent_ai",
          )
        
          source_validation_issues = (
              odrl_source_issues + ai_source_issues
          )

        """
        comparison["source_validation_issues"] = source_validation_issues
        comparison["context_validation_warnings"] = context_warnings

        if source_validation_issues:
            comparison["review_required"] = True
            if not errors:
                comparison["requires_reconciliation"] = True

        reconciliation_method = "automatic_agreement"
        reconciliation_error = ""

        if odrl_error is not None and llm_error is not None:
            validated_rules: list[dict[str, str]] = []
            selected_from = "none"

        elif comparison["requires_reconciliation"] and os.getenv("ODRL_AI_RECONCILIATION_ENABLED", "True").lower() == "true":

            quality_issues: list[str] = []
            try:

                #
                reconciled = await reconcile_policy_extractions(
                    policy_text,
                    odrl_rules,
                    llm_rules,
                    comparison,
                )
                quality_issues = reconciliation_quality_issues(
                    reconciled,
                    odrl_rules,
                    llm_rules,
                )
                comparison["reconciliation_quality_issues"] = quality_issues
                if quality_issues:
                    raise ValueError(
                        "Reconciliation quality validation failed: "
                        + " | ".join(quality_issues)
                    )
                validated_rules = reconciled
                selected_from = "ai_reconciliation"
                reconciliation_method = "ai_reconciliation"
            except Exception as exc:
                reconciliation_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                comparison["review_required"] = True
                validated_rules = (
                    enrich_atomic_rules_from_odrl(llm_rules, odrl_rules)
                    if llm_rules
                    else odrl_rules
                )
                selected_from = (
                    "independent_ai_fallback_enriched"
                    if llm_rules and odrl_rules
                    else (
                        "independent_ai_fallback"
                        if llm_rules
                        else "odrl_fallback"
                    )
                )
                reconciliation_method = (
                    "fallback_after_reconciliation_quality_failure"
                    if quality_issues
                    else "fallback_after_reconciliation_error"
                )
        elif llm_error is None:
            validated_rules = llm_rules
            selected_from = "independent_ai"
            if odrl_error is not None:
                reconciliation_method = "independent_ai_degraded_mode"
            else:
                reconciliation_method = (
                    "independent_ai_without_reconciliation"
                    if comparison["requires_reconciliation"]
                    else "automatic_agreement"
                )
        else:
            validated_rules = odrl_rules
            selected_from = "odrl_endpoint"
            reconciliation_method = "odrl_degraded_mode"

        final_source_issues = _rule_source_issues(
            validated_rules,
            policy_text,
            "validated",
        )
        comparison["validated_source_issues"] = final_source_issues
        if final_source_issues:
            comparison["review_required"] = True
            validated_rules = _source_grounded_rules(
                validated_rules,
                policy_text,
            )
            if not validated_rules:
                # A reconciliation pass can occasionally return paraphrased
                # source_text. Recover only independently extracted rules with
                # exact policy evidence; never send an unsupported rule onward.
                fallback_candidates = normalize_policy_rules(
                    [*llm_rules, *odrl_rules]
                )
                validated_rules = _source_grounded_rules(
                    fallback_candidates,
                    policy_text,
                )
                if validated_rules:
                    selected_from = "source_grounded_extraction_fallback"
                    reconciliation_method = (
                        "fallback_after_source_validation"
                    )
        final_rules = _assign_policy_rule_ids(
            validated_rules,
            selected_from,
        )
        include_source = os.getenv(
            "ODRL_AUDIT_INCLUDE_SOURCE_TEXT", "True"
        ).lower() == "true"


        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extraction_id": extraction_id,
            "session_id": session_id,
            "policy_hash": hashlib.sha256(
                policy_text.encode("utf-8")
            ).hexdigest(),
            "source_text": policy_text if include_source else "",
            "odrl_extraction": {"rules": odrl_rules},
            "independent_llm_extraction": {
                "rules": llm_rules,
                "policy_context": raw_policy_context,
            },
            "validated_policy_context": policy_context,
            "comparison": comparison,
            "validated_rules": final_rules,
            "reconciliation_method": reconciliation_method,
            "reconciliation_error": reconciliation_error,
            "metadata": {
                "odrl_latency_ms": odrl_latency,
                "llm_latency_ms": llm_latency,
                "prompt_version": "policy-understanding-v1",
                "llm_model": llm_config.llm_model,
                "odrl_endpoint": get_odrl_endpoint(),
                "odrl_model_version": os.getenv(
                    "ODRL_EXTRACTION_MODEL_VERSION", ""
                ),
            },
            "errors": errors,
        }
        audit_path = write_odrl_audit_record(audit_record)

        if odrl_error is not None and llm_error is not None:
            raise RuntimeError(
                "Both policy extractors failed: " + " | ".join(errors)
            )

        update = {
            "task_type": "policy_compliance",
            "policy_text": policy_text,
            "odrl_extracted_rules": odrl_rules,
            "llm_extracted_rules": llm_rules,
            "policy_extraction_comparison": comparison,
            "validated_policy_rules": final_rules,
            "policy_context": policy_context,
            "policy_understanding_errors": [
                *errors,
                *context_warnings,
                *source_validation_issues,
                *final_source_issues,
                *(
                    [f"reconciliation: {reconciliation_error}"]
                    if reconciliation_error
                    else []
                ),
            ],
            "odrl_audit_log_path": audit_path,
        }
        add_done_task(session_id, function_name, is_stream)
        logger.info(
            f">>> [{function_name}] finished: session_id={session_id}, "
            f"odrl_rules={len(odrl_rules)}, llm_rules={len(llm_rules)}, "
            f"validated_rules={len(final_rules)}, "
            f"review_required={comparison['review_required']}"
        )
        return update
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}, "
            f"extraction_id={extraction_id}"
        )
        raise
