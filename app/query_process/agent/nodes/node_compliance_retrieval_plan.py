"""
Build rule-aware legal retrieval inputs from
structured policy state.
"""

from __future__ import annotations

import sys
from typing import Any

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


MAX_POLICY_RULES_FOR_RETRIEVAL = 50
MAX_COMBINED_QUERY_CHARACTERS = 8_000
MAX_COMPLIANCE_KEY_TERMS = 24


def _clean_text(value: Any) -> str:
    """Normalize one value for a retrieval query without changing meaning."""
    if not isinstance(value, (str, int, float)):
        return ""
    return " ".join(str(value).split())


def _context_values(policy_context: Any) -> list[str]:
    """Flatten supported context items into unique retrieval phrases."""
    if not isinstance(policy_context, dict):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for field, raw_items in policy_context.items():
        if field == "missing_context" or not isinstance(raw_items, list):
            continue
        for raw_item in raw_items:
            value = (
                raw_item.get("value")
                if isinstance(raw_item, dict)
                else raw_item
            )
            text = _clean_text(value)
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                values.append(text)
    return values


def _rule_description(rule: dict[str, Any]) -> str:
    """Render one atomic policy rule as a compact, source-grounded sentence."""
    rule_type = _clean_text(rule.get("rule_type")) or "rule"
    assignee = _clean_text(rule.get("assignee"))
    action = _clean_text(rule.get("action"))
    asset = _clean_text(rule.get("asset"))
    constraint = _clean_text(rule.get("constraint"))
    purpose = _clean_text(rule.get("purpose"))

    parts = [f"{rule_type} to {action}" if action else rule_type]
    if asset:
        parts.append(asset)
    if assignee:
        parts.append(f"for assignee {assignee}")
    if constraint:
        parts.append(f"subject to {constraint}")
    if purpose:
        parts.append(f"for purpose {purpose}")
    return " ".join(parts)


def _rule_key_terms(rule: dict[str, Any]) -> list[str]:
    """Create concise lexical terms from one structured policy rule."""
    rule_type = _clean_text(rule.get("rule_type"))
    action = _clean_text(rule.get("action"))
    asset = _clean_text(rule.get("asset"))
    assignee = _clean_text(rule.get("assignee"))
    constraint = _clean_text(rule.get("constraint"))
    purpose = _clean_text(rule.get("purpose"))
    candidates = [
        " ".join(part for part in (rule_type, action) if part),
        " ".join(part for part in (action, asset) if part),
        assignee,
        constraint,
        purpose,
    ]
    terms: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = _clean_text(candidate)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            terms.append(text)
    return terms


def build_compliance_retrieval_plan(
    policy_text: str,
    validated_rules: list[dict[str, Any]],
    policy_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create one legal-evidence request for every validated policy rule."""
    context_values = _context_values(policy_context)
    context_clause = (
        f" Relevant policy context: {', '.join(context_values[:12])}."
        if context_values
        else ""
    )
    plans: list[dict[str, Any]] = []
    for index, rule in enumerate(
        validated_rules[:MAX_POLICY_RULES_FOR_RETRIEVAL],
        start=1,
    ):
        if not isinstance(rule, dict):
            continue
        rule_id = _clean_text(rule.get("policy_rule_id")) or f"policy-rule-{index}"
        description = _rule_description(rule)
        query = (
            "What applicable legal requirements, safeguards, responsibilities, "
            "procedures, documentation, exceptions, and evidence govern this "
            f"policy {description}?{context_clause}"
        )
        plans.append({
            "policy_rule_id": rule_id,
            "policy_rule": rule,
            "query": query,
            "key_terms": _rule_key_terms(rule),
        })

    if not plans:
        # Preserve retrieval capability when both extractors legitimately find
        # no atomic deontic rule but the user still requested policy analysis.
        plans.append({
            "policy_rule_id": "policy-general",
            "policy_rule": {},
            "query": (
                "Identify the applicable legal requirements, safeguards, "
                "responsibilities, procedures, documentation, and evidence for "
                f"this policy text: {policy_text}{context_clause}"
            ),
            "key_terms": context_values[:MAX_COMPLIANCE_KEY_TERMS],
        })
    return plans


def _merge_key_terms(
    existing_terms: Any,
    plans: list[dict[str, Any]],
) -> list[str]:
    """Merge intent and policy-rule terms while preserving their order."""
    values: list[Any] = []
    if isinstance(existing_terms, list):
        values.extend(existing_terms)
    for plan in plans:
        values.extend(plan.get("key_terms", []))

    merged: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _clean_text(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            merged.append(text)
        if len(merged) >= MAX_COMPLIANCE_KEY_TERMS:
            break
    return merged


def _combined_retrieval_query(
    plans: list[dict[str, Any]],
) -> str:
    """Create the bounded single query consumed by current retrieval nodes."""
    sections = [
        "Assess the applicable legal requirements for the supplied policy. "
        "Retrieve provisions governing each policy rule, including required "
        "safeguards, responsibilities, procedures, documentation, exceptions, "
        "and evidence."
    ]
    sections.extend(
        f"{plan['policy_rule_id']}: {plan['query']}"
        for plan in plans
    )
    return "\n".join(sections)[:MAX_COMBINED_QUERY_CHARACTERS].rstrip()


def node_compliance_retrieval_plan(
    state: QueryGraphState,
) -> dict[str, Any]:
    """Prepare per-rule plans and existing single-query retrieval fields."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    policy_text = str(state.get("policy_text") or "").strip()
    if not session_id:
        raise ValueError("node_compliance_retrieval_plan requires a session_id.")
    if state.get("task_type") != "policy_compliance":
        raise ValueError(
            "node_compliance_retrieval_plan requires task_type='policy_compliance'."
        )
    if not policy_text:
        raise ValueError("Compliance retrieval planning requires policy text.")

    raw_rules = state.get("validated_policy_rules", [])
    if not isinstance(raw_rules, list):
        raise TypeError("validated_policy_rules must be a list.")
    policy_context = state.get("policy_context", {})
    if not isinstance(policy_context, dict):
        raise TypeError("policy_context must be a dictionary.")

    add_running_task(session_id, function_name, is_stream)
    logger.info(
        f">>> [{function_name}] started: session_id={session_id}, "
        f"validated_rules={len(raw_rules)}"
    )
    try:
        plans = build_compliance_retrieval_plan(
            policy_text,
            raw_rules,
            policy_context,
        )
        merged_terms = _merge_key_terms(
            state.get("query_key_terms", []),
            plans,
        )
        combined_query = _combined_retrieval_query(plans)
        add_done_task(session_id, function_name, is_stream)
        logger.info(
            f">>> [{function_name}] finished: session_id={session_id}, "
            f"retrieval_plans={len(plans)}, key_terms={len(merged_terms)}"
        )
        return {
            "compliance_retrieval_queries": plans,
            "rewritten_query": combined_query,
            "query_key_terms": merged_terms,
        }
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}"
        )
        raise
