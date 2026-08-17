"""Create the canonical ODRL extraction state from natural-language policy text."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sys
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.config.llm_config import llm_config
from app.core.logger import logger
from app.query_process.agent.nodes.node_policy_understanding import (
    MAX_POLICY_CHARACTERS,
    _assign_policy_rule_ids,
    _rule_source_issues,
    _source_grounded_rules,
    _timed_call,
    compare_policy_extractions,
    enrich_atomic_rules_from_odrl,
    extract_policy_independently_with_ai,
    normalize_policy_context,
    normalize_policy_rules,
    reconcile_policy_extractions,
    reconciliation_quality_issues,
    validate_policy_context_evidence,
    write_odrl_audit_record,
)
from app.query_process.agent.state import ODRLState, QueryGraphState
from app.tools.oder_extraction_tool import (
    aextract_odrl_components,
    get_odrl_endpoint,
)
from app.utils.task_utils import add_done_task, add_running_task


async def node_odrl_extraction(
    state: QueryGraphState,
) -> dict[str, Any]:

    """Extract, compare, reconcile, validate, and audit policy rules."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))

    policy_text = str(
        state.get("policy_text") or state.get("original_query") or ""
    ).strip()

    if not session_id:
        raise ValueError("node_odrl_extraction requires a session_id.")
    if not policy_text:
        raise ValueError("node_odrl_extraction requires policy text.")
    if len(policy_text) > MAX_POLICY_CHARACTERS:
        raise ValueError(
            "Policy text exceeds the ODRL extraction limit of "
            f"{MAX_POLICY_CHARACTERS} characters."
        )

    add_running_task(session_id, function_name, is_stream)
    extraction_id = str(uuid4())
    logger.info(
        f">>> [{function_name}] started: session_id={session_id}, "
        f"extraction_id={extraction_id}, characters={len(policy_text)}"
    )

    try:
        # call the API extracting the ODRL components
        endpoint_call = _timed_call(aextract_odrl_components(policy_text))
        # using AI model to extracting ODRL components, also extracting policy-context
        independent_ai_call = _timed_call(
            extract_policy_independently_with_ai(policy_text)
        )
        # parallel calling
        endpoint_timed, independent_ai_timed = await asyncio.gather(
            endpoint_call, independent_ai_call,)

        endpoint_result, endpoint_error, endpoint_latency = endpoint_timed
        ai_result, ai_error, llm_latency = independent_ai_timed

        errors: list[str] = []
        if endpoint_error is not None:
            errors.append(
                "odrl_endpoint: "
                f"{type(endpoint_error).__name__}: {endpoint_error}"
            )
        if ai_error is not None:
            errors.append(
                f"independent_ai: {type(ai_error).__name__}: {ai_error}"
            )

        endpoint_rules = normalize_policy_rules(
            (endpoint_result or {"rules": []}).get("rules", [])
        )
        independent_ai_rules = normalize_policy_rules(
            (ai_result or {"rules": []}).get("rules", [])
        )
        raw_policy_context = normalize_policy_context(
            (ai_result or {}).get("policy_context")
        )
        policy_context, context_warnings = validate_policy_context_evidence(
            raw_policy_context,
            policy_text,
        )
        # compare endpoint extracted rules and AI-extracted rules
        comparison = compare_policy_extractions(
            endpoint_rules,
            independent_ai_rules,
            errors=errors,
        )
        source_validation_issues = [
            *_rule_source_issues(endpoint_rules, policy_text, "odrl"),
            *_rule_source_issues(
                independent_ai_rules,
                policy_text,
                "independent_ai",
            ),
        ]
        comparison["source_validation_issues"] = source_validation_issues
        comparison["context_validation_warnings"] = context_warnings
        if source_validation_issues:
            comparison["review_required"] = True
            if not errors:
                comparison["requires_reconciliation"] = True

        reconciliation_method = "automatic_agreement"
        reconciliation_error = ""
        selected_from = "none"

        if endpoint_error is not None and ai_error is not None:
            validated_rules: list[dict[str, str]] = []
        elif (
            comparison["requires_reconciliation"]
            and os.getenv(
                "ODRL_AI_RECONCILIATION_ENABLED",
                "True",
            ).lower() == "true"
        ):
            quality_issues: list[str] = []
            try:
                reconciled = await reconcile_policy_extractions(
                    policy_text,
                    endpoint_rules,
                    independent_ai_rules,
                    comparison,
                )
                quality_issues = reconciliation_quality_issues(
                    reconciled,
                    endpoint_rules,
                    independent_ai_rules,
                )
                comparison["reconciliation_quality_issues"] = quality_issues
                if quality_issues:
                    raise ValueError(
                        "Reconciliation quality validation failed: "
                        + " | ".join(quality_issues)
                    )
                # valid rules combining two extracted results
                validated_rules = reconciled

                selected_from = "ai_reconciliation"
                reconciliation_method = "ai_reconciliation"
            except Exception as exc:

                reconciliation_error = f"{type(exc).__name__}: {exc}"
                comparison["review_required"] = True
                validated_rules = (
                    enrich_atomic_rules_from_odrl(
                        independent_ai_rules,
                        endpoint_rules,
                    )
                    if independent_ai_rules
                    else endpoint_rules
                )
                selected_from = (
                    "independent_ai_fallback_enriched"
                    if independent_ai_rules and endpoint_rules
                    else (
                        "independent_ai_fallback"
                        if independent_ai_rules
                        else "odrl_fallback"
                    )
                )
                reconciliation_method = (
                    "fallback_after_reconciliation_quality_failure"
                    if quality_issues
                    else "fallback_after_reconciliation_error"
                )
        elif ai_error is None:
            validated_rules = independent_ai_rules
            selected_from = "independent_ai"
            if endpoint_error is not None:
                reconciliation_method = "independent_ai_degraded_mode"
            elif comparison["requires_reconciliation"]:
                reconciliation_method = (
                    "independent_ai_without_reconciliation"
                )
        else:
            validated_rules = endpoint_rules
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
                fallback_candidates = normalize_policy_rules(
                    [*independent_ai_rules, *endpoint_rules]
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
        all_errors = [
            *errors,
            *context_warnings,
            *source_validation_issues,
            *final_source_issues,
            *(
                [f"reconciliation: {reconciliation_error}"]
                if reconciliation_error
                else []
            ),
        ]

        audit_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "extraction_id": extraction_id,
            "session_id": session_id,
            "policy_hash": hashlib.sha256(
                policy_text.encode("utf-8")
            ).hexdigest(),
            "source_text": (
                policy_text
                if os.getenv(
                    "ODRL_AUDIT_INCLUDE_SOURCE_TEXT",
                    "True",
                ).lower() == "true"
                else ""
            ),
            "odrl_extraction": {"rules": endpoint_rules},
            "independent_llm_extraction": {
                "rules": independent_ai_rules,
                "policy_context": raw_policy_context,
            },
            "validated_policy_context": policy_context,
            "comparison": comparison,
            "validated_rules": final_rules,
            "reconciliation_method": reconciliation_method,
            "reconciliation_error": reconciliation_error,
            "metadata": {
                "odrl_latency_ms": endpoint_latency,
                "llm_latency_ms": llm_latency,
                "prompt_version": "policy-understanding-v1",
                "llm_model": llm_config.llm_model,
                "odrl_endpoint": get_odrl_endpoint(),
                "odrl_model_version": os.getenv(
                    "ODRL_EXTRACTION_MODEL_VERSION",
                    "",
                ),
            },
            "errors": errors,
        }
        audit_path = write_odrl_audit_record(audit_record)

        if endpoint_error is not None and ai_error is not None:
            raise RuntimeError(
                "Both policy extractors failed: " + " | ".join(errors)
            )
        # odrl_state, saving all related with ODRL
        odrl_state: ODRLState = {
            "source_text": policy_text,
            "validated_policy_rules": final_rules,
            # Placeholder populated by node_odrl_serialization.
            "odrl_jsonld_document": None,
            "endpoint_rules": endpoint_rules,
            "independent_ai_rules": independent_ai_rules,
            "comparison": comparison,
            "extracted_policy_context": policy_context,
            "extraction_errors": all_errors,
            "extraction_id": extraction_id,
            "reconciliation_method": reconciliation_method,
            "reconciliation_error": reconciliation_error,
            "audit_log_path": audit_path,
            "endpoint_latency_ms": endpoint_latency,
            "llm_latency_ms": llm_latency,
            "serialization_status": "not_requested",
            "serialization_errors": [],
            "validation_status": "not_run",
            "validation_result": {},
            "initial_validation_result": {},
            "original_odrl_jsonld_document": None,
            "validation_repair_attempted": False,
            "validation_errors": [],
        }

        add_done_task(session_id, function_name, is_stream)
        logger.info(
            f">>> [{function_name}] finished: session_id={session_id}, "
            f"endpoint_rules={len(endpoint_rules)}, "
            f"llm_rules={len(independent_ai_rules)}, "
            f"validated_rules={len(final_rules)}, "
            f"review_required={comparison['review_required']}"
        )
        return {
            "policy_text": policy_text,
            "odrl_state": odrl_state,
            # Temporary compatibility aliases. New consumers should read
            # odrl_state; these can be removed after the migration is complete.
            "odrl_extracted_rules": endpoint_rules,
            "llm_extracted_rules": independent_ai_rules,
            "policy_extraction_comparison": comparison,
            "validated_policy_rules": final_rules,
            "policy_understanding_errors": all_errors,
            "odrl_audit_log_path": audit_path,
        }
    except Exception:
        logger.exception(
            f">>> [{function_name}] failed: session_id={session_id}, "
            f"extraction_id={extraction_id}"
        )
        raise
