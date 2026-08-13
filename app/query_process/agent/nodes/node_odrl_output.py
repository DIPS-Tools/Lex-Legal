"""Return validated ODRL extraction results without legal retrieval."""

from __future__ import annotations

import sys
from typing import Any

from app.clients.mongo_utils import save_chat_message
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.sse_utils import SSEEvent, push_to_session
from app.utils.task_utils import add_done_task, add_running_task, set_task_result


def _format_validation_report(validation_result: dict[str, Any]) -> str:
    """Format the engine's structured result as a readable Markdown report."""
    valid = validation_result.get("valid")
    diagnostic = str(
        validation_result.get("diagnostic_report") or ""
    ).strip()
    if not diagnostic:
        return ""

    # Preserve an engine-supplied report envelope instead of nesting it.
    if "REPORT START" in diagnostic and "REPORT END" in diagnostic:
        report = diagnostic
    else:
        compliant = "True" if valid is True else "False"
        report = (
            "REPORT START\n"
            f"Is it compliant? {compliant}\n"
            f"{diagnostic}\n"
            "REPORT END"
        )
    return f"ODRL validation report:\n\n```text\n{report}\n```"


def _build_odrl_answer(
    rules: list[dict[str, Any]],
    status: str,
    errors: list[str],
    validation_status: str,
    validation_result: dict[str, Any],
    validation_errors: list[str],
) -> str:
    noun = "rule" if len(rules) == 1 else "rules"
    answer = f"Extracted {len(rules)} validated ODRL-style policy {noun}."
    if status == "generated":
        answer = f"{answer} A formal ODRL JSON-LD document was also generated."
    if status in {"not_implemented", "validation_failed"}:
        detail = errors[0] if errors else "Formal serialization is unavailable."
        return f"{answer}\n\nODRL JSON-LD note: {detail}"

    if validation_status == "valid":
        summary = f"{answer} The ODRL Engine validation passed."
        report = _format_validation_report(validation_result)
        return f"{summary}\n\n{report}" if report else summary
    if validation_status == "repaired":
        summary = (
            f"{answer} The initial document failed validation, was corrected "
            "once using the diagnostic report, and then passed revalidation."
        )
        report = _format_validation_report(validation_result)
        return f"{summary}\n\n{report}" if report else summary
    if validation_status == "invalid":
        report = _format_validation_report(validation_result)
        detail = report or (
            validation_errors[0]
            if validation_errors
            else "The formal document did not pass validation."
        )
        return f"{answer}\n\nODRL validation failed.\n\n{detail}"
    if validation_status == "error":
        detail = (
            validation_errors[0]
            if validation_errors
            else "The ODRL validation service was unavailable."
        )
        return f"{answer}\n\nODRL validation could not be completed: {detail}"
    return answer


async def node_odrl_output(state: QueryGraphState) -> dict[str, Any]:
    """Publish the ODRL branch result and terminate without retrieval."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_odrl_output requires a session_id.")

    odrl_state = state.get("odrl_state") or {}
    rules = [
        dict(rule)
        for rule in odrl_state.get("validated_policy_rules", [])
        if isinstance(rule, dict)
    ]
    document = odrl_state.get("odrl_document")
    status = str(odrl_state.get("serialization_status") or "not_requested")
    errors = [str(item) for item in odrl_state.get("serialization_errors", [])]
    validation_status = str(odrl_state.get("validation_status") or "not_run")
    validation_result = dict(odrl_state.get("validation_result") or {})
    validation_errors = [
        str(item) for item in odrl_state.get("validation_errors", [])
    ]
    answer = _build_odrl_answer(
        rules,
        status,
        errors,
        validation_status,
        validation_result,
        validation_errors,
    )

    add_running_task(session_id, function_name, is_stream)
    if is_stream:
        push_to_session(session_id, SSEEvent.DELTA, {"delta": answer})
        push_to_session(
            session_id,
            SSEEvent.FINAL,
            {
                "answer": answer,
                "status": "completed",
                "image_urls": [],
                "validated_policy_rules": rules,
                "odrl_document": document,
            },
        )
    else:
        set_task_result(session_id, "answer", answer)
        set_task_result(session_id, "validated_policy_rules", rules)
        set_task_result(session_id, "odrl_document", document)

    await save_chat_message(
        session_id=session_id,
        role="assistant",
        text=answer,
        rewritten_query=str(state.get("original_query") or ""),
        query_key_terms=state.get("query_key_terms", []),
        image_urls=[],
        validated_policy_rules=rules,
        odrl_document=document if isinstance(document, dict) else None,
    )
    add_done_task(session_id, function_name, is_stream)
    logger.info(
        f">>> [{function_name}] finished: session_id={session_id}, "
        f"validated_rules={len(rules)}, serialization_status={status}"
    )
    return {
        "answer": answer,
        "image_urls": [],
        "validated_policy_rules": rules,
        "odrl_state": odrl_state,
    }
