"""Serialize extracted rules when the user explicitly requests ODRL JSON-LD."""

from __future__ import annotations

import sys
from typing import Any, cast

from app.core.logger import logger
from app.query_process.agent.state import ODRLState, QueryGraphState
from app.utils.serialization_odrl import (
    serialize_odrl_document,
)
from app.utils.task_utils import add_done_task, add_running_task


async def node_odrl_serialization(
    state: QueryGraphState,
) -> dict[str, Any]:
    """Attempt formal serialization without losing validated extracted rules."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_odrl_serialization requires a session_id.")

    current = dict(state.get("odrl_state") or {})
    rules = [
        dict(rule)
        for rule in current.get("validated_policy_rules", [])
        if isinstance(rule, dict)
    ]
    if not current:
        raise ValueError(
            "node_odrl_serialization must run after node_odrl_extraction."
        )

    add_running_task(session_id, function_name, is_stream)
    try:
        document = serialize_odrl_document(rules)
        current.update(
            odrl_document=document,
            serialization_status="generated",
            serialization_errors=[],
        )
    except Exception as exc:
        current.update(
            odrl_document=None,
            serialization_status="validation_failed",
            serialization_errors=[f"{type(exc).__name__}: {exc}"],
        )
        logger.exception(
            f"ODRL serialization failed: session_id={session_id}"
        )

    add_done_task(session_id, function_name, is_stream)
    return {"odrl_state": cast(ODRLState, current)}
