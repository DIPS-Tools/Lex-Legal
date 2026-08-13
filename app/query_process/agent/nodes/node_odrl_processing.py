"""Placeholder for future processing of serialized ODRL policies."""

from __future__ import annotations

import sys
from typing import Any

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_done_task, add_running_task


async def node_odrl_processing(state: QueryGraphState) -> dict[str, Any]:
    """Pass serialized ODRL through unchanged until business logic is added."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_odrl_processing requires a session_id.")

    add_running_task(session_id, function_name, is_stream)
    logger.info(
        f">>> [{function_name}] placeholder executed: session_id={session_id}"
    )
    add_done_task(session_id, function_name, is_stream)

    # Future ODRL validation, evaluation, transformation, or persistence can
    # be implemented here. Returning an empty update preserves QueryGraphState.
    return {}
