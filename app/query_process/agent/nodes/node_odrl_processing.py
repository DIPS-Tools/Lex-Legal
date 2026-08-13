"""Validate and, once if necessary, repair a serialized ODRL policy."""

from __future__ import annotations

import json
import re
import sys
from typing import Any, cast

from langchain_core.messages import HumanMessage

from app.core.logger import logger
from app.llm.llm_utils import get_llm_client
from app.prompt.load_prompt import load_prompt
from app.query_process.agent.state import ODRLState, QueryGraphState
from app.tools.odrl_validation_tool import avalidate_odrl_policy
from app.utils.task_utils import add_done_task, add_running_task


def _response_content_to_text(content: Any) -> str:
    """Normalize common LangChain response content into plain text."""
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


def _parse_json_document(content: Any) -> dict[str, Any]:
    """Parse one repaired JSON-LD object, tolerating an outer code fence."""
    text = _response_content_to_text(content)
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|jsonld)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"ODRL repair AI returned invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ValueError("ODRL repair AI must return one JSON object.")
    return parsed


async def repair_odrl_document_with_ai(
    document: dict[str, Any],
    diagnostic_report: str,
    *,
    source_text: str = "",
) -> dict[str, Any]:
    """Use one diagnostic-grounded AI call to repair formal JSON-LD syntax."""
    prompt = load_prompt(
        "odrl_validation_repair",
        source_text=source_text,
        diagnostic_report=diagnostic_report,
        odrl_document=json.dumps(document, ensure_ascii=False, indent=2),
    )
    model = get_llm_client(json_mode=True)
    response = await model.ainvoke([HumanMessage(content=prompt)])
    return _parse_json_document(response.content)


async def node_odrl_processing(state: QueryGraphState) -> dict[str, Any]:
    """Validate serialized ODRL and make at most one AI repair attempt."""
    function_name = sys._getframe().f_code.co_name
    session_id = str(state.get("session_id") or "").strip()
    is_stream = bool(state.get("is_stream", False))
    if not session_id:
        raise ValueError("node_odrl_processing requires a session_id.")

    current = dict(state.get("odrl_state") or {})
    document = current.get("odrl_document")
    if not isinstance(document, dict) or not document:
        current.update(
            validation_status="error",
            validation_result={},
            initial_validation_result={},
            validation_repair_attempted=False,
            validation_errors=[
                "ODRL validation requires a serialized odrl_document."
            ],
        )
        logger.warning(
            f">>> [{function_name}] skipped: session_id={session_id}, "
            "reason=missing odrl_document"
        )
        return {"odrl_state": cast(ODRLState, current)}

    add_running_task(session_id, function_name, is_stream)

    try:
        # one of processing functions, avalidate_odrl_policy
        initial_result = await avalidate_odrl_policy(document)
        current.update(
            initial_validation_result=dict(initial_result),
            validation_result=dict(initial_result),
            validation_errors=[],
        )

        if initial_result["valid"]:
            current.update(
                validation_status="valid",
                original_odrl_document=None,
                validation_repair_attempted=False,
            )
        else:
            # Keep the original for auditing. Only replace odrl_document after
            # the repaired candidate passes the same external validator.
            current.update(
                validation_status="invalid",
                original_odrl_document=dict(document),
                validation_repair_attempted=True,
            )
            repaired = await repair_odrl_document_with_ai(
                document,
                initial_result["diagnostic_report"],
                source_text=str(current.get("source_text") or ""),
            )
            repaired_result = await avalidate_odrl_policy(repaired)
            current["validation_result"] = dict(repaired_result)
            if repaired_result["valid"]:
                current.update(
                    odrl_document=repaired,
                    validation_status="repaired",
                )
            else:
                current["validation_errors"] = [
                    "The AI-repaired ODRL document did not pass validation."
                ]
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        current["validation_errors"] = [
            *current.get("validation_errors", []),
            error,
        ]
        # Preserve an explicit invalid result when repair failed; otherwise
        # distinguish service/processing errors from a SHACL-invalid policy.
        if current.get("initial_validation_result", {}).get("valid") is not False:
            current["validation_status"] = "error"
        logger.exception(
            f"ODRL validation or repair failed: session_id={session_id}"
        )
    finally:
        add_done_task(session_id, function_name, is_stream)



    logger.info(
        f">>> [{function_name}] finished: session_id={session_id}, "
        f"validation_status={current.get('validation_status')}"
    )
    return {"odrl_state": cast(ODRLState, current)}
