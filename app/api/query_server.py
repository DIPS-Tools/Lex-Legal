
# Query API endpoints: chat page, query submission, SSE streaming,
# conversation-history retrieval, and conversation-history deletion.
import uuid
from fastapi import APIRouter, Depends, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import *
from app.utils.sse_utils import (
    create_sse_queue,
    get_sse_queue,
    push_to_session,
    SSEEvent,
    sse_generator,
)
from app.clients.mongo_utils import (
    clear_history,
    create_conversation,
    delete_conversation,
    ensure_conversation,
    get_recent_messages,
    list_conversations,
    require_owned_conversation,
)
from app.query_process.agent.query_main_graph import query_app
from app.api.authentication import (
    AuthenticatedUser,
    require_authenticated_user,
)


query_router = APIRouter()


# Return the standalone legal-compliance chat template.
@query_router.get("/chat.html", include_in_schema=False)
async def chat_html():
    # Resolve the template from the project root rather than the process CWD.
    chat_html_path  = PROJECT_ROOT / 'lex-legal_frontend' / 'templates' / 'legal-compliance.html'
    if not chat_html_path.exists():
        raise HTTPException(
            status_code=404,
            detail="The legal-compliance chat template was not found.",
        )
    return FileResponse(chat_html_path)


# Validate query-submission parameters received from the frontend.
# Object of the receiving parameter from front end
class QueryRequest(BaseModel):
    query: str = Field(..., title="The query content must be passed on.")
    session_id:str = Field(None, title="Session ID: It is not mandatory to pass this parameter; the backend will generate a unique UUID automatically.")
    is_stream:bool = Field(False, title="Whether to return results in a streaming manner")


class ConversationCreateRequest(BaseModel):
    """Optional display title for a newly created conversation."""

    title: str = Field(default="", max_length=120)


def _conversation_not_found() -> HTTPException:
    """Use one response for absent and foreign sessions to avoid disclosure."""
    return HTTPException(status_code=404, detail="Conversation was not found.")


async def _require_owned_session(
    identity: AuthenticatedUser,
    session_id: str,
) -> None:
    try:
        await require_owned_conversation(identity.subject, session_id)
    except (LookupError, PermissionError) as exc:
        raise _conversation_not_found() from exc


async def run_query_graph(query: str, session_id: str, is_stream: bool):
    # A session ID represents a conversation and is intentionally reused for
    # follow-up questions. Task progress, however, belongs to one query only.
    # Clear the previous query's running/done lists and cached result before
    # publishing the new processing state; MongoDB conversation history is not
    # affected by this in-memory reset.
    clear_task(session_id)
    update_task_status(session_id, "processing", is_stream)

    state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream
    )
    try:
        await query_app.ainvoke(state)
        # Publish the terminal task status to SSE clients when streaming.
        update_task_status(session_id, "completed", is_stream)
    except Exception as e:
        logger.exception(
            f"Query workflow failed: session_id={session_id}, error={e}"
        )
        # Mark the task as failed before publishing the error event.
        update_task_status(session_id, "failed", is_stream)
        # Send a typed SSE error event to the active session stream.
        push_to_session(session_id, SSEEvent.ERROR, {"error": str(e)})
    finally:
        # Each POST creates one stream lifecycle. Closing it after the terminal
        # status prevents a completed query's generator from surviving into a
        # later query that reuses the same conversation session ID.
        if is_stream:
            push_to_session(session_id, SSEEvent.CLOSE, {})


@query_router.get("/conversations", summary="List conversations")
async def conversations(
    limit: int = 50,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """Return the authenticated user's recently active conversations."""
    return {
        "items": await list_conversations(identity.subject, limit),
    }


@query_router.post(
    "/conversations",
    summary="Create conversation",
    status_code=201,
)
async def new_conversation(
    request: ConversationCreateRequest,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """Create an empty conversation before its first legal query."""
    session_id = f"legal-{uuid.uuid4()}"
    return await create_conversation(
        identity.subject,
        session_id,
        request.title,
    )


@query_router.delete(
    "/conversations/{session_id}",
    summary="Delete conversation",
)
async def remove_conversation(
    session_id: str,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """Delete one owned conversation and its complete message history."""
    try:
        deleted_count = await delete_conversation(
            identity.subject,
            session_id,
        )
    except (LookupError, PermissionError) as exc:
        raise _conversation_not_found() from exc
    return {
        "session_id": session_id,
        "deleted_messages": deleted_count,
    }


@query_router.post("")  # Client -> LangGraph -> retrieval -> response.
async def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """
    :param request:
    :param background_tasks: Runs the graph after returning a streaming task ID.
    :return:
    """


    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
    if is_stream and get_task_status(session_id) == TASK_STATUS_PROCESSING:
        raise HTTPException(
            status_code=409,
            detail=(
                "A query is already running for this conversation. "
                "Wait for its response stream to finish before submitting "
                "another query."
            ),
        )
    try:
        await ensure_conversation(
            identity.subject,
            session_id,
            title_seed=query,
        )
    except PermissionError as exc:
        raise _conversation_not_found() from exc
    # Streaming mode immediately returns the session ID, runs the graph in the
    # background, and publishes progress and answer events through SSE.
    if is_stream:
        # Register one in-memory event queue for this session before scheduling
        # the graph, ensuring early progress events are not lost.
        create_sse_queue(session_id)
        # FastAPI executes this task after the HTTP response has been sent.
        background_tasks.add_task(run_query_graph, query, session_id, is_stream)

        logger.info(f"For query:{query}, Asynchronous and streaming processing have been enabled!")
        return {
            "session_id": session_id,
            "message": "The query is being processed."
        }
    else:
        # Synchronous mode waits for the complete graph result.
        await run_query_graph(query, session_id, is_stream)
        # node_answer_output stores the final non-streaming answer by session.
        answer = get_task_result(session_id,"answer")
        validated_policy_rules = get_task_result(
            session_id,
            "validated_policy_rules",
            [],
        )
        logger.info(
            f"Synchronous query completed: session_id={session_id}, "
            f"answer_characters={len(answer)}"
        )
        return {
            "answer": answer,
            "session_id": session_id,
            "message": "The query completed successfully.",
            "done_list": [],
            "validated_policy_rules": validated_policy_rules,
        }


@query_router.get("/stream/{session_id}")
async def stream(
    session_id: str,
    request: Request,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """
    :param session_id:
    :param request: Native request used to detect client disconnection.
    :return:
    """
    await _require_owned_session(identity, session_id)
    if get_sse_queue(session_id) is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "No active query stream exists for this conversation. "
                "Submit a streaming query before opening its event stream."
            ),
        )
    logger.info(
        f"SSE connection established: session_id={session_id}"
    )
    return StreamingResponse(
        sse_generator(session_id,request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@query_router.get("/history/{session_id}")
async def history(
    session_id: str,
    limit: int = 10,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """
    :param session_id:
    :param limit: Maximum number of recent messages to return.
    :return:
    """
    await _require_owned_session(identity, session_id)
    # Retrieve recent messages in chronological display order.
    chats = await get_recent_messages(session_id, limit)
    # MongoDB ObjectId values are normalized by the history repository before
    # FastAPI serializes this response.
    # items = []
    # for chat in chats:
    #     items.append(chat)
    logger.info(
        f"Conversation history retrieved: session_id={session_id}, "
        f"messages={len(chats)}"
    )
    return {
        "session_id":session_id,
        "items": chats
    }

@query_router.delete("/history/{session_id}")
async def delete_history(
    session_id: str,
    identity: AuthenticatedUser = Depends(require_authenticated_user),
):
    """
    :param session_id:
    :return:
    """
    await _require_owned_session(identity, session_id)
    # Delete only persisted chat history for this conversation session.
    delete_count = await clear_history(session_id)
    logger.info(
        f"Conversation history deleted: session_id={session_id}, "
        f"deleted_messages={delete_count}"
    )
    return {
        "deleted_count":delete_count,
        "message": f"Conversation history deleted for session {session_id}."
    }
