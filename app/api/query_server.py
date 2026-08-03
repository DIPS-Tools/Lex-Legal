
# Query API endpoints: health, chat page, query submission, SSE streaming,
# conversation-history retrieval, and conversation-history deletion.
from pathlib import Path
import uuid
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
from app.core.logger import logger
from app.query_process.agent.state import create_query_default_state
from app.utils.path_util import PROJECT_ROOT

from app.utils.task_utils import *
from app.utils.sse_utils import create_sse_queue, SSEEvent, sse_generator
from app.clients.mongo_utils import get_recent_messages, clear_history
from app.query_process.agent.query_main_graph import query_app


# Create the FastAPI application.
app = FastAPI(title="Legal Complance", description="Legal Complance Assistant")

# Configure cross-origin access for the separately hosted Vue frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Report whether the query API process is available.
@app.get("/health")
async def health():
    logger.info("Query API health check completed successfully.")
    return {"status": "ok"}


# Return the standalone legal-compliance chat template.
@app.get("/chat.html")
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


@app.post("/query")  # Client -> LangGraph -> retrieval -> response.
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    :param request:
    :param background_tasks: Runs the graph after returning a streaming task ID.
    :return:
    """


    query = request.query
    session_id = request.session_id or str(uuid.uuid4())
    is_stream = request.is_stream
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
        logger.info(
            f"Synchronous query completed: session_id={session_id}, "
            f"answer_characters={len(answer)}"
        )
        return {
            "answer": answer,
            "session_id": session_id,
            "message": "The query completed successfully.",
            "done_list": []
        }


@app.get("/stream/{session_id}")
async def stream(session_id: str,request:Request):
    """
    :param session_id:
    :param request: Native request used to detect client disconnection.
    :return:
    """
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


@app.get("/history/{session_id}")
async def history(session_id: str,limit: int = 10):
    """
    :param session_id:
    :param limit: Maximum number of recent messages to return.
    :return:
    """
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

@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    """
    :param session_id:
    :return:
    """
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


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8005)
