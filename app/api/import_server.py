from __future__ import annotations

import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.clients.es_utils import es_client_manager
from app.clients.milvus_utils import milvus_client_manager
from app.core.logger import logger
from app.import_process.agent.state import get_default_state
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PENDING,
    TASK_STATUS_PROCESSING,
    add_done_task,
    add_running_task,
    clear_task,
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)


ALLOWED_EXTENSIONS = {".md", ".pdf"}
UPLOAD_CHUNK_SIZE = 1024 * 1024
IMPORT_PAGE_PATH = (
    PROJECT_ROOT
    / "lex-legal_frontend"
    / "templates"
    / "import.html"
)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Close shared asynchronous database clients during server shutdown."""
    yield
    await es_client_manager.close()
    await milvus_client_manager.close()


app = FastAPI(
    title="Legal Document Import Service",
    description=(
        "Upload PDF or Markdown legal documents and process them through "
        "the law-aware LangGraph ingestion workflow."
    ),
    lifespan=lifespan,
)

# Development deployments commonly serve the HTML and API from different
# local ports. Set IMPORT_CORS_ORIGINS to a comma-separated allowlist in
# production, for example: https://rag.example.org.
configured_origins = os.getenv("IMPORT_CORS_ORIGINS", "*")
cors_origins = [
    origin.strip()
    for origin in configured_origins.split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", summary="Service health")
async def health() -> dict[str, str]:
    """Return a lightweight health response without contacting dependencies."""
    return {"status": "ok"}


@app.get("/import", response_class=FileResponse)
async def get_import_page() -> FileResponse:
    """Serve the upload page maintained in lex-legal_frontend/templates."""
    if not IMPORT_PAGE_PATH.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Import page does not exist: {IMPORT_PAGE_PATH}",
        )
    return FileResponse(
        path=IMPORT_PAGE_PATH,
        media_type="text/html",
    )


def _safe_upload_name(filename: str | None) -> str:
    """Return a safe supported filename or raise an HTTP 400 response."""
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Every uploaded file must have a filename.",
        )

    safe_name = Path(filename).name
    safe_path = Path(safe_name)
    suffix = safe_path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type for {safe_name!r}. "
                f"Allowed extensions: {supported}."
            ),
        )
    # node_entry routes by lowercase suffix, so normalize only the extension
    # while preserving the user-visible filename stem.
    return f"{safe_path.stem}{suffix}"


async def _save_upload(upload: UploadFile, destination: Path) -> int:
    """Stream one UploadFile to disk and return the number of saved bytes."""
    saved_bytes = 0
    try:
        with destination.open("wb") as output:
            while chunk := await upload.read(UPLOAD_CHUNK_SIZE):
                output.write(chunk)
                saved_bytes += len(chunk)
    finally:
        await upload.close()

    if saved_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"Uploaded file {destination.name!r} is empty.",
        )
    return saved_bytes


@lru_cache(maxsize=1)
def _get_import_graph():
    """
    Load the compiled graph only when the first import task starts.

    Some document-processing dependencies initialize external clients during
    import. Lazy loading keeps the HTTP service and health endpoint available
    even if one of those dependencies is temporarily offline.
    """
    from app.import_process.agent.import_main_graph import kb_import_app

    return kb_import_app


async def run_import_graph(
    task_id: str,
    local_file_path: str,
    local_dir: str,
) -> None:
    """
    Execute the asynchronous LangGraph ingestion workflow in the background.

    Several graph nodes perform asynchronous Elasticsearch, TEI, and Milvus
    operations. Using ``astream`` is therefore required; the synchronous
    ``stream`` API cannot execute those nodes correctly.
    """
    update_task_status(task_id, TASK_STATUS_PROCESSING)
    set_task_result(task_id, "error", "")

    initial_state = get_default_state()
    initial_state.update(
        {
            "task_id": task_id,
            "local_file_path": local_file_path,
            "local_dir": local_dir,
        }
    )

    try:
        import_graph = _get_import_graph()
        async for event in import_graph.astream(
            initial_state,
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                update_keys = (
                    sorted(update)
                    if isinstance(update, dict)
                    else []
                )
                logger.info(
                    f"Import task node completed: "
                    f"task_id={task_id}, node={node_name}, "
                    f"updated_fields={update_keys}"
                )

        update_task_status(task_id, TASK_STATUS_COMPLETED)
        logger.info(f"Import workflow completed. task_id={task_id}")
    except Exception as exc:
        set_task_result(task_id, "error", str(exc))
        update_task_status(task_id, TASK_STATUS_FAILED)
        logger.exception(f"Import workflow failed. task_id={task_id}")


@app.post(
    "/upload",
    summary="Upload legal documents",
    status_code=202,
)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """
    Save uploaded PDF/Markdown files and queue one import task per file.

    Files are validated before anything is written, preventing a mixed request
    from partially starting when one filename has an unsupported extension.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="At least one file is required.",
        )

    safe_names = [
        _safe_upload_name(upload.filename)
        for upload in files
    ]
    date_directory = (
        PROJECT_ROOT
        / "output"
        / datetime.now().strftime("%Y%m%d")
    )

    task_ids: list[str] = []
    tasks: list[dict[str, str]] = []
    for upload, safe_name in zip(files, safe_names, strict=True):
        task_id = str(uuid.uuid4())
        task_directory = date_directory / task_id
        task_directory.mkdir(parents=True, exist_ok=False)
        local_file_path = task_directory / safe_name

        update_task_status(task_id, TASK_STATUS_PENDING)
        add_running_task(task_id, "upload_file")


        try:
            saved_bytes = await _save_upload(upload, local_file_path)
        except Exception as exc:
            set_task_result(task_id, "error", str(exc))
            update_task_status(task_id, TASK_STATUS_FAILED)
            shutil.rmtree(task_directory, ignore_errors=True)
            clear_task(task_id)
            raise

        # the task is over, finished
        add_done_task(task_id, "upload_file")

        # 调用一个方法， call the graph
        """
            This code schedules run_import_graph() to execute after FastAPI sends the upload response:
            It is approximately equivalent to:
            
            await run_import_graph(
            task_id,
            str(local_file_path),
            str(task_directory),
            )
            
            However, the important difference is:
            
            - await run_import_graph(...) makes the user wait for the complete RAG ingestion.
            - background_tasks.add_task(...) returns the HTTP response quickly and processes the document afterward.
            
            for running run_import_graph, need three parameters: local_file_path (文件地址), task_directory(输出地址) and task_id (a session id)
  
        """
        background_tasks.add_task(
            run_import_graph,
            task_id,
            str(local_file_path),
            str(task_directory),
        )

        task_ids.append(task_id)
        tasks.append(
            {
                "task_id": task_id,
                "filename": safe_name,
                "status_url": f"/status/{task_id}",
            }
        )
        logger.info(
            f"Upload saved and import queued: "
            f"task_id={task_id}, filename={safe_name}, "
            f"bytes={saved_bytes}"
        )

    return {
        "code": 202,
        "message": (
            f"Uploaded {len(tasks)} file(s) and queued their import tasks."
        ),
        "task_ids": task_ids,
        "tasks": tasks,
    }


@app.get("/status/{task_id}", summary="Get import task status")
async def get_task_progress(task_id: str) -> dict[str, Any]:
    """Return the global status and completed/running workflow stages."""
    status = get_task_status(task_id)
    if not status:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown import task: {task_id}",
        )

    return {
        "code": 200,
        "task_id": task_id,
        "status": status,
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
        "error": get_task_result(task_id, "error", ""),
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("IMPORT_SERVER_HOST", "127.0.0.1"),
        port=int(os.getenv("IMPORT_SERVER_PORT", "8007")),
    )
