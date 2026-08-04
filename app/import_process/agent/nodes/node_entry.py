import os
import sys

from pathlib import Path
from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_entry(state: ImportGraphState) -> ImportGraphState:
    """
    Entry point for the document-import graph.

    This node receives the graph's external input, validates the source file,
    and initializes the state fields used to route subsequent processing.

    Relevant state fields:
        local_file_path: Path of the document supplied by the caller, i.e., the file is needed to import.
        is_md_read_enabled: Enables the Markdown-reading branch.
        is_pdf_read_enabled: Enables the PDF-reading branch.
        md_path: Source path passed to the Markdown reader.
        pdf_path: Source path passed to the PDF reader.
        file_title: Source filename without its final extension.

    Workflow:
        1. Log the node input and mark the task as running.
        2. Ensure that ``local_file_path`` is present.
        3. Select the Markdown or PDF branch and populate its path according to is_md_read_enabled.
        4. Derive a fallback document title from the source filename.
        5. Log the resulting state and mark the task as complete.

    Future validation may also create a temporary output directory when the
    caller does not supply one.

    """
    # Record entry into this node so the task status can be sent to the UI.
    function_name = sys._getframe().f_code.co_name

    logger.info(f">>> [{function_name}]Execution has begun! Current status: {state}")

    add_running_task(state['task_id'], function_name)

    # Stop processing when no source file was supplied.
    #
    local_file_path = state['local_file_path']
    if not local_file_path:
        logger.error(f"[{function_name}]The check revealed no input file; parsing cannot proceed！！")
        # Return the unchanged state and let the graph determine the next step.
        return state

    # Select the appropriate reader and populate its source-path state.
    if local_file_path.endswith(".md"):
        # Route Markdown documents to the Markdown reader.
        state['is_md_read_enabled'] = True
        state['md_path'] = local_file_path
    elif local_file_path.endswith(".pdf"):
        # Route PDF documents to the PDF reader.
        state['is_pdf_read_enabled'] = True
        state['pdf_path'] = local_file_path
    else:
        logger.error(f"[{function_name}]The file format is not MD or PDF; parsing cannot proceed！！")

    # Derive a fallback title for cases where the model cannot identify the
    # document's item_name. For example, /xx/xxx/aaaa.pdf becomes "aaaa".
    # Path.stem removes only the final suffix, so aa.bb.tar.gz becomes
    # "aa.bb.tar".

    # Legacy os.path-based extraction; retained for compatibility/reference.
    file_title_os = os.path.basename(local_file_path).split(".")[0]

    # Use pathlib to extract the filename without its final suffix.
    file_title = Path(local_file_path).stem
    state['file_title'] = file_title


    # Record node completion and make the updated task status available to the UI.
    logger.info(f">>> [{function_name}]The start has ended! The current status is: {state}")
    add_done_task(state['task_id'], function_name)

    return state
