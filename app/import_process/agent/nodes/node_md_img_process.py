"""
Node summary: Markdown image processing pipeline.

This node post-processes a Markdown file that was usually generated from a PDF.
Its job is to find local images referenced by the Markdown, generate a useful
description for each image with a vision-capable LLM, upload those images to
MinIO object storage, and rewrite the Markdown image links so they point to the
uploaded MinIO URLs with the generated description as the alt text.

Step-by-step workflow:
1. Read and validate input state.
   - The node expects state["md_path"] to point to an existing Markdown file.
   - If state["md_content"] is empty, it reads the Markdown content from disk.
   - It assumes local images are stored in an "images" folder beside the Markdown file.

2. Scan the local images folder.
   - Only files with supported image extensions are considered.
   - For each image file, the node checks whether the Markdown content references it.
   - Images that are not referenced in the Markdown are skipped.

3. Extract context around each referenced image.
   - The Markdown image syntax is located with a regular expression.
   - The node captures text before and after the first occurrence of each image.
   - This surrounding context helps the vision model generate a more relevant image summary.

4. Generate image summaries with the vision model.
   - The node rate-limits API calls before each model request.
   - Each local image is read, base64-encoded, and sent to the configured vision model.
   - The prompt includes the Markdown root folder name and the extracted image context.
   - The result is stored as a mapping from image file name to generated summary.

5. Upload images to MinIO.
   - Before uploading, the node deletes existing MinIO objects for the same Markdown stem.
   - Each referenced image is uploaded to the configured MinIO bucket and image directory.
   - A public image URL is built from the MinIO endpoint, bucket name, image directory, Markdown stem, and image file name.

6. Rewrite Markdown image links.
   - Each original local Markdown image reference is replaced with:
     ![generated image summary](uploaded MinIO image URL)
   - Only images that have both a generated summary and an uploaded URL are replaced.

7. Save and update state.
   - The rewritten Markdown content is saved next to the original file as "<original_name>_new.md".
   - state["md_path"] is updated to the new Markdown file path.
   - state["md_content"] is updated to the rewritten Markdown content.
   - The updated state is returned to the import workflow.

Important assumptions:
- The image directory name is fixed as "images" and must be next to the Markdown file.
- The replacement logic matches image references by file name, not by fully normalized paths.
- The code currently treats uploaded images as "image/jpeg" content type for MinIO, regardless of file extension.
- If the images directory does not exist, the node returns the original state without creating a new Markdown file.
"""

import os
import re
import sys
import base64
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque

# MinIO dependencies.
from minio import Minio
from minio.deleteobjects import DeleteObject

# Core refactor: use the shared LangChain utility instead of native OpenAI calls,
# and import the multimodal message utilities required by the vision model.
from app.clients.minio_client import get_minio_client
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
# Shared LLM client utility. This centralizes model creation and replaces direct OpenAI client usage.
from app.llm.llm_utils import get_llm_client
# LangChain multimodal dependencies for message construction and exception handling.
from langchain.messages import HumanMessage
from langchain_core.exceptions import LangChainException
# Project configuration.
from app.config.minio_config import minio_config
from app.config.llm_config import llm_config
# Project logger. All node logs should go through this logger for consistent output.
from app.core.logger import logger
# API rate limiting utility.
from app.utils.rate_limit_utils import apply_api_rate_limit
# Prompt loading utility.
from app.prompt.load_prompt import load_prompt

# Image formats supported by this MinIO upload pipeline.
# Extensions are lowercase so file matching is case-insensitive after normalization.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

def is_supported_image(filename: str) -> bool:
    """
    Check whether a file uses one of the supported image extensions.

    The extension comparison is case-insensitive because the file extension is
    normalized to lowercase before matching.

    :param filename: File name including extension.
    :return: True if the file is a supported image type, otherwise False.
    """
    return os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS




"""

1. Upload each local image to the MinIO image server and get a network URL.
2. Send each image to the vision model to generate a description.
3. Replace the original Markdown image reference with the generated description
   and the MinIO URL: [image summary](network image URL).

  Main goal:
        Process Markdown images separately so downstream models can understand
        the meaning of each image from text.
  Main transformation:
        image -> file server -> network image URL
        + 100 characters of previous context
        + image
        + 100 characters of following context
        -> vision model -> image summary
        -> [image summary](network image URL)
        -> update state["md_content"] and state["md_path"].
  Main technologies:
        MinIO stores image files.
        The vision model uses the prompt and image context to generate summaries.
  Detailed steps:
     1. Validate and collect input data for this operation.
        Input: state with md_path and md_content.
        Output: validated Markdown content, Markdown Path object, and images folder Path object.
     2. Identify images referenced by the Markdown.
        Only referenced images are processed; unused image files are ignored.
        Input: Markdown content and images directory.
        Output: [(image_name, image_path, (previous_context, following_context))].
     3. Generate image summaries with the vision model.
        Input: image target list and Markdown file stem, which is also used as the image folder name in prompts.
        Output: {image_name: summary, ...}.
     4. Upload images to MinIO and update the Markdown content.
        Input: MinIO client, summaries, image target list, original Markdown content, and Markdown file stem.
        Output: new Markdown content.
        state[md_content] = new_md_content
     5. Finalize and save the processed Markdown file.
        Input: new Markdown content and original Markdown path, e.g. xx.md -> xx_new.md.
        Output: new Markdown path.
        state[md_path] = new_new_md_path
    return state
"""


def step_1_get_content(state: ImportGraphState) -> Tuple[str, Path, Path]:
    """
    Read and validate Markdown content from the import state.

    This step guarantees that the downstream image-processing steps have:
    - Markdown text to inspect.
    - A Path object for the source Markdown file.
    - The expected local image directory beside the Markdown file.

    :param state: Import graph state containing md_path and optionally md_content.
    :return: Tuple of Markdown content, Markdown Path object, and images directory Path object.
    """
    # 1. Read the Markdown file path from state["md_path"].
    md_file_path = state["md_path"]
    if not md_file_path:
        raise ValueError("md_path cannot be empty.")

    md_path_obj = Path(md_file_path)
    if not md_path_obj.exists():
        raise FileNotFoundError(f"md_path:{md_file_path} does not exist.")

    # Ensure md_content is available.
    if not state['md_content']:
        # If md_content is missing, read it from disk.
        # If it already exists, it was likely populated by the previous PDF parsing node.
        with md_path_obj.open("r", encoding="utf-8") as f:
            state['md_content'] = f.read()
    # Build the expected local images directory path.
    # Convention: custom Markdown imports must place image files in a sibling folder named "images".
    images_dir_obj = md_path_obj.parent / "images"
    return state['md_content'], md_path_obj, images_dir_obj


def find_image_in_md_content(md_content, image_file, context_length: int = 100):
    """
    Find a Markdown image reference and extract nearby text context.

    The default context window is 100 characters before and after the matched
    Markdown image reference. If an image appears multiple times, this function
    intentionally uses only the first occurrence.

    :param md_content: Full Markdown content to search.
    :param image_file: Image file name to find inside Markdown image links.
    :param context_length: Number of characters to capture before and after the image reference.
    :return: Tuple of (previous_context, following_context), or None when the image is not referenced.
    """

    """
    Example Markdown content:

    Some text before the image.
    More surrounding context.
    [start] ![example image](/xxx/xx/zhaoweifeng.jpgxxx) [end]
    Text after the image.
    Another reference:
    ![example image](/xxx/xx/zhaoweifeng.jpgxxx)
    More trailing text.

    file_name zhaoweifeng.jpg
    """
    # Build a regex that matches Markdown image syntax containing this image file name.
    # The non-greedy wildcard prevents one image link from consuming text across multiple links.
    pattern = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")

    content = None  # Stores the selected context. If the image is used multiple times, use the first match.
    # Find all Markdown image references that contain this image file name.
    items = list(pattern.finditer(md_content))
    if not items:
        return None

    # item = items[0]
    # start, end = item.span()

    if item := items[0]:
        start, end = item.span()  # span returns the start and end indexes of the matched image reference.
        # Capture text before the image reference.
        pre_text = md_content[max(start - context_length, 0):start]  # Clamp to 0 when fewer previous characters exist.
        post_text = md_content[end:min(end + context_length, len(md_content))]  # Clamp to content length at the end.
        # Capture text after the image reference.
        content = (pre_text, post_text)
    # Return the text surrounding the matched image reference.
    if content:
        logger.info(f"Image {image_file} found in content preview {md_content[:100]}; first context extracted: {content}")
        return content


def step_2_scan_images(md_content: str, images_dir_obj: Path) -> List[Tuple[str, str, Tuple[str, str]]]:
    """
    Scan the images directory and collect only images referenced by the Markdown.

    For each supported image file in the local images directory, this function
    checks whether the Markdown content contains a reference to that file. When
    a reference is found, the function also captures nearby text context for use
    in image summarization.

    :param md_content: Markdown content to inspect.
    :param images_dir_obj: Directory containing local image files.
    :return: List of (image_name, image_path, (previous_context, following_context)).
    """
    # 1. Create the target list for images that need processing.
    targets = []
    # 2. Iterate through the images directory, keep supported referenced images, and capture context.
    for image_file in os.listdir(images_dir_obj):
        # Inspect each file name.
        # Skip files that are not supported image types.
        if not is_supported_image(image_file):
            logger.warning(f"Current file {image_file} is not a supported image format; skipping.")
            continue
        # For supported images, check whether the Markdown references the file.
        # If referenced, capture (previous_context, following_context).
        content_data = find_image_in_md_content(md_content, image_file)
        if not content_data:
            logger.warning(f"Image {image_file} is not referenced in the Markdown content; context is empty.")
            continue

        # Output: [(image_name, image_path, (previous_context, following_context))].
        targets.append((image_file, str(images_dir_obj / image_file), content_data))

    return targets


def step_3_generate_img_summaries(targets, stem):
    """
    Generate a text summary for each image with the configured vision model.

    Each image is encoded as base64 and sent together with a prompt that includes
    nearby Markdown context. The summary is later used as the Markdown image alt
    text after the image is uploaded to MinIO.

    :param targets: List of (image_name, image_path, (previous_context, following_context)).
    :param stem: Markdown file stem, also used as the image folder name in prompts.
    :return: Mapping of image_name to generated summary text.
    """
    summaries = {}  # Final mapping of image file name to generated summary.
    # Request one summary per image from the vision model.
    # request_times tracks recent requests so rate limiting can be applied consistently.
    request_times = deque()
    for image_file, image_path, context in targets:
        # Unpack image name, local image path, and surrounding Markdown context.
        # 1. Apply API rate limiting before calling the model.
        # The configured limit prevents too many requests in a short time window.

        logger.info(f"Processing image {image_file}, its path: {image_path} and context : {context}")

        """
        for example:
        Processing image cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg, 
        its path: /local/scratch/mh1f25/python_projects/guigu_ai/guigu_ai_projects/dataset_rag/output/hak180_Product_Safety_Manual/images/cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg 
        and context : ('利的边缘可能导致受伤。\n\n## 为设备选择一个安全的位置\n\n•\t提起本设备时，请使用双手抓稳本设备的两侧。如果抓住的是进纸托板和出纸盒，它们可能会掉下来。必须通过将双手放在本设备下面来搬运本设备。\n\n', '  \n确保本设备的任何部位均未伸出设备所在的桌面或支架。特别是当本设备位于桌面、支架等边缘时，请勿让出纸盒打开。确保本设备位于平整、水平且稳定的表面上，避免震动。不遵守这些预防措施可能导致设备跌落，从')
        
        """

        apply_api_rate_limit(request_times, max_requests=500)
        # 2. Send the image and prompt to the vision model.
        # 2.1 Get the configured vision model client.
        vm_model = get_llm_client(model=llm_config.lv_model)
        # 2.2 Load and format the image-summary prompt.
        prompt = load_prompt("image_summary", root_folder=stem, image_content=context)

        # import base64
        # get the image fom given path
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")  # Convert image bytes to a base64 string.
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            # A direct image URL can also be used here, for example:
                            # "url": "https://example.com/path/to/image.jpeg"
                            # This implementation sends the local image as a base64 data URL.
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                    {"type": "text", "text": f"{prompt}"},
                ],
            },
        ]
        # 2.3 Invoke the model and store the generated summary.
        response = vm_model.invoke(messages)
        summary = response.content.strip().replace("\n", "")
        summaries[image_file] = summary
        logger.info(f"Image {image_file} summary result: {summary}")
        """
        - Image cc5ee1ac24ebb2707d40dc7a234a8b243f55f5bf08fabc683859be6fdf096ffa.jpg summary result: 搬运设备时双手扶稳，避免边缘受伤与托盘脱落
        """
    logger.info(f"Image summary results: {summaries}")
    return summaries


def step_4_upload_images_and_replace_md(summaries, targets, md_content, stem):
    """
    Upload referenced images to MinIO and replace Markdown image references.

    For each image that has both a generated summary and an uploaded URL, the
    original Markdown image link is replaced with a remote MinIO URL and the
    generated summary as alt text.

    :param summaries: Mapping of image_name to generated description.
    :param targets: List of (image_name, original_image_path, (previous_context, following_context)).
    :param md_content: Original Markdown content.
    :param stem: Markdown file name without extension.
    :return: Rewritten Markdown content.
    """
    # MinIO object layout: bucket / upload-images / <stem> / image_file.jpg.
    minio_client = get_minio_client()
    # 1. Delete existing MinIO images for this Markdown file stem.
    # 1.1 List objects that should be removed.
    # Object object_name
    # Note: minio_config.minio_img_dir[1:] intentionally removes the leading slash.
    object_list = minio_client.list_objects(minio_config.bucket_name,
                                            prefix=f"{minio_config.minio_img_dir[1:]}/{stem}",
                                            recursive=True)
    # Convert every listed object into a DeleteObject request.
    delete_object_list = [DeleteObject(obj.object_name) for obj in object_list]
    # remove_objects expects DeleteObject instances.
    # 1.2 Delete all matching objects.
    errors = minio_client.remove_objects(minio_config.bucket_name, delete_object_list)
    for errors in errors:
        logger.error(f"Failed to delete object: {errors}")

    logger.info(f"Finished clearing objects under {stem}; deleted {len(delete_object_list)} objects.")

    # 2. Upload images to MinIO.
    # Store the resulting public URL for each uploaded image.
    images_url = {}
    # targets contains (image_name, original_image_path, (previous_context, following_context)).
    for image_file, image_path, _ in targets:
        try:
            minio_client.fput_object(
                bucket_name=minio_config.bucket_name,
                # given the saved folder in minio
                object_name=f"{minio_config.minio_img_dir}/{stem}/{image_file}",
                # Object name inside the MinIO bucket, for example xx.png or xx/xxx/xx.png.
                file_path=image_path,
                content_type="image/jpeg"
            )
            # Record the image URL after upload.
            # URL format: protocol + endpoint + bucket name + object name.
            # Example: http://47.94.86.115:9000/<bucket>/<object_name>.
            images_url[
                image_file] = f"http://{minio_config.endpoint}/{minio_config.bucket_name}{minio_config.minio_img_dir}/{stem}/{image_file}"
            logger.info(f"Finished uploading image {image_file}. URL: {images_url[image_file]}")
        except Exception as e:
            logger.error(f"Failed to upload image {image_file}. Reason: {e}")
    # 3. Replace image references in the Markdown content.
    # summaries = image_name: description
    # images_url = image_name: URL
    # image_infos = image_name: (description, URL)
    image_infos = {}
    for image_file, summary in summaries.items():
        if url := images_url.get(image_file):
            image_infos[image_file] = (summary, url)
    logger.info(f"Image processing summary: {image_infos}")

    """
    for example:
    
    image_infos = {
    "a.png": ("A chart showing sales", "http://minio/bucket/images/doc/a.png")
    }
    """

    if image_infos:
        """
        xxxx
        xxx  ![old alt text](image_path/image_file) -> ![summary](MinIO URL)
        xxx
        """
        for image_file, (summary, url) in image_infos.items():
            # Use a regex to replace any Markdown image link that contains this image file name.
            # Example: ![](/xxx/xx/image_file) -> ![summary](MinIO URL).
            # rep = re.compile(r"!\[.*?\]\(.*?" + image_file + ".*?\)")
            rep = re.compile(r"!\[.*?\]\(.*?" + re.escape(image_file) + r".*?\)")
            md_content = rep.sub(f"![{summary}]({url})", md_content)    # creates a new string and assigns it back to md_content.
        # logger.info(f"Finished replacing Markdown image references. New content: {md_content}")
        logger.info(f"Finished replacing Markdown image references.")
    return md_content


def step_5_replace_md_and_save(new_md_content, md_path_obj):
    """
    Save the rewritten Markdown content to a new file and return its path.

    The new file name is built by appending "_new" before the .md extension.
    Example: xxx.md -> xxx_new.md.

    :param new_md_content: Rewritten Markdown content.
    :param md_path_obj: Original Markdown path.
    :return: New Markdown path.
    """
    # Build the new Markdown file path.
    #   c:/xxx/xxx/xxx/xxxx/erdaye.md -> splitext(md_path_obj)[0]
    #   -> c:/xxx/xxx/xxx/xxxx/erdaye_new.md
    new_md_path_str = os.path.splitext(md_path_obj)[0] + "_new.md"

    with open(new_md_path_str, "w", encoding="utf-8") as f:
        f.write(new_md_content)
    logger.info(f"Finished writing rewritten Markdown content. New path: {new_md_path_str}")
    return new_md_path_str


def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """
    Node: Markdown image processing (node_md_img).

    This node processes image resources referenced in Markdown content.
    It scans Markdown image links, uploads the referenced images to MinIO,
    generates image descriptions with a multimodal model, replaces local image
    links with MinIO URLs, and stores the rewritten Markdown back into state.
    """
    function_name = sys._getframe().f_code.co_name
    logger.info(f">>> [{function_name}] Started. Current state: {state}")

    add_running_task(state['task_id'], function_name)
    # 1. Validate and collect the data required for this operation.
    #    Input: state with md_path and md_content.
    #    Output: validated Markdown content, Markdown Path object, and images directory Path object.
    md_content, md_path_obj, images_dir_obj = step_1_get_content(state)
    # If the expected local image directory does not exist, return the original state unchanged.
    if not images_dir_obj.exists():
        logger.info(f">>> [{function_name}] No images directory found; returning state unchanged.")
        return state
    # 2. Identify images used by the Markdown and prepare them for summarization.
    # Result format: [(image_name, image_path, (previous_context, following_context = 100 characters))].
    targets = step_2_scan_images(md_content, images_dir_obj)
    # Input: Markdown content and local images directory.
    # Output: [(image_name, image_path, (previous_context, following_context))].

    # 3. Generate image summaries with the vision model.
    # Input: image target list and Markdown file stem.
    # Output: {image_name: summary, ...}.
    summaries = step_3_generate_img_summaries(targets, md_path_obj.stem)
    # 4. Upload images to MinIO and replace Markdown image links with summary + URL.
    # Input: summaries, image target list, original Markdown content, and Markdown file stem.
    # Output: rewritten Markdown content.
    # state["md_content"] will be updated with this rewritten content.
    new_md_content = step_4_upload_images_and_replace_md(summaries, targets, md_content, md_path_obj.stem)

    # 5. Save the rewritten Markdown content to a new file.
    # Input: new Markdown content and original Markdown path, e.g. xx.md -> xx_new.md.
    # Output: new Markdown path.
    # state["md_path"] will be updated with the new path.
    new_md_file_path = step_5_replace_md_and_save(new_md_content, md_path_obj)
    # Update state with the new Markdown path and rewritten content.
    state["md_path"] = new_md_file_path
    state["md_content"] = new_md_content
    logger.info(f">>> [{function_name}] Finished. Current state: {state}")
    add_done_task(state['task_id'], function_name)
    return state


if __name__ == "__main__":
    """Local test entry point: run the full Markdown image-processing workflow directly."""
    from app.utils.path_util import PROJECT_ROOT

    logger.info(f"Local test - project root: {PROJECT_ROOT}")

    # Test Markdown file path. The test file must be placed in the expected directory manually.
    test_md_name = os.path.join(r"output/hak180_Product_Safety_Manual", "hak180_Product_Safety_Manual.md")
    test_md_path = os.path.join(PROJECT_ROOT, test_md_name)

    # Validate that the local test file exists.
    if not os.path.exists(test_md_path):
        logger.error(f"Local test - test file does not exist: {test_md_path}")
        logger.info("Check the file path, or manually place the test Markdown file under the project's output directory.")
    else:
        # Build a test state object to simulate workflow input.
        test_state = {
            "md_path": test_md_path,
            "task_id": "test_task_123456",
            "md_content": ""
        }
        logger.info("Starting local test - full Markdown image-processing workflow.")
        # Run the core processing workflow.
        result_state = node_md_img(test_state)
        logger.info(f"Local test completed - result state: {result_state}")
