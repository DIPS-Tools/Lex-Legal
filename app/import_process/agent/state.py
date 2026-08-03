import copy
from app.core.logger import logger
from typing import Any, NotRequired, Required, TypedDict

class ChunkRecord(TypedDict, total=False):
    """One parent or child legal-document record."""

    chunk_id: Required[str]
    parent_id: Required[str]

    law_id: Required[str]
    law_title: str

    chapter_number: str
    chapter_title: str

    article_number: str
    article_title: str

    paragraph_start: str
    paragraph_end: str

    record_type: Required[str]      # parent | child | parent_child
    content_type: Required[str]     # article | recital | annex | definition

    key_terms: list[str]

    language: str
    # "status": "in_force", meaning: This legal provision is currently legally effective.
    status: str
    source_file: str

    # SHA-256 fingerprints used for import idempotency and change auditing.
    document_hash: str
    chunk_hash: str

    text: Required[str]

    # Added only for child chunks after embedding.
    dense_vector: list[float]

class MilvusRecord(TypedDict, total=False):
    """Child record prepared for insertion into Milvus."""

    chunk_id: Required[str]
    parent_id: Required[str]
    law_id: Required[str]

    chapter_number: str
    article_number: str

    record_type: str
    content_type: str

    key_terms: list[str]
    language: str
    status: str

    document_hash: str
    chunk_hash: str

    dense_vector: Required[list[float]]



class ImportGraphState(TypedDict):

    task_id: str

    is_md_read_enabled: bool
    is_pdf_read_enabled: bool


    # related split document
    is_normal_split_enabled: bool
    is_silicon_flow_api_enabled: bool
    is_advanced_split_enabled: bool
    is_vllm_enabled: bool

    # --- related path ---
    local_dir: str  # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    file_title: str  # 文件标题（文件名去后缀）
    pdf_path: str  # PDF 文件路径 (如果输入是PDF)
    md_path: str  # Markdown 文件路径 (转换后或直接输入的)
    split_path: str  # 分块后的文件路径 【没用】
    embeddings_path: str  # 向量数据库文件路径【没用】

    # --- related content ---
    md_content: str  # Markdown content
    chunks: list  # chunks, including metadata
    key_terms: str  # recognized named entities, enhance the retrieving performance
    document_hash: str

    # ------------------------------------------------------------------
    # Elasticsearch import
    # ------------------------------------------------------------------
    es_import_success: bool
    es_imported_count: int
    es_deleted_stale_count: int
    es_index_name: str
    es_import_error: str

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------
    embedding_success: bool
    embedded_count: int
    embedding_error: str

    # Records prepared for Milvus.
    embeddings_content: list[MilvusRecord]

    # ------------------------------------------------------------------
    # Milvus import
    # ------------------------------------------------------------------
    milvus_import_success: bool
    milvus_imported_count: int
    milvus_deleted_stale_count: int
    milvus_collection_name: str
    milvus_import_error: str


    # ------------------------------------------------------------------
    # General workflow status
    # ------------------------------------------------------------------
    current_node: str
    error_message: str


"""
state["chunks"] = [
    {
        "chunk_id": "gdpr:article_6",
        "parent_id": "gdpr:article_6",
        "law_id": "gdpr",
        "record_type": "parent",
        "content_type": "article",
        "text": "Full Article 6 text...",
    },
    {
        "chunk_id": "gdpr:article_6:paragraph_1_point_f",
        "parent_id": "gdpr:article_6",
        "law_id": "gdpr",
        "record_type": "child",
        "content_type": "article",
        "key_terms": [
            "legitimate interests",
            "controller",
            "data subject",
        ],
        "text": "Processing shall be lawful only if...",
    },
]
"""

# init object
graph_default_state: ImportGraphState = {
    "task_id":"",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,

    "is_normal_split_enabled": True,
    "is_silicon_flow_api_enabled": True,
    "is_advanced_split_enabled": False,
    "is_vllm_enabled": False,

    "local_dir": "",
    "local_file_path": "",

    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "split_path": "",

    "embeddings_path": "",
    "md_content": "",
    "chunks": [],
    "key_terms": "",
    "document_hash": "",

    "embeddings_content": [],
    "es_import_success": False,
    "es_index_name": "",
    "es_imported_count": 0,
    "es_deleted_stale_count": 0,
    "es_import_error":"",

    "embedding_success": False,
    "embedded_count": 0,
    "embedding_error": "",

    "milvus_import_success": False,
    "milvus_collection_name": "",
    "milvus_imported_count": 0,
    "milvus_deleted_stale_count": 0,
    "milvus_import_error": "",

    # General workflow status
    "current_node": "",
    "error_message": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖

    Args:
        **overrides: 要覆盖的字段（关键字参数解包）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(task_id="task_001", local_file_path="doc.pdf")
    """

    # 默认状态
    state = copy.deepcopy(graph_default_state)
    # 用 overrides 覆盖默认值
    state.update(overrides)
    # 返回创建好的状态字典实例
    return state

def get_default_state() -> ImportGraphState:
    """
    返回一个新的状态实例，避免全局变量污染
    """
    return copy.deepcopy(graph_default_state)


if __name__ == "__main__":
    """
    测试
    """
    # 创建默认状态
    state = create_default_state(local_file_path="EU_ACT.pdf")
    logger.info(state)
