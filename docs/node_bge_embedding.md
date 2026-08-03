# `node_bge_embedding` Technical Note

## Purpose

`node_bge_embedding` converts searchable legal child chunks into dense vectors
using a Hugging Face Text Embeddings Inference (TEI) service.

The node is located at:

```text
app/import_process/agent/nodes/node_bge_embedding.py
```

Its central rule is:

> Embed child chunks for precise retrieval; keep full parent text in
> Elasticsearch for context expansion.

## Position in the Import Workflow

```text
node_import_es
    -> authoritative parent and child text is already stored

node_bge_embedding
    -> selects child chunks
    -> calls TEI in bounded batches
    -> creates validated Milvus entities

node_import_milvus
    -> stores those entities in the vector collection
```

## What Was Implemented

- Added child-only embedding selection.
- Added support for `child` and `parent_child` record types.
- Added empty-text validation.
- Added a TEI client compatible with LangChain's `Embeddings` interface.
- Added configurable batching.
- Added bounded TEI request concurrency.
- Added retry and backoff for HTTP `429` and `503`.
- Added vector count, dimension, numeric, and finite-value validation.
- Added an explicit embedding/Milvus dimension contract check.
- Added YAML-generated `MilvusRecordInfo` conversion.
- Excluded Elasticsearch-only fields and authoritative text from Milvus.
- Preserved the original chunk dictionaries without adding vectors to them.
- Added task-state fields and detailed logs.
- Added unit tests and a GDPR manual integration test.

## Why Only Children Are Embedded

The splitter produces:

```text
parent
    -> complete Article or larger legal context

child
    -> smallest legally meaningful retrieval unit
```

Embedding all full Articles would make vector retrieval less precise and would
duplicate large amounts of text in Milvus.

The selected records are:

```python
record_type in {"child", "parent_child"}
```

Parent-only records remain searchable through Elasticsearch BM25 and can be
fetched by `parent_id` after a child match.

## Input Contract

The node reads:

```python
state["task_id"]
state["chunks"]
```

`state["chunks"]` must:

- be a non-empty list;
- contain dictionaries;
- contain at least one child or `parent_child`;
- give every selected child non-empty `text`.

Example child input:

```json
{
  "chunk_id": "gdpr:article_6:paragraph_1_points_d_f",
  "parent_id": "gdpr:article_6",
  "law_id": "gdpr",
  "law_title": "General Data Protection Regulation",
  "article_number": "6",
  "record_type": "child",
  "content_type": "article",
  "key_terms": [
    "legitimate interests"
  ],
  "language": "en",
  "status": "in_force",
  "text": "Processing shall be lawful only if..."
}
```

## Output Contract

The node updates and returns the graph state:

```python
state["embeddings_content"] = embedding_records
state["embedding_success"] = True
state["embedded_count"] = len(embedding_records)
state["embedding_error"] = ""
```

Each item in `embeddings_content` has the exact Milvus schema:

```json
{
  "chunk_id": "gdpr:article_6:paragraph_1_points_d_f",
  "parent_id": "gdpr:article_6",
  "law_id": "gdpr",
  "article_number": "6",
  "record_type": "child",
  "content_type": "article",
  "key_terms": [
    "legitimate interests"
  ],
  "language": "en",
  "status": "in_force",
  "dense_vector": [
    0.0124,
    -0.0318
  ]
}
```

The real vector contains the configured number of dimensions.

The Milvus record intentionally excludes:

- `text`;
- `law_title`;
- `chapter_title`;
- `article_title`;
- other Elasticsearch-only fields.

The authoritative text remains in Elasticsearch and is joined through the
shared `chunk_id` or `parent_id`.

## Execution Steps

### 1. Validate the chunk container

```python
if not isinstance(chunks, list) or not chunks:
    raise ValueError(...)
```

This prevents a successful-looking embedding stage with no data.

### 2. Validate the vector dimension contract

```python
_validate_embedding_dimension()
```

The node compares:

```text
embedding_config.dimension
```

with:

```text
meta_config.yml -> milvus_meta -> dense_vector -> dimension
```

They must be identical.

For `bge-base-en-v1.5`:

```text
dimension = 768
```

For BGE-M3 dense embeddings:

```text
dimension = 1024
```

Changing the model without changing both contracts causes an early, readable
error instead of a later Milvus insertion failure.

### 3. Select searchable child chunks

```python
children = _children_for_embedding(chunks)
```

The helper:

- rejects non-dictionary items;
- ignores parent-only records;
- accepts `child` and `parent_child`;
- rejects child records with empty text.

### 4. Initialize and reuse the TEI client

```python
embedding_client_manager.init()
client = embedding_client_manager.get_client()
```

The manager creates one reusable `TEIEmbeddings` instance.

The TEI endpoint is:

```text
http://EMBEDDING_HOST:EMBEDDING_PORT
```

The current deployment serves the model from a Docker container, so the Python
process sends text to TEI rather than loading model weights itself.

### 5. Process bounded batches

```python
for start in range(0, len(children), embedding_config.batch_size):
```

Batching:

- reduces HTTP request overhead;
- respects TEI `MAX_BATCH_TOKENS`;
- limits memory use;
- reduces overload risk.

The text list is created with:

```python
texts = [chunk["text"] for chunk in batch]
```

### 6. Generate vectors asynchronously

```python
vectors = await client.aembed_documents(texts)
```

`TEIEmbeddings` implements LangChain's synchronous `embed_documents`.
LangChain supplies the asynchronous wrapper used by the node, keeping the
LangGraph node awaitable.

TEI is called using Hugging Face `InferenceClient.feature_extraction` with:

```python
normalize=embedding_config.normalize
truncate=embedding_config.truncate
```

### 7. Handle overload safely

The client retries:

```text
HTTP 429 -> too many requests / model overloaded
HTTP 503 -> service temporarily unavailable
```

Backoff behavior:

1. use the `Retry-After` header when available;
2. otherwise use exponential backoff;
3. cap delay at `EMBEDDING_RETRY_MAX_DELAY`;
4. stop after `EMBEDDING_MAX_RETRIES`.

A bounded semaphore limits simultaneous requests:

```text
EMBEDDING_MAX_CONCURRENT_REQUESTS
```

The batch knowledge-base builder also processes documents sequentially to
avoid overwhelming a CPU TEI deployment.

### 8. Validate returned vectors

The TEI adapter validates:

- input texts are non-empty strings;
- returned vector count equals input text count;
- every vector is a list;
- every vector has the configured dimension;
- every value can be converted to `float`;
- no value is `NaN` or infinite.

The node independently confirms:

```python
len(vectors) == len(batch)
```

before pairing records.

### 9. Pair each chunk with its vector

```python
for chunk, vector in zip(batch, vectors, strict=True):
```

Position is the association:

```text
batch[0]   -> vectors[0]
batch[1]   -> vectors[1]
...
```

`strict=True` raises if the two sequences have different lengths.

The new dictionary:

```python
{
    **chunk,
    "dense_vector": vector,
}
```

means:

1. copy all fields from `chunk`;
2. add the corresponding `dense_vector`;
3. do not modify the original chunk.

### 10. Convert to the exact Milvus record

```python
record = MilvusRecordInfo.from_embedding_record(...)
embedding_records.append(record.to_entity())
```

Conversion performs:

```text
chunk plus vector
    -> keep YAML-defined Milvus fields
    -> populate configured defaults
    -> validate types and constraints
    -> remove ES-only fields
    -> output plain dictionary
```

This is why `text` remains in `state["chunks"]` but is absent from
`state["embeddings_content"]`.

## Dense Embedding Technology

Dense embeddings represent semantic meaning as fixed-length floating-point
vectors.

Documents with similar meanings should have vectors with a high cosine
similarity even when their exact words differ.

Example:

```text
"right to erase personal data"
"individual may request deletion of their information"
```

BM25 may favor exact shared words. Dense retrieval can recognize semantic
similarity. The application can later combine Elasticsearch BM25 and Milvus
dense rankings with reciprocal-rank fusion.

The project deliberately does not generate Milvus sparse vectors because
Elasticsearch already supplies lexical BM25 retrieval.

## Normalization and Cosine Similarity

`EMBEDDING_NORMALIZE=true` asks TEI to normalize each vector.

With normalized vectors:

```text
vector length = 1
```

Cosine similarity can then compare vector direction consistently. The Milvus
schema uses:

```yaml
metric_type: COSINE
```

Document and query embeddings must use the same model and compatible
normalization settings.

## Query Prefix

The client can prepend an instruction only for query embeddings:

```env
EMBEDDING_QUERY_PREFIX=Represent this sentence for searching relevant passages:
```

Document ingestion uses `aembed_documents`, so source chunks are embedded
without the query prefix.

Retrieval should use `embed_query`, which applies the prefix.

## Configuration

Typical environment:

```env
EMBEDDING_HOST=10.22.12.250
EMBEDDING_PORT=8083
EMBEDDING_MODEL=bge-base-en-v1.5
EMBEDDING_DIMENSION=768
EMBEDDING_BATCH_SIZE=8
EMBEDDING_NORMALIZE=true
EMBEDDING_TRUNCATE=true
EMBEDDING_TIMEOUT=60
EMBEDDING_MAX_RETRIES=5
EMBEDDING_RETRY_BASE_DELAY=1
EMBEDDING_RETRY_MAX_DELAY=15
EMBEDDING_MAX_CONCURRENT_REQUESTS=2
```

Loaded by:

```text
app/config/embedding_config.py
```

The Milvus vector contract is configured in:

```text
configs/meta_config.yml
```

## Changing the Embedding Model

To switch from `bge-base-en-v1.5` to BGE-M3:

1. change the TEI Docker `MODEL_ID`;
2. change `EMBEDDING_MODEL`;
3. change `EMBEDDING_DIMENSION` from `768` to `1024`;
4. recreate or migrate the Milvus collection because vector dimensions are
   fixed in the collection schema;
5. re-embed all existing child chunks;
6. use the same new model for query embeddings.

Changing only the model name is unsafe when dimensions differ.

## Failure Behavior

On failure:

```python
state["embedding_success"] = False
state["embedding_error"] = str(exc)
```

The node logs the full exception and re-raises it so downstream Milvus import
does not run as if embeddings were valid.

Typical failures:

- no chunks;
- no searchable children;
- empty child text;
- embedding/Milvus dimension mismatch;
- TEI timeout or network failure;
- repeated HTTP 429/503 after the retry limit;
- unexpected vector count;
- incorrect vector dimensions;
- non-numeric, `NaN`, or infinite values;
- a chunk cannot be converted into `MilvusRecordInfo`.

## Testing

Run isolated tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest \
  app.test.test_node_bge_embedding
```

The unit tests use fake clients and verify:

- document and query TEI options;
- query-prefix behavior;
- retry behavior after HTTP 429;
- retry-limit failure;
- child-only embedding;
- Milvus vector dimensions;
- exclusion of parent, text, and Elasticsearch-only fields.

Run the manual TEI integration example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m \
  app.import_process.agent.nodes.node_bge_embedding
```

The manual test prints every generated child vector and requires the configured
TEI service.

