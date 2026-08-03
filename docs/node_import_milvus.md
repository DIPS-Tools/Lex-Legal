# `node_import_milvus` Technical Note

## Purpose

`node_import_milvus` stores validated child dense-vector records in Milvus.

The node is located at:

```text
app/import_process/agent/nodes/node_import_milvus.py
```

Milvus is the semantic retrieval store:

```text
query
    -> query embedding
    -> Milvus COSINE search
    -> matching child chunk IDs
    -> fetch authoritative child/parent text from Elasticsearch
```

Milvus does not replace Elasticsearch. It stores vectors and filtering metadata,
while Elasticsearch stores legal text and provides BM25 retrieval.

## Position in the Import Workflow

```text
node_bge_embedding
    -> state["embeddings_content"]
    -> one vector record per child chunk

node_import_milvus
    -> validates collection compatibility
    -> creates collection/index when absent
    -> idempotently upserts child vectors

END
```

## What Was Implemented

- Added an asynchronous Milvus import node.
- Added a reusable `AsyncMilvusClient` manager.
- Added strict collection schema generation from `configs/meta_config.yml`.
- Added HNSW vector-index generation from YAML.
- Added existing-collection schema validation.
- Added YAML-generated `MilvusRecordInfo`.
- Added optional-field handling for legal units without chapters or Articles.
- Added bounded bulk upserts.
- Used stable string `chunk_id` values as primary keys.
- Added SHA-256 `document_hash` and `chunk_hash` metadata.
- Added automatic migration of the two optional hash fields for compatible
  existing collections.
- Added stale-vector deletion after successful replacement upserts.
- Verified the Milvus confirmed upsert count.
- Added dense-vector search to `MilvusRepository`.
- Added legal metadata filtering and score-threshold support.
- Added task progress and failure-state fields.
- Added isolated repository/node tests and a GDPR integration example.

## Input Contract

The node reads:

```python
state["task_id"]
state["embeddings_content"]
state["embedding_success"]
```

`embeddings_content` must be a non-empty list created by
`node_bge_embedding`.

Example record:

```json
{
  "chunk_id": "gdpr:article_6:paragraph_1_points_d_f",
  "parent_id": "gdpr:article_6",
  "law_id": "gdpr",
  "chapter_number": "II",
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

The actual vector length must match the YAML dimension.

## Output Contract

On success, the node returns:

```python
{
    "milvus_import_success": True,
    "milvus_imported_count": success_count,
    "milvus_deleted_stale_count": deleted_stale_count,
    "milvus_collection_name": repository.collection_name,
    "milvus_import_error": "",
}
```

LangGraph merges this partial update into the state.

On repository failure, it first updates:

```python
state["milvus_import_success"] = False
state["milvus_imported_count"] = 0
state["milvus_collection_name"] = repository.collection_name
state["milvus_import_error"] = str(exc)
```

and then re-raises the exception.

## Execution Steps

### 1. Validate embedding records

```python
if not isinstance(embedding_records, list) or not embedding_records:
    raise ValueError(...)
```

The node refuses to report success when no vectors are available.

It also rejects an explicitly failed embedding stage:

```python
if state.get("embedding_success") is False:
    raise ValueError(...)
```

### 2. Initialize the shared Milvus client

```python
milvus_client_manager.init()
client = milvus_client_manager.get_client()
```

The manager creates one reusable:

```python
AsyncMilvusClient(uri=MILVUS_URL)
```

The application should close it once at shutdown:

```python
await milvus_client_manager.close()
```

The standalone integration test closes it in `finally`.

### 3. Create the repository

```python
repository = MilvusRepository(
    client=milvus_client_manager.get_client(),
    collection_name=milvus_config.chunks_collection,
)
```

The node coordinates graph state. The repository owns Milvus schemas, indexes,
validation, upserts, reads, and searches.

### 4. Ensure collection compatibility

```python
await repository.ensure_collection()
```

If the collection does not exist, the repository creates:

- a strict schema;
- no automatic IDs;
- no dynamic fields;
- the configured vector index.

Equivalent design:

```python
auto_id = False
enable_dynamic_field = False
```

Every field comes from:

```text
configs/meta_config.yml -> milvus_meta
```

After upsert, the repository queries existing primary keys for each `law_id`
and deletes IDs absent from the current child set. Reimporting an unchanged law
therefore replaces the same keys without duplicates, while importing an
updated law also removes obsolete vectors.

### 5. Validate an existing collection

If the collection already exists, the repository verifies:

- actual field names equal the configured field names;
- `chunk_id` is the primary key;
- the primary key is a string `VARCHAR`;
- `dense_vector` has the configured dimension.

An incompatible collection raises:

```text
MilvusSchemaMismatchError
```

This prevents vectors from being inserted into a legacy collection with an
integer primary key, missing fields, or the wrong dimension.

### 6. Build the vector index

The current YAML defines:

```yaml
dense_vector:
  data_type: FLOAT_VECTOR
  metric_type: COSINE
  index_type: HNSW
  index_params:
    M: 16
    efConstruction: 200
```

HNSW is an approximate nearest-neighbor graph index.

Important parameters:

- `M`: maximum graph connectivity; higher values can improve recall but consume
  more memory.
- `efConstruction`: construction search width; higher values usually improve
  index quality but make index creation slower.
- query-time `ef`: search width; higher values usually improve recall but make
  searches slower.

The repository defaults query-time `ef` to:

```python
max(64, limit)
```

### 7. Convert records to exact entities

For every item:

```text
embedding dictionary
    -> MilvusRecordInfo.from_embedding_record
    -> YAML validation/default handling
    -> exact Milvus entity
```

Unknown fields such as `text` or `law_title` are removed during conversion.

`enable_dynamic_field=False` then guarantees that only schema-defined fields
are stored.

### 8. Handle optional legal metadata

Some valid legal chunks do not have:

```text
chapter_number
article_number
```

Examples:

- recitals;
- Annex sections;
- document-level fallback chunks.

New collections mark optional fields nullable.

For compatibility with older collections that still require every field,
`_to_entity()` fills missing optional fields:

```text
optional VARCHAR       -> ""
optional ARRAY_VARCHAR -> []
```

This specifically prevents errors such as:

```text
Insert missed a field `chapter_number`
```

### 9. Upsert bounded batches

```python
success_count = await repository.bulk_upsert(embedding_records)
```

Batch size comes from:

```text
MILVUS_BATCH_SIZE
```

For every batch:

```python
await client.upsert(
    collection_name=...,
    data=entities,
)
```

`chunk_id` is the primary key, so re-importing a chunk replaces its vector and
metadata rather than creating a duplicate.

### 10. Verify confirmed records

The repository reads:

```text
upsert_count
```

or:

```text
insert_count
```

from the response.

If Milvus does not confirm exactly the requested number of entities, the
repository raises:

```text
MilvusUpsertError
```

## YAML-Generated Entity Model

`MilvusRecordInfo` is generated in:

```text
app/entities/milvus_record.py
```

using Pydantic `create_model` and:

```text
meta_config.milvus_meta
```

One YAML definition therefore controls:

```text
Python record fields
runtime validation
defaults
allowed values
Milvus collection schema
vector dimension
index type
index parameters
```

This avoids maintaining a separate manual `MilvusRecord` class whenever the
schema changes.

## Dense Search Support

`MilvusRepository.search()` performs vector retrieval:

```python
results = await repository.search(
    query_vector,
    limit=10,
    filter_expression='law_id == "gdpr" and status == "in_force"',
    score_threshold=0.5,
)
```

Validation includes:

- positive result limit;
- maximum Milvus result limit;
- numeric vector values;
- exact vector dimension;
- finite values only;
- valid requested output fields;
- search metric matching the index metric.

The repository loads the collection and calls:

```python
await client.search(
    data=[vector],
    anns_field="dense_vector",
    filter=filter_expression,
    limit=limit,
    search_params=...,
)
```

Results are flattened to dictionaries containing:

```text
chunk_id
score
parent_id
law_id
other requested metadata
```

The stable `chunk_id` allows Milvus results to be fused with Elasticsearch BM25
results and then used to retrieve authoritative text.

## Why COSINE Similarity

Cosine similarity compares vector direction rather than raw magnitude.

This is suitable for normalized BGE embeddings:

```text
higher cosine score -> more semantically similar
```

The ingestion and query code must use the same embedding model, vector
dimension, normalization behavior, and Milvus metric.

## Configuration

Typical environment:

```env
MILVUS_URL=http://10.22.12.250:19530
CHUNKS_COLLECTION=rag_query_law_chunks
DOCU_NAME_COLLECTION=rag_query_law_ducument_names
KEYWORDS_NAME_COLLECTION=rag_query_law_keywords
MILVUS_TIMEOUT=30
MILVUS_BATCH_SIZE=200
```

Loaded by:

```text
app/config/milvus_config.py
```

The current import node uses:

```text
CHUNKS_COLLECTION
```

The other configured collection names are available for future document-name
and keyword-specific storage but are not written by this node.

Schema and vector-index settings:

```text
configs/meta_config.yml -> milvus_meta
```

## Schema Change Rules

Milvus collection schemas and vector dimensions are not automatically migrated.

When changing fields or embedding dimensions:

1. update `configs/meta_config.yml`;
2. update embedding configuration when necessary;
3. create a new versioned collection or drop/recreate the development
   collection;
4. re-embed all child chunks;
5. re-import all vectors;
6. update retrieval to use the new collection.

Do not insert 1024-dimensional BGE-M3 vectors into an existing 768-dimensional
`bge-base-en-v1.5` collection.

## Failure Behavior

Typical failures:

- missing or empty `embeddings_content`;
- `embedding_success` is explicitly false;
- Milvus is unreachable;
- collection schema differs from YAML;
- vector dimension differs from the collection;
- a required record field is missing;
- string or array limits are exceeded;
- an invalid vector value is present;
- Milvus confirms fewer upserts than requested.

The node logs, records the failure fields, and re-raises the exception so the
outer import task is marked failed.

## Testing

Run isolated tests:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest \
  app.test.test_node_import_milvus
```

The tests verify:

- YAML-derived schema creation;
- string primary key configuration;
- array field creation;
- nullable optional fields;
- vector dimensions;
- old-schema rejection;
- optional-field fallback values;
- upsert confirmation checks;
- lookup by stable chunk IDs;
- dense search;
- legal filter expressions;
- COSINE metric use;
- score thresholds.

Run the manual TEI and Milvus integration example:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run python -m \
  app.import_process.agent.nodes.node_import_milvus
```

The manual test:

1. splits the GDPR Markdown example;
2. embeds all child chunks;
3. prints each vector;
4. imports the records into the configured Milvus collection.
