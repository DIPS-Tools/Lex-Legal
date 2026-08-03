# Law Document Splitter Notes

## Goal

`node_document_split.py` now targets EU law/regulation Markdown instead of generic documents. The splitter creates structure-aware parent-child records for RAG:

```text
recitals -> recital parent groups + individual recital children
articles -> full Article parent + paragraph/point children
definitions -> full Article parent + grouped definition children
annexes -> annex parent + paragraph children
```

## Main Changes

- Removed fixed Markdown-heading splitting as the main strategy.
- Added law-aware parsing for `Whereas`, `CHAPTER`, `Article N`, definitions, and annex markers.
- Preserves chapter number/title metadata when available.
- Uses the full Article as the parent record.
- Uses numbered paragraphs, definition groups, or lettered point groups as child records.
- Keeps `chunk_id` and `parent_id` stable for Elasticsearch/Milvus sharing.
- Writes generated chunks to a timestamped `chunks_*.json` file in
  `state["local_dir"]` for inspection.

## Node Pipeline

`node_document_split.py` now uses explicit step functions:

```text
step_1_get_content
  -> read/normalize Markdown content and infer file title

step_2_split_law_document
  -> parse legal structure and create initial parent/child records

step_3_refine_chunks
  -> final safety pass: remove empty chunks, validate IDs, validate parent links

step_4_add_content_hashes
  -> add one SHA-256 document hash and a structural hash for every chunk

step_5_backup_chunks
  -> write a timestamped chunks_*.json backup for inspection
```

`step_3_refine_chunks` intentionally does not do generic fixed-size re-splitting. The legal split has already happened in `step_2_split_law_document`; re-splitting at this stage could break a legal sentence, paragraph, or point group.

`step_4_add_content_hashes` keeps the stable legal `chunk_id` unchanged. The
hashes are additional version and auditing metadata:

```text
document_hash
    -> same value for every record from one normalized Markdown version

chunk_hash
    -> identity + structure + authoritative text for one record
```

Line-ending changes and trailing spaces do not change `document_hash`, but a
legal-text change does.

## Record Design

Parent Article example:

```json
{
  "chunk_id": "gdpr:article_6",
  "parent_id": "gdpr:article_6",
  "law_id": "gdpr",
  "record_type": "parent",
  "content_type": "article",
  "article_number": "6",
  "article_title": "Lawfulness of processing",
  "text": "Full Article 6 text..."
}
```

Child Article example:

```json
{
  "chunk_id": "gdpr:article_6:paragraph_1_points_a_c",
  "parent_id": "gdpr:article_6",
  "law_id": "gdpr",
  "record_type": "child",
  "content_type": "article",
  "article_number": "6",
  "paragraph_start": "1(a)",
  "paragraph_end": "1(c)",
  "text": "Article heading plus the matched legal unit..."
}
```

## Current Heuristics

- `law_id` is inferred from file name/path for known laws: GDPR, EU Data Act, Data Governance Act, EU AI Act, and Free Flow of Non-Personal Data.
- `law_title` is inferred from an early Markdown heading containing `REGULATION`; otherwise it falls back to the file title.
- Definitions are detected from an Article title containing `Definitions`.
- Lettered point lists with more than three points are split into groups of up to three points.
- Child chunk size uses token-count heuristics as safety limits, but legal structure is the primary split boundary.

## Validation Performed

The splitter was tested against:

- `output/General_Data_Protection_Regulation_GDPR/General_Data_Protection_Regulation_GDPR.md`
- `output/EU_Data_Act/EU_Data_Act.md`

Observed output:

```text
GDPR: 504 chunks, 118 parents, 386 children
EU Data Act: 291 chunks, 66 parents, 225 children
```

All generated records passed `meta_config.validate_es_record(..., reject_unknown=False)` and had no duplicate `chunk_id` values.

Automated tests:

```text
UV_CACHE_DIR=/tmp/uv-cache uv run python -m unittest app.test.test_node_document_split app.test.test_meta_config app.test.test_node_import_es -v
```

The local manual example from the old splitter style is retained as `test_node_document_split_with_md_img_example()` in `node_document_split.py`. It keeps the flow:

```text
node_md_img -> node_document_split -> inspect final chunks
```

That example is not part of automated tests because `node_md_img` can depend on local MinIO/LLM configuration.

## Known Limits

- It is a deterministic parser, not a full legal XML/ELI parser.
- OCR/PDF Markdown errors can still affect title detection or paragraph grouping.
- Recital grouping is numeric and size-based, not topic-based.
- Long legal sentences are not cut just to hit a target size; this is intentional.
