---
name: rag_status
description: >
  Retrieve high-level statistics about the RAG database, including document
  counts, word totals, unique sources, and Plex ingestion progress. Use this
  skill for quick health checks and monitoring.
tags:
  - rag
  - status
  - monitoring
tools:
  - rag_status_tool
---

# RAG Status Skill

## Overview

A quick health check for the RAG database. Shows how much content is stored,
how many sources are indexed, and (if Plex is configured) the current state
of Plex media ingestion progress.

## When to Use

- "Show RAG status"
- "How many documents are in RAG?"
- "How many items have been ingested?"
- "Give me a RAG health summary"
- "Is RAG working?"
- Before or after ingestion to confirm documents were added

## Available Tool

### `rag_status_tool`
Returns a snapshot of the current database state.

**No parameters required.**

**Returns:**

`rag_database` section:
- `total_documents` — Number of text chunks stored in the vector index
- `total_words` — Total word count across all documents
- `unique_sources` — Number of distinct source URLs or labels

`ingestion_tracking` section (requires Plex):
- `total_plex_items` — Total movies/shows in your Plex library
- `successfully_ingested` — Items fully processed and searchable
- `marked_no_subtitles` — Items skipped due to missing subtitle files
- `not_yet_processed` — Items in the queue awaiting ingestion

`summary` — One-line human-readable status

## Interpreting Results

### Healthy state
```json
{
  "rag_database": {
    "total_documents": 1842,
    "total_words": 924000,
    "unique_sources": 24
  },
  "ingestion_tracking": {
    "total_plex_items": 312,
    "successfully_ingested": 288,
    "marked_no_subtitles": 12,
    "not_yet_processed": 12
  },
  "summary": "288 items ingested out of 312 total (92.3% complete)"
}
```

### What each number means

| Field | What to do if unexpected |
|-------|--------------------------|
| `total_documents` is 0 | Nothing ingested yet — run `plex_ingest_batch` or `rag_add_tool` |
| `unique_sources` much lower than expected | Some sources may have been deleted — check with `rag_list_sources_tool` |
| `not_yet_processed` is high | Run `plex_ingest_batch limit=20` to catch up |
| `marked_no_subtitles` is high | Add subtitle files to Plex, then run `rag_rescan_no_subtitles` |
| Plex stats show error | `PLEX_URL` / `PLEX_TOKEN` not set in `.env`, or Plex server is unreachable |

## Common Workflows

### Workflow 1: Quick health check
```
User: "How's the RAG database doing?"
→ rag_status_tool
```

### Workflow 2: Before and after ingestion
```
→ rag_status_tool       (note current document count)
→ plex_ingest_batch limit=10
→ rag_status_tool       (confirm count increased)
```

### Workflow 3: Full audit
```
→ rag_status_tool       (high-level counts)
→ rag_list_sources_tool (what sources are indexed)
→ rag_diagnose_tool     (what's missing or broken)
```

## Notes

- Status reflects the database at time of call — run again after ingestion to see updates
- `total_documents` counts individual text chunks, not whole articles. A single Wikipedia
  page typically generates 10–30 chunks depending on length and `chunk_size`
- Plex ingestion tracking requires `PLEX_URL` and `PLEX_TOKEN` in `.env` — the RAG
  section still works without Plex