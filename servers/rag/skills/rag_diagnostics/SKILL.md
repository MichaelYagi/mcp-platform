---
name: rag_diagnostics
description: >
  Diagnose the RAG database for missing subtitles, incomplete ingestion, or
  problematic entries. Use this skill when the user wants to identify gaps or
  issues in the RAG pipeline.
tags:
  - rag
  - diagnostics
  - ingestion
  - plex
tools:
  - rag_diagnose_tool
---

# RAG Diagnostics Skill

## Overview

This skill diagnoses gaps and problems in the RAG pipeline — specifically for
Plex media ingestion. It identifies items that couldn't be ingested due to
missing subtitles, items that haven't been attempted yet, and overall pipeline
health.

## When to Use

- "What items are missing subtitles?"
- "Which Plex items haven't been ingested?"
- "Diagnose RAG"
- "Show me ingestion problems"
- "Why is X not in the RAG database?"
- "How many items still need processing?"

## Available Tool

### `rag_diagnose_tool`
Scans the Plex library against the RAG database and reports gaps.

**No parameters required.**

**Returns:**
- `total_items` — Total items in the Plex library
- `ingested_count` — Items successfully ingested into RAG
- `missing_subtitles` — Items that exist in Plex but have no subtitle file
  (can't be ingested without subtitles)
- `not_yet_ingested` — Items that haven't been attempted yet
- `statistics` — Overall ingestion health summary

## Interpreting Results

### Missing Subtitles
Items in this list have no subtitle track available. Options:
1. Add subtitle files to Plex for these items and run `rag_rescan_no_subtitles`
2. Accept that these items won't be searchable via RAG
3. These items are **not a bug** — they simply lack the source text needed for embeddings

### Not Yet Ingested
Items that exist in Plex but haven't been processed. These are candidates for
the next ingestion batch:
```
→ plex_ingest_batch limit=10    (ingest the next 10 unprocessed items)
→ plex_find_unprocessed         (see the full list before committing)
```

## Common Workflows

### Workflow 1: Full pipeline audit
```
User: "Give me a full RAG health check"
→ rag_status_tool               (overall document counts)
→ rag_diagnose_tool             (pipeline gaps)
```

### Workflow 2: Why isn't movie X searchable?
```
User: "Why can't I find Inception in RAG searches?"
→ rag_diagnose_tool             (check if it's in missing_subtitles or not_yet_ingested)
→ If not_yet_ingested: plex_ingest_single media_id="..."
→ If missing_subtitles: add subtitles to Plex, then rag_rescan_no_subtitles
```

### Workflow 3: After adding subtitle files
```
User: "I added subtitle files for those items, try again"
→ rag_rescan_no_subtitles       (reset no-subtitle flags)
→ plex_ingest_batch             (process them)
→ rag_diagnose_tool             (verify they're now ingested)
```

## Notes

- Diagnosis requires Plex to be reachable (`PLEX_URL` and `PLEX_TOKEN` in `.env`)
- Results reflect the state at time of call — run again after ingestion to see updated counts
- "Missing subtitles" is the most common reason items don't appear in semantic search