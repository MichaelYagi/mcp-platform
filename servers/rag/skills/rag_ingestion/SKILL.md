---
name: rag_ingestion
description: >
  Ingest text or Plex media into the RAG vector database. Covers adding raw
  text, batch and single-item Plex ingestion, resetting items marked as having
  no subtitles, and re-ingesting updated sources. Use this skill when the user
  wants to add new content to RAG or refresh existing content.
tags:
  - rag
  - ingestion
  - embeddings
  - plex
tools:
  - rag_add_tool
  - plex_ingest_batch
  - plex_ingest_items
  - plex_ingest_single
  - plex_find_unprocessed
---

# RAG Ingestion Skill

## Overview

Ingestion is how content gets into RAG. There are two ingestion paths:

1. **Text ingestion** — add any raw text directly (articles, notes, web content)
2. **Plex ingestion** — extract subtitle text from Plex media and embed it

Both paths generate embeddings and store them in the vector database for
semantic search.

## When to Use

- "Add this text to RAG"
- "Ingest the next batch of Plex items"
- "Add this article to the knowledge base"
- "Process the next 5 unprocessed movies"
- "Re-scan items with missing subtitles"
- "Refresh [source] in RAG with the latest version"

## Available Tools

### `rag_add_tool` — Add Raw Text
Add any text directly to RAG with an optional source label.

**Parameters:**
- `text` (required) — The content to add
- `source` (optional) — Label for the source, e.g. a URL or document name (default: "manual")
- `chunk_size` (optional) — Words per chunk (default: 500)

**Use for:**
- Web articles you've fetched
- Notes or documents
- Research you want to store for later

**Example:**
```
use rag_add_tool: text="..." source="https://example.com/article"
```

**Re-ingesting updated content:**
```
1. rag_delete_source_tool source="<url>"   (remove stale version)
2. rag_add_tool text="..." source="<url>"  (add fresh version)
```

### `plex_ingest_batch` — Ingest Next N Items (Simple)
All-in-one tool: finds unprocessed items and ingests them in one call.

**Parameters:**
- `limit` (optional) — Number of items to process (default: 5)
- `rescan_no_subtitles` (optional) — Re-check items previously marked as having no subtitles (default: False)

**Use for:** Simple one-shot ingestion. Handles discovery + processing automatically.

```
use plex_ingest_batch limit=10
```

### `plex_find_unprocessed` — Find Items Awaiting Ingestion
List Plex items that haven't been ingested yet, without processing them.

**Parameters:**
- `limit` (optional) — How many to list (default: 5)

**Use for:** Previewing what's in the queue before committing to a batch.

```
use plex_find_unprocessed limit=20
```

### `plex_ingest_items` — Ingest Specific Items by ID
Ingest a specific list of Plex items by their IDs (parallel processing).

**Parameters:**
- `item_ids` (required) — Comma-separated Plex media IDs, or `"auto:N"` to auto-find N items

**Use for:** Targeted ingestion when you know which items you want.

```
use plex_ingest_items item_ids="12345,12346,12347"
use plex_ingest_items item_ids="auto:5"
```

### `plex_ingest_single` — Ingest One Item
Ingest a single Plex item by ID.

**Parameters:**
- `media_id` (required) — Plex media ID, or `"auto"` to pick the next unprocessed item

```
use plex_ingest_single media_id="12345"
use plex_ingest_single media_id="auto"
```

## Common Workflows

### Workflow 1: Regular batch ingestion
```
User: "Ingest the next 10 Plex items"
→ plex_ingest_batch limit=10
```

### Workflow 2: Check queue, then ingest
```
User: "What's waiting to be ingested?"
→ plex_find_unprocessed limit=20

User: "Process those 5 movies"
→ plex_ingest_items item_ids="id1,id2,id3,id4,id5"
```

### Workflow 3: Add a web article
```
User: "Add this Wikipedia article to RAG: https://en.wikipedia.org/wiki/..."
→ [fetch the page content]
→ rag_add_tool text="<content>" source="https://en.wikipedia.org/wiki/..."
```

### Workflow 4: Refresh a stale source
```
User: "That article is outdated, re-ingest it with the latest version"
→ rag_delete_source_tool source="<url>"         (remove old)
→ [fetch updated content]
→ rag_add_tool text="<content>" source="<url>"  (add new)
```

### Workflow 5: Re-scan items that had no subtitles
```
User: "I added subtitle files — try those items again"
→ rag_rescan_no_subtitles                        (reset no-subtitle flags)
→ plex_ingest_batch rescan_no_subtitles=True     (process them)
```

## Performance Notes

- Each Plex item takes ~2–5 seconds depending on subtitle length and GPU availability
- `plex_ingest_items` processes items in parallel (up to 3 concurrent)
- `plex_ingest_batch` is single-threaded but simpler — use for small batches
- Stop a long ingestion at any time with `:stop`

## After Ingestion

Verify with:
```
→ rag_status_tool               (check new document counts)
→ rag_search_tool query="..."   (confirm content is searchable)
```