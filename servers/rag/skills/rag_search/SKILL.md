---
name: rag_search
description: >
  Search, browse, and maintain the RAG vector database. Use this skill
  for semantic search over stored content, listing and deleting sources,
  and removing individual documents.
tags:
  - rag
  - search
  - semantic
  - browse
  - sources
  - delete
  - maintenance
tools:
  - rag_search_tool
  - rag_list_sources_tool
  - rag_browse_tool
  - rag_delete_source_tool
  - rag_delete_document_tool
---

# RAG Search, Browse & Maintenance Skill

## Overview

The RAG (Retrieval-Augmented Generation) system stores content when you research
topics via URLs. This skill covers searching, browsing, and maintaining that
content — including deleting stale sources and removing individual chunks.

Duplicate prevention is automatic — re-ingesting a URL that's already stored
will skip any chunks that already exist, so no cleanup is needed.

## When Content Gets Stored

Content is **automatically** added to RAG when:
- Researching topics using URLs (e.g., Wikipedia articles)
- Using web search with specific sources
- The system fetches and processes web pages

## Available Tools

### 1. `rag_search_tool` — Semantic Search
Search stored content using natural language queries.

**Parameters:** `query`, `top_k` (default 5), `min_score` (default 0.3)

**Use when:** "search RAG for X", "what do you have about Y", "find content about Z"

### 2. `rag_list_sources_tool` — List All Sources
Show all unique sources with document counts and sample text.

**Use when:** "what sources are in RAG", "list stored content", "show me what URLs are stored"

### 3. `rag_browse_tool` — Browse Recent Documents
Preview recent documents with text snippets and IDs.

**Parameters:** `limit` (default 10, max 50)

**Use when:** "browse the RAG", "show recent documents", "what was stored recently"

### 4. `rag_delete_source_tool` — Delete by Source ⚠️
Delete all documents from a specific source URL or label.

**Parameters:** `source` (required — exact value from `rag_list_sources_tool`)

**Use when:**
- "delete [URL] from RAG"
- "remove that source from RAG"
- "that article is outdated, delete it"
- Re-ingesting updated content (delete first, then re-add)

**Workflow:**
```
1. rag_list_sources_tool      → find exact source string
2. rag_delete_source_tool     → delete it
3. rag_add_tool (optional)    → re-ingest fresh version
```

### 5. `rag_delete_document_tool` — Delete Single Document ⚠️
Delete one specific document chunk by ID.

**Parameters:** `document_id` (required — from `rag_browse_tool` results)

**Use when:** "remove that specific chunk", "delete document [ID]"

**Workflow:**
```
1. rag_browse_tool            → find document ID
2. rag_delete_document_tool   → delete that chunk only
```

## Common Patterns

### Pattern 1: Search stored content
```
User: "What do you have about quantum computing?"
→ rag_search_tool query="quantum computing"
```

### Pattern 2: Update a stale source
```
User: "That Wikipedia article is outdated, refresh it"
→ rag_list_sources_tool       (find exact URL)
→ rag_delete_source_tool      (remove old version)
→ rag_add_tool / fetch URL    (re-ingest fresh)
```

### Pattern 3: Remove a single bad chunk
```
User: "That result had garbled text, remove it"
→ rag_browse_tool             (find the document ID)
→ rag_delete_document_tool    (remove that chunk)
```

## Important Notes

- **Always use `rag_list_sources_tool` before deleting** — source strings must match exactly
- Deleting documents removes them from the vector index but does not delete the raw chunk text from `sessions.db` — this is intentional and safe
- After deleting a source, re-ingest it with `rag_add_tool` or by fetching the URL again
- Duplicate chunks are prevented automatically at ingestion time — re-ingesting the same URL is safe