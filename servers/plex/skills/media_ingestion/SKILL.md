---
name: media_ingestion
description: >
  Ingest Plex media items into the RAG system. Includes granular, batch, and
  parallel ingestion workflows. Use this skill when the user wants to process
  new media, extract subtitles, or run ingestion pipelines.
tags:
  - plex
  - ingestion
  - rag
  - processing
tools:
  - plex_find_unprocessed
  - plex_ingest_items
  - plex_ingest_single
  - plex_ingest_batch
---

# Media Ingestion Skill

Use this skill when the user asks for:

- "Ingest 5 new movies"
- "Process unprocessed items"
- "Ingest these specific IDs"
- "Run the ingestion pipeline"
- "Extract subtitles for this movie"
