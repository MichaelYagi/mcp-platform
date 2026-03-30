---
name: ingestion_monitoring
description: >
  Retrieve ingestion statistics and diagnose the Plex RAG database. Use this
  skill when the user wants progress reports, ingestion summaries, or wants
  to find items with missing or incomplete subtitle data.
tags:
  - plex
  - ingestion
  - monitoring
  - stats
tools:
  - plex_get_stats
  - rag_diagnose_plex_tool
---

# Ingestion Monitoring Skill

Use this skill when the user asks for:

- "How many items have been ingested?"
- "Show me ingestion progress"
- "How many items are missing subtitles?"
- "Give me a summary of ingestion status"
- "What's in the RAG database?"
- "Which movies haven't been ingested yet?"

## Tool guidance

- `plex_get_stats` — overall progress: total items, ingested count,
  missing subtitles, remaining unprocessed, completion percentage
- `rag_diagnose_plex_tool` — detailed gap analysis: which specific titles
  are missing subtitle data or haven't been ingested yet