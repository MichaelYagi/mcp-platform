---
name: text_splitting_and_summarization
description: >
  Split long text into chunks, summarize individual segments, merge summaries,
  and perform multi-step summarization workflows. Use this skill for processing
  large documents or multi-part text. When the user provides a file path,
  use read_file_tool_handler first to load the content, then summarize.
tags:
  - text
  - summarization
  - chunking
  - processing
  - file
tools:
  - read_file_tool_handler
  - split_text_tool
  - summarize_chunk_tool
  - merge_summaries_tool
  - summarize_text_tool
---

# Text Splitting & Summarization Skill

Use this skill when the user asks for:

- "Summarize this long document"
- "Break this text into chunks"
- "Merge these summaries"
- "Summarize this file"
- "Process this large article"
- "Summarize the file at /path/to/file.txt"

## Workflow for file paths

1. Call `read_file_tool_handler` with the path to load the content
2. If `truncated` is true or content is long, use `split_text_tool` → `summarize_chunk_tool` → `merge_summaries_tool`
3. If content fits in one pass, use `summarize_text_tool` directly