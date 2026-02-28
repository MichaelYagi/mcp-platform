---
name: text_direct_summarization
description: >
  Summarize short or medium-length text in a single step without chunking.
  Use this skill for quick, direct summarization tasks. When the user provides
  a file path, use read_file_tool_handler first to load the content.
tags:
  - text
  - summarization
  - direct
  - file
tools:
  - read_file_tool_handler
  - summarize_direct_tool
---

# Direct Text Summarization Skill

Use this skill when the user asks for:

- "Give me a quick summary"
- "Summarize this paragraph"
- "Short summary please"
- "Summarize this without splitting"
- "Quick summary of the file at /path/to/file.md"

## Workflow for file paths

1. Call `read_file_tool_handler` to load the file content
2. Pass the returned `content` directly to `summarize_direct_tool`