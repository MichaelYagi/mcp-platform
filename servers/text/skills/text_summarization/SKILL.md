---
name: text_summarization
description: >
  Summarize text of any length from direct input or a file path. Handles short
  and long content automatically — no manual chunking needed. Use this skill
  when the user wants to summarize text, a passage, or a local file.
tags:
  - text
  - summarization
  - file
tools:
  - summarize_text_tool
---

# Text Summarization Skill

Use this skill when the user asks for:

- "Summarize this text: ..."
- "Give me a summary of this passage"
- "Summarize the file at /path/to/file.txt"
- "Short/detailed summary of this article"
- Any request to condense or summarize content already provided in the chat

## Workflow

1. Call `summarize_text_tool(text=<content>, style=<style>)` for inline text
2. Call `summarize_text_tool(file_path=<path>, style=<style>)` for file paths
3. Use `style="short"` for a quick overview, `"medium"` (default) for balanced,
   `"detailed"` for comprehensive output

## Notes

- `summarize_text_tool` handles chunking internally — never split manually
- For URLs, use `summarize_url_tool` instead
- Provide `text` OR `file_path`, never both