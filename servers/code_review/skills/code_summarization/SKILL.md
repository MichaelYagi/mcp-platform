---
name: code_summarization
description: >
  Summarize or review Python code files and codebases. Use review_code for
  per-file or per-directory quality and security analysis. Use summarize_code
  for a broad architectural overview of the entire project.
tags:
  - code
  - review
  - summarization
  - analysis
tools:
  - review_code
  - summarize_code
---

# Code Summarization Skill

Use this skill when the user asks for:

- "Review this file" → `review_code(path=<file>)`
- "Summarize this file" → `review_code(path=<file>)`
- "Review all files in this directory" → `review_code(path=<dir>)`
- "Give me an overview of the codebase" → `summarize_code()`
- "Explain this project's architecture" → `summarize_code()`
- "Help me understand this code" → `review_code(path=<file>)`

## Tool guidance

- `review_code` — use for any single file or directory; returns metrics,
  issues by severity, and recommendations. Covers both "summarize this file"
  and "review this file" requests.
- `summarize_code` — use only when the user wants a whole-project bird's-eye
  view: architecture notes, entry points, dependency breakdown. Takes no args.