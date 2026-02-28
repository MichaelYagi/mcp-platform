---
name: file_reading
description: >
  Read any local file and return its contents for analysis, summarization,
  or insights. Use this skill when the user provides a file path and wants
  to understand, analyse, or summarize the file contents. Supports CSV, TSV,
  TXT, MD, JSON, YAML, TOML, XML, LOG, PY, JS, TS, INI, CFG, CONF, SH and more.
tags:
  - file
  - reading
  - analysis
  - csv
  - spreadsheet
  - insights
tools:
  - read_file_tool_handler
---

# File Reading Skill

Use this skill when the user asks for:

- "Look at my file at /path/to/file.csv"
- "Open this file and give me insights"
- "Analyse my budget spreadsheet"
- "Read my log file and summarize it"
- "What's in this JSON file?"
- "Give me insights into my expenses CSV"

## Workflow

1. Call `read_file_tool_handler` with the file path from the user's message
2. For short files (under ~2000 words): analyse the content directly from the returned `content` field
3. For longer files: chain with `summarize_direct_tool` or `summarize_text_tool`
4. For CSV/TSV files: use the returned `columns` and `row_count` metadata to frame your analysis before reading the full content

## Path Formats Accepted

- Linux/WSL: `/mnt/c/Users/Michael/Downloads/file.csv`
- Windows: `C:\Users\Michael\Downloads\file.csv` (auto-translated)
- Relative or home paths: `~/Documents/file.txt`