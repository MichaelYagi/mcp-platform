---
name: web_summarize
description: >
  Fetch and summarize the content of a web page in a single step. Use this
  skill when the user provides a URL and wants a summary of the page contents.
  Handles fetching and summarization automatically without manual chaining.
tags:
  - web
  - summarize
  - url
  - fetch
tools:
  - summarize_url_tool
---

# Web Summarize Skill

## Use this skill when the user asks to:

- "Summarize this URL: https://..."
- "Summarize the content at https://..."
- "Give me a summary of https://..."
- "What does this page say: https://..."
- "TL;DR this article: https://..."
- "Summarize this link: https://..."

## Workflow

1. Call `summarize_url_tool(url=<url>, style=<style>)` — that's it, one step
2. Style defaults to `"medium"` — use `"short"` or `"detailed"` if the user
   specifies

## Notes

- Requires `OLLAMA_TOKEN` in `.env`
- Content is truncated at 10,000 characters before summarization
- For pages requiring authentication, the tool will return an error
- For a raw page fetch without summarization, use `web_fetch_tool` directly