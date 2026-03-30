---
name: web_search
description: >
  Search the web for current information and fetch or summarize page content
  using Ollama's web search API. Use this skill when the user asks about
  current events, breaking news, stock prices, or any information that requires
  up-to-date data beyond the model's training cutoff.
tags:
  - web
  - search
  - news
  - current events
  - stock
  - fetch
tools:
  - web_search_tool
  - summarize_url_tool
---

# Web Search Skill

## Use this skill when the user asks to:

- "What's the latest news on [topic]?"
- "What's happening with [current event]?"
- "What is [company] stock price?"
- "What's the market cap of [X]?"
- "What's going on in the news today?"
- Any question requiring current or real-time information

## Prerequisites

Requires `OLLAMA_TOKEN` set in `.env`. Get a free token at https://ollama.com.

## Workflow

1. Extract the search query from the user's message
2. Call `web_search_tool(query=<query>)` to get a list of results with titles,
   URLs and snippets
3. If a result needs more detail, call `summarize_url_tool(url=<url>)` to fetch
   and summarize the full page in one step
4. Synthesise the results into a clear answer for the user

## Notes

- Use `web_search_tool` first — only call `summarize_url_tool` if the snippet
  is insufficient
- `web_search_tool` returns up to 5 results with inline summaries
- `summarize_url_tool` fetches and summarizes in one call; use `style="short"`
  for quick detail, `"detailed"` for deep dives
- If `OLLAMA_TOKEN` is not set, both tools will return an error — inform the user
- For stock prices, include the ticker symbol in the query (e.g. "AAPL stock price")
- For news, add "today" or the current year to bias toward recent results