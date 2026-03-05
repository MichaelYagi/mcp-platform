---
name: shashin_search
description: >
  Search the Shashin self-hosted media gallery by keyword term.
  Use this skill when the user wants to find photos or browse their personal gallery.
tags:
  - shashin
  - photo
  - gallery
  - search
  - image
tools:
  - shashin_search_tool
---

# Shashin Search Skill

## Use this skill when the user asks to:

- "Find photos of Noah"
- "Show me pictures from Japan 2023"
- "Search my gallery for cats"
- "Find pictures of sunsets"
- "Browse my photo gallery"

## Parameters

| Parameter | Description                            | Example                          |
|-----------|----------------------------------------|----------------------------------|
| term      | Search keyword (required)              | "cat", "Noah", "beach", "sunset" |
| page      | Page index for pagination (default: 0) | 1                                |

## Workflow

1. Extract the search term from the user's query
2. Call `shashin_search_tool(term=..., page=0)`
3. Present the results — include `fileName`, `takenAt`, `camera`, and `keywords`
4. If the user wants to analyze a specific image, pass its `id` to `shashin_analyze_tool`

## Pagination

If the user asks for more results or a specific page, increment the `page`
parameter. `total_pages` in the response indicates how many pages exist.