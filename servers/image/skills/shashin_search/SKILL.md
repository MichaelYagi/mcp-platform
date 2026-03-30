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
3. Present results as a simple numbered list. For each result include:
   - File name
   - Date taken (`takenAt`)
   - Image ID (`id`)
   - Keywords (if any)
4. Do NOT describe or analyse the JSON structure. Do NOT write summaries about the data format.
5. If the user wants to analyze a specific image, pass its `id` to `shashin_analyze_tool`

## Output Format

Present results exactly like this — nothing more:

```
Found 8 photos matching "Noah":

1. PXL_20251228_021411755.jpg — 2025-12-27
   ID: 0b9bebbf-5468-3cee-b673-29c31244530b
   Keywords: cup, dining table, bowl

2. IMG_5538.jpg — 2025-12-25
   ID: 22ab0912-5913-3487-a0a0-b562d3980db8
   Keywords: backpack
```

If there are more pages, say: "Showing page 1 of N. Ask for more to see the next page."

## Pagination

If the user asks for more results or a specific page, increment the `page`
parameter. `total_pages` in the response indicates how many pages exist.