---
name: trilium_search
description: >
  Search Trilium notes using full-text search with support for wildcards, labels,
  and boolean operators. Find notes by content, title, tags, or any combination.
  Use this skill for finding information stored in your Trilium knowledge base.
tags:
  - trilium
  - notes
  - search
tools:
  - search_notes
  - search_by_label
  - get_note_by_id
---

# Trilium Search Skill

Use this skill when the user asks to:

- "Search my notes for X"
- "Find notes about Y"
- "Look up Z in my notes"
- "What notes do I have about..."
- "Show me notes tagged with..."
- "Find notes with #label"
- "Search Trilium for..."

## Search Capabilities

### Full-Text Search
Use `search_notes` for general searches:
- Simple words: "meeting"
- Phrases: "project planning"
- Wildcards: "meet*" (matches meeting, meetings)
- Boolean: "project AND urgent"
- Labels: "#todo", "#priority=high"

### Label-Based Search
Use `search_by_label` for tag-specific queries:
- Find all #todo notes
- Find #priority=high notes
- Find #project=website notes

### Get Specific Note
Use `get_note_by_id` when you have the note ID and need full content.

## Examples

**User:** "Find my notes about machine learning"
**Action:** `search_notes("machine learning", limit=10)`

**User:** "Show me all my todo notes"
**Action:** `search_by_label("todo")`

**User:** "Find high priority project notes"
**Action:** `search_notes("#project AND #priority=high")`

**User:** "Get the full content of note abc123"
**Action:** `get_note_by_id("abc123")`