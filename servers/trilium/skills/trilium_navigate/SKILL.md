---
name: trilium_navigate
description: >
  Browse and navigate the Trilium note hierarchy. View child notes, explore
  note structure, and find recently modified notes. Use this for exploring
  your knowledge base organization.
tags:
  - trilium
  - notes
  - navigation
  - browse
  - recent
tools:
  - get_note_children
  - get_recent_notes
---

# Trilium Navigation Skill

Use this skill when the user wants to:

- "Show me my recent notes"
- "What notes did I modify recently"
- "List the child notes of..."
- "What's under this note"
- "Browse my notes"
- "What are the latest notes"

## Navigation Operations

### Recent Activity
Use `get_recent_notes` to find recently modified notes:
- Default: last 20 notes
- Sorted by modification date (newest first)
- Shows title, type, and last modified date

### Hierarchy Browsing
Use `get_note_children` to explore note structure:
- Get top-level notes: `get_note_children("root")`
- Explore sub-notes: `get_note_children(parent_id)`
- View note organization

## Examples

**User:** "Show me my 10 most recent notes"
**Action:** `get_recent_notes(limit=10)`

**User:** "What notes are at the top level"
**Action:** `get_note_children("root")`

**User:** "Show me what's under my Projects note"
**Action:** 
1. First find Projects: `search_notes("Projects")`
2. Then get children: `get_note_children(note_id)`

**User:** "What have I been working on lately"
**Action:** `get_recent_notes(limit=15)`

## Use Cases

### Finding Recent Work
```
User: "What did I work on this week?"
→ get_recent_notes(limit=30)
→ Filter by date in response
```

### Exploring Organization
```
User: "Show me my note structure"
→ get_note_children("root")
→ For each child, get_note_children(child_id)
```

### Quick Navigation
```
User: "Recent notes about meetings"
→ get_recent_notes(limit=50)
→ Filter results for "meeting" in title
```