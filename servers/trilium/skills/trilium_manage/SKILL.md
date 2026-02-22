---
name: trilium_manage
description: >
  Create, update, and delete notes in Trilium. Manage note content, titles,
  and organization. Use this skill when the user wants to modify their
  Trilium knowledge base.
tags:
  - trilium
  - notes
  - create
  - update
  - delete
  - management
tools:
  - create_note
  - update_note_content
  - update_note_title
  - delete_note
  - add_label_to_note
  - get_note_labels
---

# Trilium Note Management Skill

Use this skill when the user wants to:

- "Create a new note"
- "Add a note about X"
- "Update my note"
- "Change the title of..."
- "Delete the note about..."
- "Add a tag to..."
- "Label this note as..."

## Management Operations

### Creating Notes
Use `create_note` to add new notes:
- Specify parent note ID ("root" for top level)
- Set title and content
- Choose note type (text, code, book)

### Updating Notes
- `update_note_content`: Change note content
- `update_note_title`: Rename a note

### Deleting Notes
Use `delete_note` to remove notes (moves to trash)

### Labels/Tags
- `add_label_to_note`: Add tags for organization
- `get_note_labels`: View all tags on a note

## Examples

**User:** "Create a new note about today's meeting"
**Action:** 
```
create_note(
    parent_note_id="root",
    title="Meeting Notes - 2024-02-21",
    content="Attendees:\n- Alice\n- Bob\n\nDiscussion:\n..."
)
```

**User:** "Add a todo tag to note abc123"
**Action:** `add_label_to_note("abc123", "todo")`

**User:** "Update the content of my project note"
**Action:** 
1. First find note: `search_notes("project")`
2. Then update: `update_note_content(note_id, new_content)`

**User:** "Mark this as high priority"
**Action:** `add_label_to_note(note_id, "priority", "high")`