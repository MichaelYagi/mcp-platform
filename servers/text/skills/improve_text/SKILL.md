---
name: improve_text
description: >
  Improve, rewrite, expand, shorten, fix, or change the tone of any text using
  a local Ollama model. Use this skill when the user wants to transform existing
  text — not summarize it, not explain it, but rewrite or enhance it in some way.
tags:
  - write
  - ai
  - text
  - rewrite
tools:
  - improve_text_tool
---

# Text Improvement Skill

## Use this skill when the user asks to:

- "Improve this text / rewrite this"
- "Expand on this / make it longer / add more detail"
- "Fix the grammar / fix spelling / proofread this"
- "Shorten this / make it more concise / condense this"
- "Make this more formal / professional"
- "Make this more casual / conversational / friendly"
- "Clean this up / polish this"
- Any request to transform existing text

## Do NOT use this skill for:

- Summarizing text → use `summarize_text_tool`
- Explaining a concept → use `explain_simplified_tool`
- Contextualizing an idea → use `concept_contextualizer_tool`

## Workflow

1. Extract the text to be improved from the user's message
2. Identify the appropriate mode from the user's intent
3. Call `improve_text_tool(text=<text>, mode=<mode>)`
4. Return the `result` field from the response

## Mode selection guide

| User says | Mode |
|-----------|------|
| "expand", "add detail", "make longer", "elaborate" | `expand` |
| "improve", "rewrite", "clean up", "polish", "make better" | `improve` |
| "fix grammar", "fix spelling", "proofread", "correct" | `fix` |
| "shorten", "condense", "make concise", "trim" | `shorten` |
| "make formal", "professional", "business tone" | `formal` |
| "make casual", "conversational", "friendly", "relaxed" | `casual` |
| Anything else specific | `custom` with the instruction |

## Custom mode

When the user's request doesn't fit a standard mode, use `custom` and pass their
instruction directly:

```
improve_text_tool(text="...", mode="custom", instruction="Rewrite this as bullet points")
improve_text_tool(text="...", mode="custom", instruction="Translate this to a pirate voice")
improve_text_tool(text="...", mode="custom", instruction="Make this sound more persuasive")
```

## Notes

- Runs entirely locally via Ollama — no data leaves the machine
- Uses the currently active model (set via `OLLAMA_MODEL` in `.env`)
- Returns only the improved text — no preamble or explanation from the model