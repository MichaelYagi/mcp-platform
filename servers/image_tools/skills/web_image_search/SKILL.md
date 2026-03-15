---
name: web_image_search
description: >
  Search the web for an image of a person, place, or thing using the DuckDuckGo
  Instant Answer API. Use this skill when the user asks to see a picture of
  something that is NOT in their Shashin personal gallery — e.g. a celebrity,
  public figure, landmark, animal, or any real-world entity.
tags:
  - web
  - image
  - search
  - duckduckgo
  - person
  - place
tools:
  - web_image_search_tool
---

# Web Image Search Skill

## Use this skill when the user asks to:

- "Show me a picture of [person/place/thing]"
- "What does [X] look like?"
- "Find me a photo of [X]"
- "Web image of [X]"
- Any request for an image of a real-world entity not in Shashin

## Workflow

1. Extract the subject from the user's message (e.g. "Jorma Tommila", "Eiffel Tower")
2. Call `web_image_search_tool(query=<subject>)`
3. The tool returns `image_url`, `title`, and `abstract` from DuckDuckGo
4. If `image_url` is present, the frontend will render it inline automatically
5. Present the `title` and `abstract` as context alongside the image

## Notes

- Do NOT use this for photos in the user's personal Shashin gallery — use `shashin_search_tool` instead
- If `image_url` is empty, inform the user that no image was found and suggest a more specific search term
- The DuckDuckGo API works best for well-known entities (public figures, landmarks, brands)
- For less well-known subjects, suggest the user try a more specific query