---
name: shashin_random
description: >
  Fetch a random photo from the Shashin gallery and describe it using vision
  inference. Use this skill when the user asks for a random photo, wants to
  be surprised, or asks Shashin to pick a photo for them.
tags:
  - shashin
  - photo
  - random
  - vision
  - gallery
tools:
  - shashin_random_tool
  - shashin_analyze_tool
---

# Shashin Random Photo Skill

## Use this skill when the user asks to:

- "Show me a random photo"
- "Pick a random picture from my gallery"
- "Surprise me with a photo"
- "Show me a random Shashin image"
- Any request for a random or surprise photo

## Workflow

1. Call `shashin_random_tool()` — no arguments needed
2. It returns the `image_id` of a random photo plus basic metadata
3. Call `shashin_analyze_tool(image_id=...)` with the returned ID
4. The vision shortcut will automatically describe the image
5. Present the description to the user along with any metadata (date, location, keywords)

## Notes

- Do NOT search first — `shashin_random_tool` handles selection entirely
- If `shashin_random_tool` fails, let the user know the gallery may be unavailable