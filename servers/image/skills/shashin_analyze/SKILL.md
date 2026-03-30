---
name: shashin_analyze
description: >
  Fetch a Shashin gallery image by ID and return its base64-encoded data for
  vision inference. Use this skill when the user wants to analyze, describe,
  or ask questions about a specific photo in their Shashin gallery.
tags:
  - shashin
  - photo
  - vision
  - analyze
  - describe
  - gallery
tools:
  - shashin_search_tool
  - shashin_analyze_tool
---

# Shashin Analyze Skill

## Use this skill when the user asks to:

- "Analyze this photo from my gallery"
- "Describe what's in this Shashin image"
- "What's in the photo of Noah from last week?"
- "Look at image [ID] and tell me what you see"
- Any request to visually inspect a specific Shashin photo

## Workflow

1. If the user has not provided an image ID, call `shashin_search_tool` first
   to find the image — pass a descriptive `term` matching what the user is looking for
2. From the search results, identify the most relevant image and note its `id`
3. Call `shashin_analyze_tool` with that `id`
4. On success, the tool returns `image_base64`
5. Pass the base64 data to Ollama with the user's prompt as the vision instruction
6. If search results included `keywords` for the image, include those as context in your prompt
7. Return Ollama's description to the user

## Thumbnail vs original

- Default (`use_thumbnail: true`) uses the 225px thumbnail — fast, sufficient
  for most descriptions
- Set `use_thumbnail: false` only if the user explicitly requests high detail
  or the thumbnail may be too small for the task

## Vision model requirement

The active Ollama model must support vision inputs. If Ollama returns an error
indicating the model does not support images, relay this message to the user:

> "The current model does not support vision. Please switch to a vision-capable
> model such as **qwen2.5vl:7b** or **llava:7b** and try again."

Do NOT retry with the same model.