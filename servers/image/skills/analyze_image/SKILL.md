---
name: analyze_image
description: >
  Fetch and base64-encode any image from a URL, local file path, or base64
  string for vision inference. Use this skill when the user provides an image
  URL, file path, or base64 string and wants it analyzed, described, or
  interpreted using a vision model.
tags:
  - image
  - vision
  - analyze
  - describe
  - url
  - file
tools:
  - analyze_image_tool
---

# Analyze Image Skill

## Use this skill when the user asks to:

- "Analyze this image: https://..."
- "Describe what's in this photo [URL]"
- "Look at this image file: /home/michael/photo.jpg"
- "Analyze C:\\Users\\Michael\\Pictures\\photo.png"
- "What does this image show?"
- Any request involving an image that is NOT from Shashin

## Input types

| Arg               | When to use                                      |
|-------------------|--------------------------------------------------|
| `image_url`       | Any HTTP(S) URL                                  |
| `image_file_path` | Local file path (Linux, WSL, Windows, or `~/`)   |
| `image_base64`    | Pre-encoded base64 string                        |

Provide exactly one.

## Workflow

1. Call `analyze_image_tool` with the appropriate argument
2. On success, the tool returns `image_base64`
3. Pass the base64 data to Ollama with the user's prompt as the vision instruction
4. Return Ollama's description to the user

## Vision model requirement

The active Ollama model must support vision inputs. If Ollama returns an error
indicating the model does not support images, relay this message to the user:

> "The current model does not support vision. Please switch to a vision-capable
> model such as **qwen2.5vl:7b** or **llava:7b** and try again."

Do NOT retry with the same model.