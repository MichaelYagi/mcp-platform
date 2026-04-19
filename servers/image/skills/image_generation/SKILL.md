---
name: image_generation
description: >
  Generate an image from a text prompt using Pollinations.ai (free, no API key
  required). Use this skill when the user asks to create, generate, or draw an
  image — any request where the image does not already exist and needs to be
  produced from a description.
tags:
  - ai
  - generate
  - image
  - pollinations
tools:
  - generate_image_tool
---

# Image Generation Skill

## Use this skill when the user asks to:

- "Generate an image of [description]"
- "Create a picture of [description]"
- "Draw [description]"
- "Make an image that shows [description]"
- "Can you generate [X]?"
- Any request to produce a new image from a text description

## Do NOT use this skill for:

- Photos already in the Shashin gallery → use `shashin_search_tool`
- Finding existing images on the web → use `web_image_search_tool`
- Analyzing or describing an existing image → use `analyze_image_tool`

## Workflow

1. Extract a clear, descriptive prompt from the user's message
2. Call `generate_image_tool(prompt=<prompt>)`
3. The `image_base64` in the result will be rendered inline automatically
4. Present the prompt used so the user knows what was sent

## Prompt writing tips

- Be specific and descriptive: "a red fox sitting in a snowy forest at dusk" not just "fox"
- Include style cues if the user implies one: "photorealistic", "oil painting", "watercolor", "anime style"
- Include lighting, mood, and composition when relevant: "golden hour lighting", "close-up portrait"
- If the user's request is vague, make a reasonable creative choice and mention it

## Model options (pass via `model` parameter)

| Model | Best for |
|-------|----------|
| `flux` | Default — general purpose, good quality |
| `flux-realism` | Photorealistic images |
| `flux-anime` | Anime / manga style |
| `flux-3d` | 3D rendered look |
| `turbo` | Faster generation, lower quality |

## Handling errors

- **Rate limit (429)**: Pollinations.ai is temporarily rate limiting — tell the user to try again in a moment
- **Timeout**: Generation took too long — suggest retrying
- No API key required — this service is completely free with no credits or limits