---
name: image_to_image
description: >
  Transform an existing image using a text prompt via Pollinations.ai kontext
  model (free, no API key required). Use this skill when the user wants to
  modify, restyle, or transform an image they already have — not generate one
  from scratch.
tags:
  - ai
  - image
  - transform
  - img2img
  - pollinations
tools:
  - image_to_image_tool
---

# Image-to-Image Skill

## Use this skill when the user asks to:

- "Turn this photo into a watercolor painting"
- "Make this look like an oil painting"
- "Transform this image into anime style"
- "Apply a style to my photo"
- "Edit this image to look like [style/description]"
- "Change the style of this image"
- "Make this look more [adjective]"
- Any request to transform or restyle an existing image

## Do NOT use this skill for:

- Generating a new image from scratch → use `generate_image_tool`
- Finding existing images on the web → use `web_image_search_tool`
- Analyzing or describing an image → use `analyze_image_tool`
- Photos from the Shashin gallery that need searching → use `shashin_search_tool`

## Workflow

1. Identify the source image — it can come from:
   - A URL the user provides
   - A Shashin gallery image (use `shashin_search_tool` first to get the `originalUrl`)
   - A local file path
   - A base64 string already in context (e.g. from `analyze_image_tool`)
2. Extract a clear transformation prompt from the user's message
3. Call `image_to_image_tool(prompt=<prompt>, image_url=<url>)`
4. The `image_base64` in the result will be rendered inline automatically
5. Report the prompt and seed used so the user can reproduce the result

## Prompt writing tips

- Describe the desired output style, not the input: "watercolor painting" not "make it look different"
- Be specific about artistic style: "impressionist oil painting", "cyberpunk neon aesthetic", "vintage film photograph"
- Include mood and lighting if relevant: "dramatic lighting", "soft pastel tones", "high contrast"
- If the user's request is vague, make a reasonable creative choice and mention it

## Getting a source image URL from Shashin

If the user refers to one of their photos (e.g. "transform my photo of the beach"):
1. Call `shashin_search_tool(term="beach")` to find it
2. Use the `originalUrl` field from the result as `image_url`

## Handling errors

- **Rate limit (429)**: Pollinations.ai is temporarily rate limiting — tell the user to try again in a moment
- **Timeout**: The kontext model can take up to 5 minutes — suggest retrying
- **base64 upload failure**: Ask the user to provide a public image URL instead
- No API key required — this service is completely free with no credits or limits