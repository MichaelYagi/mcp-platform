---
name: text_explanation
description: >
  Explain complex concepts using multiple levels of abstraction or provide
  deep contextual understanding. Use this skill when the user wants clarity,
  analogies, or conceptual background.
tags:
  - text
  - explanation
  - concepts
  - learning
tools:
  - explain_simplified_tool
  - concept_contextualizer_tool
---

# Text Explanation Skill

Use this skill when the user asks for:

- "Explain this concept simply"
- "Give me an analogy for X"
- "Help me understand this idea"
- "Provide context and examples for X"
- "What does X mean in plain English?"

## Workflow

- Use `explain_simplified_tool` for a layered explanation: analogy → plain
  language → technical definition
- Use `concept_contextualizer_tool` for broader background: definition,
  history, related concepts, real-world applications