---
name: code_assistant
description: >
  Automated code analysis, bug detection, fixing, and generation using AST analysis.
  Supports Python (deep analysis), JavaScript, TypeScript, Java, Kotlin, Rust, and Go.
  Detects mutable defaults, bare except clauses, identity comparisons, unused imports.
  Generates code from natural language, creates tests, suggests improvements, and refactors.
tags:
  - code
  - python
  - javascript
  - typescript
  - java
  - kotlin
  - debugging
  - fixing
  - analysis
  - generation
  - testing
  - refactoring
tools:
  - analyze_code_file
  - fix_code_file
  - suggest_improvements
  - explain_code
  - generate_tests
  - refactor_code
  - generate_code
  - analyze_project
---

# Code Assistant Skill

## Tool Routing

| User asks... | Tool |
|---|---|
| "analyze / check / find bugs in file" | `analyze_code_file` |
| "fix the bugs in file" | `fix_code_file` (analyze first) |
| "suggest improvements for file" | `suggest_improvements` |
| "explain this code" | `explain_code` |
| "generate tests for file" | `generate_tests` |
| "refactor / modernize file" | `refactor_code` |
| "generate / write code for X" | `generate_code` |
| "tech stack / dependencies / structure" | `analyze_project` |

## Key Workflows

**Analyze → Fix:**
```
analyze_code_file("server.py")   # find issues
fix_code_file("server.py", backup=True)  # fix with backup
```

**Generate code:** Be specific — vague descriptions (<3 words) get flagged.
```
generate_code("Validate email addresses, return True/False, handle missing @",
              language="python", style="function")
```

**Project questions:** Always call `analyze_project()` — never guess the stack.
`analyze_project` covers tech stack, languages, frameworks, dependencies
(`include_dependencies=True`), and directory structure (`include_structure=True`).

## Supported Languages

- **Python** — deep AST analysis (mutable defaults, bare except, identity comparisons)
- **JavaScript / TypeScript** — ESLint integration
- **Java** — Checkstyle / SpotBugs integration
- **Kotlin** — ktlint / detekt integration
- **Rust / Go** — basic analysis

## Safety

- `fix_code_file` creates backups by default (`backup=True`)
- Use `dry_run=True` to preview changes before applying
- Always review AI-generated code before using in production