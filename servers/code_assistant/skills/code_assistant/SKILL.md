---
name: code_assistant
description: >
  Automated code analysis, bug detection, fixing, and generation using AST analysis.
  Supports Python (deep analysis), JavaScript, TypeScript, Java, Kotlin, Rust, and Go.
  Detects mutable defaults, bare except clauses, identity comparisons, unused imports.
  Generates code from natural language, creates tests, suggests improvements, and refactors.
  Extends existing files following their conventions, detects cross-file inconsistencies,
  explains multi-file architecture and data flow, and assesses change impact across the codebase.
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
  - architecture
  - impact-analysis
  - consistency
tools:
  - analyze_code_file
  - fix_code_file
  - suggest_improvements
  - explain_code
  - generate_tests
  - refactor_code
  - generate_code
  - analyze_project
  - extend_code
  - detect_inconsistencies
  - explain_architecture
  - explain_change_impact
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
| "add to / extend this file following its patterns" | `extend_code` |
| "find inconsistencies / convention drift across files" | `detect_inconsistencies` |
| "how do these files work together / explain the architecture" | `explain_architecture` |
| "what will break if I change X / impact analysis" | `explain_change_impact` |

## Key Workflows

**Analyze → Fix:**
```
analyze_code_file("server.py")               # find issues
fix_code_file("server.py")                   # preview fixes (dry_run=True by default)
fix_code_file("server.py", dry_run=False)    # apply fixes to disk
```

**Extend a file following its existing patterns:**
```
extend_code("servers/code_assistant/server.py",
            "a tool called search_in_code that greps for a pattern across files")
# shows preview by default — pass write=True to apply
extend_code("servers/weather/server.py",
            "a get_forecast_weekly tool following existing @tool_meta pattern",
            write=True)
```

**Audit codebase for convention drift:**
```
detect_inconsistencies("servers/")                          # all categories
detect_inconsistencies("servers/", category="error_handling")
```

**Understand how a subsystem works:**
```
explain_architecture("client/client.py,client/websocket.py,client/capability_registry.py")
explain_architecture("servers/", focus="data_flow")
```

**Before making a risky change:**
```
explain_change_impact("client/client.py",
                      description="remove the text field shortcut in list builder")
explain_change_impact("servers/code_assistant/server.py",
                      change="dry_run default changed from False to True",
                      scan_path=".")
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

- `fix_code_file` and `extend_code` preview changes by default (`dry_run=True` / `write=False`)
- Pass `dry_run=False` or `write=True` explicitly to apply changes to disk
- `fix_code_file` and `extend_code` create backups before writing (`backup=True`)
- Always review AI-generated code before using in production
- Run `explain_change_impact` before modifying shared infrastructure