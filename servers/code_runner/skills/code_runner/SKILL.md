---
name: code_runner
description: >
  Execute Python snippets, run Python files, run bash/shell commands, and install packages.
  Useful for calculations, data manipulation, regex testing, date generation, JSON parsing,
  git operations, file system queries, and any task requiring guaranteed correct computation
  rather than LLM estimation.
tags:
  - code
  - python
  - bash
  - shell
  - execution
  - calculation
  - scripting
  - git
tools:
  - run_python
  - run_python_file
  - run_bash
  - pip_install
---

# Code Runner Skill

## Tool Routing

| User asks... | Tool |
|---|---|
| "calculate / compute / what is X" | `run_python` |
| "run this code / execute this snippet" | `run_python` |
| "parse this JSON / CSV" | `run_python` |
| "test if this regex matches" | `run_python` |
| "generate a list of dates / numbers" | `run_python` |
| "run this script / execute this file" | `run_python_file` |
| "git status / git log / git diff" | `run_bash` |
| "list files / check disk / find files" | `run_bash` |
| "run a shell command" | `run_bash` |
| "install package X" | `pip_install` |

## Key Workflows

**Quick calculation:**
```
run_python: code="print(round(847.23 * 0.15, 2))"
```

**Date generation:**
```
run_python: code="from datetime import date, timedelta
start = date(2026, 3, 1)
dates = [str(start + timedelta(days=i)) for i in range(46)]
print(dates)"
```

**Regex testing:**
```
run_python: code="import re
pattern = r'^\d{4}-\d{2}-\d{2}$'
tests = ['2026-03-30', 'not-a-date', '2026-3-1']
for t in tests:
    print(t, '->', bool(re.match(pattern, t)))"
```

**Capture specific variables:**
```
run_python: code="total = sum(range(1, 101))" capture_vars="total"
```

**Run an existing script:**
```
run_python_file: file_path="/mnt/c/Users/Michael/PycharmProjects/mcp-platform/tools/some_script.py"
run_python_file: file_path="/path/to/script.py" args="--input data.csv --verbose"
```

**Git operations:**
```
run_bash: command="git status" cwd="/mnt/c/Users/Michael/PycharmProjects/mcp-platform"
run_bash: command="git log --oneline -10" cwd="/mnt/c/Users/Michael/PycharmProjects/Shashin"
run_bash: command="git diff HEAD~1" cwd="/mnt/c/Users/Michael/PycharmProjects/mcp-platform"
```

**File system:**
```
run_bash: command="find /mnt/c/Users/Michael/Downloads -name '*.jpg' | wc -l"
run_bash: command="du -sh /mnt/c/Users/Michael/PycharmProjects/mcp-platform"
run_bash: command="ls -la /mnt/c/Users/Michael/Downloads/shashinpics"
```

**Install a package:**
```
pip_install: package="gitpython"
pip_install: package="pandas" upgrade="true"
```

## Safety

### run_python
- Network access blocked (`requests`, `urllib`, `socket`, etc.)
- `exec`, `eval`, `open`, `subprocess` calls blocked
- Execution capped at 60 seconds max (default: 30s)
- Runs in isolated subprocess

### run_bash
- `rm -rf /`, `shutdown`, `reboot`, `mkfs`, `curl|bash` patterns blocked
- Execution capped at 120 seconds max (default: 30s)
- Use `cwd` to set the working directory

### pip_install
- Shell metacharacters blocked in package name
- Times out after 120 seconds
- Reports if package was already installed

## Tips

- Use `print()` in `run_python` — the last expression is not auto-printed
- Use `capture_vars="var1,var2"` to inspect variable values after execution
- For `run_bash` git commands always set `cwd` to the repo path
- `math`, `json`, `re`, `datetime`, `itertools`, `collections`, `statistics` available in `run_python`