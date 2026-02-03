---
name: github_review
description: >
  Clone and review GitHub repositories. Analyze entire codebases,
  review architecture, find bugs, and get improvement suggestions.
  Works with existing code_assistant and code_review tools.
tags:
  - github
  - code-review
  - repository
  - analysis
  - architecture
tools:
  - github_clone_repo
  - github_list_files
  - github_get_file_content
  - github_cleanup_repo
---

# GitHub Repository Review Skill

## 🎯 Overview

Clone and analyze GitHub repositories using existing code analysis tools.

**Capabilities:**
1. **Clone Repos** - Download any GitHub repository
2. **List Files** - Find Python/JS/TS files to review
3. **Architecture Review** - Analyze entire codebase
4. **Bug Detection** - Find issues across all files
5. **Cleanup** - Remove cloned repos when done

**Access:**
- Public repos: Works out of the box
- Private repos: Set `GITHUB_TOKEN=ghp_xxxxx` in `.env`

---

## 📋 Workflow

### Step 1: Clone Repository

```python
github_clone_repo("https://github.com/user/repo")
github_clone_repo("https://github.com/user/repo/tree/develop")  # Specific branch
```

**Returns:**
```json
{
  "local_path": "/tmp/github_repo_xyz/",
  "owner": "user",
  "repo": "repo",
  "branch": "main",
  "files_count": 145,
  "size_mb": 2.4
}
```

### Step 2: List Reviewable Files

```python
github_list_files(
    local_path="/tmp/github_repo_xyz/",
    extensions=["py", "js", "ts"]
)
```

**Returns:**
```json
{
  "files": [
    "src/main.py",
    "src/utils.py",
    "tests/test_main.py"
  ],
  "total_files": 3,
  "by_extension": {"py": 3}
}
```

### Step 3: Review Files

**Use existing code_assistant/code_review tools:**

```python
# Analyze architecture
analyze_project(project_path="/tmp/github_repo_xyz/")

# Review specific file
review_code(path="/tmp/github_repo_xyz/src/main.py")

# Analyze for bugs
analyze_code_file(file_path="/tmp/github_repo_xyz/src/utils.py")
```

### Step 4: Cleanup

```python
github_cleanup_repo(local_path="/tmp/github_repo_xyz/")
```

---

## 🚀 Complete Examples

### Example 1: Full Repository Review

```
User: "Review https://github.com/anthropics/anthropic-sdk-python"

Agent workflow:
1. github_clone_repo("https://github.com/anthropics/anthropic-sdk-python")
   → local_path: "/tmp/github_anthropic-sdk-python_xyz/"

2. analyze_project(project_path="/tmp/github_anthropic-sdk-python_xyz/")
   → Tech stack: Python, FastAPI, Pytest
   → 45 files, 12,453 lines

3. github_list_files(local_path="...", extensions=["py"])
   → 42 Python files found

4. review_code(path="/tmp/.../src/anthropic/client.py")
   → Found 2 issues: Missing type hints, bare except

5. review_code(path="/tmp/.../src/anthropic/api.py")
   → Found 1 issue: Long function (complexity)

6. github_cleanup_repo(local_path="/tmp/...")
   → Cleaned up

Agent response:
"I reviewed the anthropic-sdk-python repository:

**Architecture:**
- Python SDK for Anthropic API
- FastAPI framework
- Well-structured with src/ layout
- 42 Python files, ~12K lines

**Issues Found:**
- src/anthropic/client.py:
  • Missing type hints on 3 functions
  • Bare except clause (line 45)
  
- src/anthropic/api.py:
  • Function too long (75 lines, complexity 12)

**Recommendations:**
1. Add type hints for better IDE support
2. Replace bare except with specific exceptions
3. Refactor long functions into smaller ones

Overall: High quality codebase with minor improvements needed."
```

### Example 2: Security Review

```
User: "Check https://github.com/user/api-server for security issues"

Agent workflow:
1. github_clone_repo("https://github.com/user/api-server")
2. github_list_files(extensions=["py", "js"])
3. review_code(path="/tmp/.../server.py")
   → CRITICAL: SQL injection vulnerability
   → HIGH: Hardcoded secrets
4. review_code(path="/tmp/.../auth.py")
   → CRITICAL: Weak password hashing

Agent response:
"🚨 SECURITY ISSUES FOUND:

**CRITICAL:**
- server.py (line 67): SQL injection via f-string
- auth.py (line 23): Using MD5 for passwords (INSECURE!)

**HIGH:**
- config.py (line 12): Hardcoded API key

**Recommendations:**
1. Use parameterized queries (URGENT!)
2. Switch to bcrypt/argon2 for passwords
3. Move secrets to environment variables"
```

### Example 3: Compare Architecture

```
User: "Compare my local project with https://github.com/similar/repo"

Agent workflow:
1. analyze_project(project_path=".")
   → Your project: Flask, SQLite, 23 files
   
2. github_clone_repo("https://github.com/similar/repo")
3. analyze_project(project_path="/tmp/...")
   → Similar repo: FastAPI, PostgreSQL, 45 files

Agent response:
"**Comparison:**

Your Project:
- Flask (older, synchronous)
- SQLite (local, simple)
- 23 files, ~5K lines

Similar Repo:
- FastAPI (modern, async)
- PostgreSQL (production-ready)
- 45 files, ~12K lines

**Suggestions:**
1. Consider migrating to FastAPI for async support
2. Add PostgreSQL for better scalability
3. Add type hints (they use them extensively)
4. Adopt their test structure (pytest + coverage)"
```

---

## 🔧 Tool Reference

| Tool | Purpose | Key Args |
|------|---------|----------|
| `github_clone_repo` | Download repository | `url`, `branch`, `depth` |
| `github_list_files` | Find files to review | `local_path`, `extensions` |
| `github_get_file_content` | Read specific file | `local_path`, `file_path` |
| `github_cleanup_repo` | Delete cloned repo | `local_path` |

**After cloning, use these existing tools:**

| Tool | Purpose | From Server |
|------|---------|-------------|
| `analyze_project` | Tech stack analysis | code_assistant |
| `analyze_code_file` | Bug detection | code_assistant |
| `review_code` | Security & quality review | code_review |
| `scan_project_structure` | Directory tree | code_assistant |

---

## ⚙️  Configuration

### Public Repos (No Token)

Works out of the box:
```bash
# No setup needed
```

### Private Repos (Requires Token)

1. Create token at: https://github.com/settings/tokens
2. Add to `.env`:
   ```bash
   GITHUB_TOKEN=ghp_your_token_here
   ```
3. Restart client

**Token Permissions:**
- `repo` - Access private repositories
- `read:org` - Read organization repos (optional)

---

## 📊 Usage Patterns

### Pattern 1: Quick Check

```
User: "Check https://github.com/user/repo for bugs"

Steps:
1. Clone
2. List Python files
3. Review each file
4. Cleanup
5. Report issues
```

### Pattern 2: Architecture Review

```
User: "Analyze architecture of https://github.com/user/repo"

Steps:
1. Clone
2. analyze_project() for tech stack
3. scan_project_structure() for layout
4. List key files (main.py, __init__.py, etc.)
5. Explain architecture
6. Cleanup
```

### Pattern 3: Specific File

```
User: "Review https://github.com/user/repo/blob/main/src/api.py"

Steps:
1. Clone (shallow, depth=1)
2. review_code("src/api.py")
3. Report issues
4. Cleanup
```

---

## ⚠️  Important Notes

**Rate Limits:**
- Without token: 60 requests/hour
- With token: 5,000 requests/hour

**Storage:**
- Repos cloned to `/tmp/` (automatic cleanup)
- Always run `github_cleanup_repo()` when done

**Performance:**
- Shallow clones (depth=1) are faster
- Filter by extensions to reduce files to review
- Large repos (500+ files) take 2-5 minutes

**Limitations:**
- Cannot review binary files
- Some repos may be too large for detailed review
- Cloning very large repos (100+ MB) may be slow

---

## 🚀 Quick Start

**Review a repo:**
```python
# 1. Clone
github_clone_repo("https://github.com/user/repo")

# 2. Get the local path from the result
local_path = "/tmp/github_repo_xyz/"

# 3. Analyze
analyze_project(project_path=local_path)

# 4. Review files
github_list_files(local_path=local_path, extensions=["py"])
review_code(path=f"{local_path}/src/main.py")

# 5. Cleanup
github_cleanup_repo(local_path=local_path)
```

**Check specific file:**
```python
github_clone_repo("https://github.com/user/repo")
review_code(path="/tmp/github_repo_xyz/src/api.py")
github_cleanup_repo("/tmp/github_repo_xyz/")
```