"""
Comprehensive Code Review Tool for Python Files
Drop-in replacement for summarize_code_file with actual analysis
Includes WSL2 path handling
"""
import ast
import re
import os
from pathlib import Path
from typing import Dict, List, Any, Optional


def normalize_wsl_path(path_str: str) -> Optional[str]:
    """
    Normalize and validate paths for WSL2 environment.

    Handles:
    - /mnt/c/... paths (WSL format)
    - C:\... or C:/... (Windows format)
    - Case sensitivity issues
    - Relative paths

    Returns:
        Absolute path if file exists, None otherwise
    """
    if not path_str:
        return None

    # Try as-is first
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Convert Windows path to WSL (C:\... -> /mnt/c/...)
    if re.match(r'^[A-Za-z]:[/\\]', path_str):
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        wsl_path = f"/mnt/{drive}{rest}"

        if os.path.exists(wsl_path):
            return os.path.abspath(wsl_path)

    # Try case variations for /mnt/c/... paths
    if path_str.startswith('/mnt/'):
        # Try original
        if os.path.exists(path_str):
            return os.path.abspath(path_str)

        # Try with case-insensitive matching
        # This is a simplified approach - in production you'd walk the path
        variants = [
            path_str,
            path_str.lower(),
        ]

        for variant in variants:
            if os.path.exists(variant):
                return os.path.abspath(variant)

    # Try relative to current directory
    if not os.path.isabs(path_str):
        abs_path = os.path.abspath(path_str)
        if os.path.exists(abs_path):
            return abs_path

    return None


def review_python_file(file_path: str, max_bytes: int = 200000) -> Dict[str, Any]:
    """
    Perform comprehensive code review on a Python file OR directory.

    If a directory is provided, reviews all Python files within it (non-recursive).

    Args:
        file_path: Absolute path to Python file or directory
        max_bytes: Maximum file size to process per file (default: 200KB)

    Returns:
        Dictionary with detailed review including:
        - For single file: metrics, issues, recommendations
        - For directory: aggregated results from all files
    """
    # Normalize path for WSL2
    normalized_path = normalize_wsl_path(file_path)

    if normalized_path is None:
        # Path doesn't exist - provide helpful debugging info
        return {
            "error": f"Path not found: {file_path}",
            "debug_info": {
                "original_path": file_path,
                "current_directory": os.getcwd(),
                "path_exists": os.path.exists(file_path),
                "is_wsl_path": file_path.startswith('/mnt/'),
                "is_windows_path": bool(re.match(r'^[A-Za-z]:[/\\]', file_path)),
            },
            "suggestions": [
                "Verify the path exists in your filesystem",
                "Check if the path is correct (case-sensitive on Linux/WSL)",
                "Try using an absolute path",
                "Ensure you're in the correct directory"
            ]
        }

    path = Path(normalized_path)

    # Handle directory - review all Python files
    if path.is_dir():
        return _review_directory(path, max_bytes)

    # Handle single file
    if not path.is_file():
        return {
            "error": f"Path exists but is not a file or directory: {file_path}",
            "suggestion": "Provide a Python file path or directory path"
        }

    return _review_single_file(path, max_bytes)


def _review_directory(dir_path: Path, max_bytes: int) -> Dict[str, Any]:
    """Review all Python files in a directory."""

    # Find all Python files (non-recursive by default)
    py_files = sorted(dir_path.glob("*.py"))

    if not py_files:
        # Try recursive search
        py_files = sorted(dir_path.glob("**/*.py"))
        if not py_files:
            return {
                "error": f"No Python files found in: {dir_path}",
                "suggestion": "Ensure the directory contains .py files"
            }

    # Limit to reasonable number of files
    if len(py_files) > 50:
        return {
            "error": f"Too many Python files ({len(py_files)}) in directory",
            "suggestion": "Review specific files or subdirectories instead",
            "files_found": len(py_files),
            "sample_files": [str(f.name) for f in py_files[:10]]
        }

    # Review each file
    all_results = []
    aggregated_issues = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": [],
        "info": []
    }

    total_metrics = {
        "total_lines": 0,
        "total_functions": 0,
        "total_classes": 0,
        "files_reviewed": 0,
        "files_with_issues": 0
    }

    for py_file in py_files:
        try:
            result = _review_single_file(py_file, max_bytes)

            if "error" in result:
                # Skip files with errors
                continue

            all_results.append({
                "file": py_file.name,
                "relative_path": str(py_file.relative_to(dir_path)),
                "summary": result.get("summary", {}),
                "critical_count": result.get("summary", {}).get("critical", 0),
                "high_count": result.get("summary", {}).get("high", 0),
                "issues": result.get("issues_by_severity", {})
            })

            # Aggregate metrics
            metrics = result.get("metrics", {})
            total_metrics["total_lines"] += metrics.get("total_lines", 0)
            total_metrics["total_functions"] += metrics.get("functions", 0)
            total_metrics["total_classes"] += metrics.get("classes", 0)
            total_metrics["files_reviewed"] += 1

            # Aggregate issues
            summary = result.get("summary", {})
            if summary.get("total_issues", 0) > 0:
                total_metrics["files_with_issues"] += 1

            issues = result.get("issues_by_severity", {})
            for severity in ["critical", "high", "medium", "low", "info"]:
                for issue in issues.get(severity, []):
                    # Add file context to issue
                    issue_copy = issue.copy()
                    issue_copy["file"] = py_file.name
                    aggregated_issues[severity].append(issue_copy)

        except Exception as e:
            # Log but continue
            continue

    # Sort issues by severity and file
    for severity in aggregated_issues:
        aggregated_issues[severity] = sorted(
            aggregated_issues[severity],
            key=lambda x: (x.get("file", ""), x.get("line", 0))
        )

    # Generate directory-level recommendations
    recommendations = _generate_directory_recommendations(
        aggregated_issues,
        total_metrics,
        len(py_files)
    )

    return {
        "status": "complete",
        "type": "directory_review",
        "directory": str(dir_path),
        "metrics": total_metrics,
        "summary": {
            "total_files": len(py_files),
            "files_reviewed": total_metrics["files_reviewed"],
            "files_with_issues": total_metrics["files_with_issues"],
            "total_issues": sum(len(issues) for issues in aggregated_issues.values()),
            "critical": len(aggregated_issues["critical"]),
            "high": len(aggregated_issues["high"]),
            "medium": len(aggregated_issues["medium"]),
            "low": len(aggregated_issues["low"]),
            "info": len(aggregated_issues["info"])
        },
        "files": all_results[:20],  # Limit details shown
        "issues_by_severity": {
            "critical": aggregated_issues["critical"][:15],
            "high": aggregated_issues["high"][:15],
            "medium": aggregated_issues["medium"][:10],
            "low": aggregated_issues["low"][:5],
            "info": aggregated_issues["info"][:5]
        },
        "recommendations": recommendations,
        "review_complete": True
    }


def _review_single_file(path: Path, max_bytes: int) -> Dict[str, Any]:
    """Review a single Python file."""

    if path.suffix != '.py':
        return {
            "error": f"Not a Python file: {path.suffix}",
            "suggestion": "This tool only reviews Python (.py) files"
        }

    # Check file size
    file_size = path.stat().st_size
    if file_size > max_bytes:
        return {
            "error": f"File too large: {file_size} bytes (max: {max_bytes})",
            "suggestion": "Split large files or increase max_bytes parameter"
        }

    # Read file
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return {
            "error": "File encoding not supported",
            "suggestion": "Ensure file is UTF-8 encoded"
        }
    except Exception as e:
        return {"error": f"Failed to read file: {str(e)}"}

    # Perform analysis
    issues = []
    metrics = {}

    # Run all checks
    syntax_valid = _check_syntax(content, issues)
    metrics['syntax_valid'] = syntax_valid

    _analyze_structure(content, issues, metrics)
    _analyze_quality(content, issues)
    _analyze_security(content, issues)
    _analyze_performance(content, issues)

    # Categorize issues
    critical = [i for i in issues if i['severity'] == 'critical']
    high = [i for i in issues if i['severity'] == 'high']
    medium = [i for i in issues if i['severity'] == 'medium']
    low = [i for i in issues if i['severity'] == 'low']
    info = [i for i in issues if i['severity'] == 'info']

    # Build result
    lines = content.split('\n')
    return {
        "status": "complete",
        "file": str(path),
        "file_size_bytes": file_size,
        "metrics": {
            "total_lines": len(lines),
            "blank_lines": metrics.get('blank_lines', 0),
            "comment_lines": metrics.get('comment_lines', 0),
            "code_lines": metrics.get('code_lines', 0),
            "functions": metrics.get('functions', 0),
            "classes": metrics.get('classes', 0),
            "syntax_valid": syntax_valid
        },
        "summary": {
            "total_issues": len(issues),
            "critical": len(critical),
            "high": len(high),
            "medium": len(medium),
            "low": len(low),
            "info": len(info)
        },
        "issues_by_severity": {
            "critical": critical[:10],  # Limit output
            "high": high[:10],
            "medium": medium[:10],
            "low": low[:5],
            "info": info[:5]
        },
        "recommendations": _generate_recommendations(issues, metrics),
        "review_complete": True
    }


def _generate_directory_recommendations(aggregated_issues: dict, metrics: dict, total_files: int) -> List[str]:
    """Generate recommendations for directory review."""
    recommendations = []

    critical_count = len(aggregated_issues.get("critical", []))
    high_count = len(aggregated_issues.get("high", []))

    if critical_count > 0:
        recommendations.append(
            f"🚨 {critical_count} critical security issue(s) across {metrics['files_with_issues']} files - address immediately"
        )

    if high_count > 0:
        recommendations.append(
            f"⚡ {high_count} high-severity issue(s) - should be fixed soon"
        )

    if metrics["files_with_issues"] == 0:
        recommendations.append(
            f"✅ All {metrics['files_reviewed']} files look good!"
        )
    elif metrics["files_with_issues"] < metrics["files_reviewed"] * 0.3:
        recommendations.append(
            f"👍 Most files are clean - {metrics['files_with_issues']}/{metrics['files_reviewed']} need attention"
        )
    else:
        recommendations.append(
            f"⚠️  {metrics['files_with_issues']}/{metrics['files_reviewed']} files have issues - consider code review process"
        )

    # Check for documentation
    total_lines = metrics.get("total_lines", 1)
    if total_lines > 500:
        recommendations.append(
            f"📝 Directory contains {total_lines} lines - ensure adequate documentation"
        )

    return recommendations


def _check_syntax(content: str, issues: list) -> bool:
    """Check if Python code has valid syntax."""
    try:
        ast.parse(content)
        return True
    except SyntaxError as e:
        issues.append({
            "severity": "critical",
            "type": "syntax_error",
            "line": e.lineno or 1,
            "message": f"Syntax error: {e.msg}",
            "suggestion": "Fix syntax before code can run",
            "code_snippet": str(e.text).strip() if e.text else None
        })
        return False


def _analyze_structure(content: str, issues: list, metrics: dict):
    """Analyze code structure and organization."""
    lines = content.split('\n')

    # Basic metrics
    blank_lines = sum(1 for line in lines if not line.strip())
    comment_lines = sum(1 for line in lines if line.strip().startswith('#'))

    metrics['blank_lines'] = blank_lines
    metrics['comment_lines'] = comment_lines
    metrics['total_lines'] = len(lines)
    metrics['code_lines'] = len(lines) - blank_lines - comment_lines

    try:
        tree = ast.parse(content)

        # Count constructs
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

        metrics['functions'] = len(functions)
        metrics['classes'] = len(classes)

        # Module docstring
        if not ast.get_docstring(tree):
            issues.append({
                "severity": "low",
                "type": "missing_docstring",
                "line": 1,
                "message": "Missing module docstring",
                "suggestion": "Add docstring explaining module purpose"
            })

        # Function analysis
        for func in functions:
            if not ast.get_docstring(func):
                issues.append({
                    "severity": "medium",
                    "type": "missing_docstring",
                    "line": func.lineno,
                    "message": f"Function '{func.name}' missing docstring",
                    "suggestion": "Document parameters, return value, and purpose"
                })

            # Function length
            if hasattr(func, 'end_lineno'):
                func_length = func.end_lineno - func.lineno
                if func_length > 50:
                    issues.append({
                        "severity": "medium",
                        "type": "long_function",
                        "line": func.lineno,
                        "message": f"Function '{func.name}' is {func_length} lines",
                        "suggestion": "Break into smaller functions (<50 lines each)"
                    })

            # Too many parameters
            num_args = len(func.args.args)
            if num_args > 5:
                issues.append({
                    "severity": "medium",
                    "type": "too_many_parameters",
                    "line": func.lineno,
                    "message": f"Function '{func.name}' has {num_args} parameters",
                    "suggestion": "Use dataclass, dict, or kwargs for complex parameters"
                })

        # Class analysis
        for cls in classes:
            if not ast.get_docstring(cls):
                issues.append({
                    "severity": "medium",
                    "type": "missing_docstring",
                    "line": cls.lineno,
                    "message": f"Class '{cls.name}' missing docstring",
                    "suggestion": "Document class purpose and usage"
                })

    except SyntaxError:
        pass  # Already reported in syntax check


def _analyze_quality(content: str, issues: list):
    """Check code quality issues."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Line length
        if len(line) > 120:
            issues.append({
                "severity": "low",
                "type": "line_too_long",
                "line": i,
                "message": f"Line exceeds 120 characters ({len(line)})",
                "suggestion": "Break into multiple lines for readability"
            })

        # Print statements
        if re.search(r'\bprint\s*\(', stripped) and not stripped.startswith('#'):
            issues.append({
                "severity": "low",
                "type": "print_statement",
                "line": i,
                "message": "Using print() instead of logging",
                "suggestion": "Use logging module for production code"
            })

        # Bare except
        if re.match(r'^\s*except\s*:', line):
            issues.append({
                "severity": "high",
                "type": "bare_except",
                "line": i,
                "message": "Bare except catches all exceptions",
                "suggestion": "Catch specific exceptions (e.g., except ValueError:)"
            })

        # Silent exceptions
        if stripped == 'pass' and i > 1:
            prev = lines[i-2].strip() if i > 1 else ""
            if prev.startswith('except'):
                issues.append({
                    "severity": "high",
                    "type": "silent_exception",
                    "line": i,
                    "message": "Exception silently swallowed",
                    "suggestion": "Log errors or handle appropriately"
                })

        # TODO/FIXME
        if 'TODO' in stripped or 'FIXME' in stripped:
            issues.append({
                "severity": "info",
                "type": "todo",
                "line": i,
                "message": "Pending work item",
                "suggestion": "Track and complete TODO items"
            })


def _analyze_security(content: str, issues: list):
    """Check for security vulnerabilities."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # eval/exec
        if re.search(r'\b(eval|exec)\s*\(', stripped):
            issues.append({
                "severity": "critical",
                "type": "dangerous_function",
                "line": i,
                "message": "Dangerous eval()/exec() usage",
                "suggestion": "Use ast.literal_eval() or safer alternatives"
            })

        # shell=True
        if 'shell=True' in stripped:
            issues.append({
                "severity": "high",
                "type": "shell_injection_risk",
                "line": i,
                "message": "subprocess with shell=True is unsafe",
                "suggestion": "Use shell=False with list arguments"
            })

        # Hardcoded credentials (but be careful not to flag legitimate test code)
        if re.search(r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+["\']', stripped, re.I):
            # Skip if it's clearly a placeholder or test value
            if not any(x in stripped.lower() for x in ['test', 'example', 'placeholder', 'your_', 'xxx']):
                issues.append({
                    "severity": "critical",
                    "type": "hardcoded_credentials",
                    "line": i,
                    "message": "Possible hardcoded credentials",
                    "suggestion": "Use environment variables or secure vault"
                })

        # SQL injection
        if re.search(r'(SELECT|INSERT|UPDATE|DELETE).*\+', stripped, re.I):
            # Skip if it's clearly not SQL (e.g., just string concatenation)
            if any(x in stripped for x in ['FROM', 'WHERE', 'INTO', 'SET']):
                issues.append({
                    "severity": "critical",
                    "type": "sql_injection_risk",
                    "line": i,
                    "message": "Possible SQL injection vulnerability",
                    "suggestion": "Use parameterized queries"
                })


def _analyze_performance(content: str, issues: list):
    """Check for performance issues."""
    lines = content.split('\n')

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Nested comprehensions
        if re.search(r'\[.*for.*for.*\]', stripped):
            issues.append({
                "severity": "medium",
                "type": "nested_comprehension",
                "line": i,
                "message": "Complex nested comprehension",
                "suggestion": "Consider generator expressions for large data"
            })

        # Global variables
        if stripped.startswith('global '):
            issues.append({
                "severity": "medium",
                "type": "global_variable",
                "line": i,
                "message": "Using global variable",
                "suggestion": "Use class attributes or parameters instead"
            })


def _generate_recommendations(issues: list, metrics: dict) -> list:
    """Generate high-level recommendations."""
    recommendations = []

    # Comment ratio
    comment_ratio = metrics.get('comment_lines', 0) / max(metrics.get('total_lines', 1), 1)
    if comment_ratio < 0.1:
        recommendations.append(
            "⚠️  Low comment ratio - consider adding more documentation"
        )

    # Critical issues
    critical = sum(1 for i in issues if i['severity'] == 'critical')
    if critical > 0:
        recommendations.append(
            f"🚨 {critical} critical issue(s) - must fix before deployment"
        )

    # High severity
    high = sum(1 for i in issues if i['severity'] == 'high')
    if high > 0:
        recommendations.append(
            f"⚡ {high} high-severity issue(s) - address as soon as possible"
        )

    # Code organization
    if metrics.get('functions', 0) > 20:
        recommendations.append(
            "📦 Consider splitting this file into multiple modules"
        )

    # Documentation
    missing_docs = sum(1 for i in issues if i['type'] == 'missing_docstring')
    if missing_docs > 3:
        recommendations.append(
            f"📝 Add docstrings to {missing_docs} functions/classes"
        )

    if not recommendations:
        recommendations.append("✅ Code looks good! No major issues found.")

    return recommendations