"""
Code Assistant MCP Server
Automated code analysis, bug detection, and fixing
WITH FEEDBACK SUPPORT FOR CODE GENERATION
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader

import inspect
import json
import ast
from typing import Optional
import logging

# Import the actual implementation from tools directory
from tools.code_assistant.tool import (
    analyze_code_file_impl,
    fix_code_file_impl,
    suggest_improvements_impl,
    explain_code_impl,
    generate_tests_impl,
    refactor_code_impl,
    generate_code_impl,
    analyze_project_impl,
    get_project_dependencies_impl,
    scan_project_structure_impl
)

# Import tool control if available (optional)
try:
    from tools.tool_control import check_tool_enabled
except ImportError:
    # Fallback if tool_control not available
    def check_tool_enabled(category=None):
        def decorator(func):
            return func
        return decorator

try:
    from client.tool_meta import tool_meta
except Exception:
    # Fallback stub — metadata is attached but not used in server subprocess
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

# ── Failure taxonomy ──────────────────────────────────────────────────────────
try:
    from metrics import FailureKind, MCPToolError, JsonFormatter
except ImportError:
    try:
        from client.metrics import FailureKind, MCPToolError, JsonFormatter
    except ImportError:
        from enum import Enum
        class FailureKind(Enum):
            RETRYABLE      = "retryable"
            USER_ERROR     = "user_error"
            UPSTREAM_ERROR = "upstream_error"
            INTERNAL_ERROR = "internal_error"
        class MCPToolError(Exception):
            def __init__(self, kind, message, detail=None):
                self.kind = kind; self.message = message; self.detail = detail or {}
                super().__init__(message)
        JsonFormatter = None

from mcp.server.fastmcp import FastMCP

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = JsonFormatter() if JsonFormatter is not None else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_code_assistant").setLevel(logging.INFO)

logger = logging.getLogger("mcp_code_assistant")
logger.info("🚀 Code Assistant server logging initialized")

mcp = FastMCP("code-assistant-server")


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["analyze code","check code file","inspect code"],idempotent=True,example='use analyze_code_file: file_path="" [language=""] [deep_analysis=""]',intent_category="code_assistant")
def analyze_code_file(file_path: str, language: str = "auto", deep_analysis: bool = True) -> str:
    """
    Analyze a code file for bugs, anti-patterns, and issues.

    Supports Python (AST-based deep analysis), JavaScript, TypeScript, Rust, and Go.

    Args:
        file_path (str, required): Path to the code file to analyze
        language (str, optional): Language override ("auto", "python", "javascript", etc.)
        deep_analysis (bool, optional): Use deep AST analysis for Python (default: True)

    Returns:
        A JSON string containing the detected language, total issue count,
        and a detailed list of identified bugs/warnings with line numbers.

    Example:
        analyze_code_file("myapp/server.py")
        analyze_code_file("src/utils.js", language="javascript")

    Use cases:
        - Pre-commit checks
        - Code review assistance
        - Learning tool (understand why something is wrong)
        - Migration prep (find issues before refactoring)
    """
    logger.info(f"🔍 [TOOL] analyze_code_file called: {file_path}")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "analyze_code_file", "file_path": file_path})
    try:
        return analyze_code_file_impl(file_path, language, deep_analysis)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ analyze_code_file failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Analysis failed: {e}",
                           {"tool": "analyze_code_file", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["write","code","ai"],triggers=["fix code","fix file","auto fix"],idempotent=False,example='use fix_code_file: file_path="" [auto_fix=""] [backup=""] [dry_run=""]',intent_category="code_assistant")
def fix_code_file(file_path: str, auto_fix: bool = True, backup: bool = True, dry_run: bool = False) -> str:
    """
    Automatically fix detected issues in a code file.

    Creates backup, applies fixes, runs formatter.

    Args:
        file_path (str, required): Path to the code file
        auto_fix (bool, optional): Apply automatic fixes (True) or just show suggestions (False)
        backup (bool, optional): Create backup before fixing (default: True, recommended)
        dry_run (bool, optional): Show what would be fixed without actually modifying (default: False)

    Returns:
        JSON with:
        - fixes_applied: Number of fixes applied
        - details: List of what was fixed
        - backup_path: Path to backup file if created
        - formatted: Whether code was formatted after fixing
        - original_content: Original code (if dry_run=True)
        - new_content: Fixed code (if dry_run=True)

    Example:
        fix_code_file("buggy.py")                          # Fix with backup
        fix_code_file("test.py", auto_fix=False)          # Just show suggestions
        fix_code_file("script.py", dry_run=True)          # Preview changes

    Safety features:
        - Always creates backup by default
        - Validates fixes don't break syntax
        - Logs all changes
        - Can be reverted using backup
    """
    logger.info(f"🔧 [TOOL] fix_code_file called: {file_path} (auto_fix={auto_fix}, backup={backup})")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "fix_code_file", "file_path": file_path})
    try:
        return fix_code_file_impl(file_path, auto_fix, backup, dry_run)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ fix_code_file failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Fix failed: {e}",
                           {"tool": "fix_code_file", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["suggest improvements","code suggestions","improve code"],idempotent=True,example='use suggest_improvements: file_path="" [context=""] [focus=""]',intent_category="code_assistant")
def suggest_improvements(file_path: str, context: str = "", focus: str = "all") -> str:
    """
    Suggest code improvements and best practices.

    Args:
        file_path (str, required): Path to code file
        context (str, optional): Additional context about what you're building
        focus (str, optional): What to focus on: "all", "performance", "readability", "security"

    Returns:
        JSON with:
        - suggestions: List of improvement suggestions
          - type: "best_practice", "performance", "security", "documentation"
          - message: What to improve
          - reason: Why it matters
          - suggestion: How to implement it
          - priority: "high", "medium", "low"
        - language: Detected language
        - focus_area: What was analyzed

    Example:
        suggest_improvements("api.py", context="REST API server")
        suggest_improvements("utils.js", focus="performance")

    Types of suggestions:
        - Best practices (logging vs print, type hints, etc.)
        - Performance opportunities (list comprehensions, caching)
        - Security issues (SQL injection, XSS, etc.)
        - Documentation gaps (missing docstrings)
        - Code organization (function length, complexity)
    """
    logger.info(f"💡 [TOOL] suggest_improvements called: {file_path}")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "suggest_improvements", "file_path": file_path})
    try:
        return suggest_improvements_impl(file_path, context, focus)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ suggest_improvements failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Suggestions failed: {e}",
                           {"tool": "suggest_improvements", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["explain code","what does this code do","explain file"],idempotent=True,example='use explain_code: file_path="" [line_start=""] [line_end=""] [detail_level=""]',intent_category="code_assistant")
def explain_code(file_path: str, line_start: Optional[int] = None, line_end: Optional[int] = None, detail_level: str = "medium") -> str:
    """
    Explain what code does in natural language.

    Args:
        file_path (str, required): Path to code file
        line_start (int, optional): Start line (optional, explain specific section)
        line_end (int, optional): End line (optional)
        detail_level (str, optional): "brief", "medium", or "detailed"

    Returns:
        JSON with:
        - explanation: Plain English explanation
        - key_concepts: List of important concepts used
        - complexity: Estimated complexity
        - dependencies: External dependencies used

    Example:
        explain_code("algorithm.py")
        explain_code("utils.py", line_start=45, line_end=67)
        explain_code("complex.py", detail_level="detailed")

    Use cases:
        - Understanding unfamiliar code
        - Onboarding new developers
        - Code review explanations
        - Documentation generation
    """
    logger.info(f"📖 [TOOL] explain_code called: {file_path}")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "explain_code", "file_path": file_path})
    try:
        return explain_code_impl(file_path, line_start, line_end, detail_level)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ explain_code failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Explanation failed: {e}",
                           {"tool": "explain_code", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["generate tests","write tests","create tests"],idempotent=True,example='use generate_tests: file_path="" [test_framework=""] [coverage_target=""]',intent_category="code_assistant")
def generate_tests(file_path: str, test_framework: str = "auto", coverage_target: str = "functions") -> str:
    """
    Generate unit tests for code.

    Args:
        file_path (str, required): Path to source file to test
        test_framework (str, optional): "auto", "pytest", "unittest", "jest", etc.
        coverage_target (str, optional): "functions", "classes", "all"

    Returns:
        JSON with:
        - test_file_path: Path where tests were/should be saved
        - test_code: Generated test code
        - functions_covered: List of functions with tests
        - framework_used: Test framework chosen
        - coverage_estimate: Estimated code coverage %

    Example:
        generate_tests("myapp/utils.py")
        generate_tests("api.py", test_framework="pytest", coverage_target="all")

    Features:
        - Analyzes function signatures
        - Creates test cases for common scenarios
        - Includes edge case tests
        - Follows framework conventions
        - Generates fixtures and mocks
    """
    logger.info(f"🧪 [TOOL] generate_tests called: {file_path}")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "generate_tests", "file_path": file_path})
    try:
        return generate_tests_impl(file_path, test_framework, coverage_target)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ generate_tests failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Test generation failed: {e}",
                           {"tool": "generate_tests", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["write","code","ai"],triggers=["refactor","modernize code","simplify code"],idempotent=False,example='use refactor_code: file_path="" refactor_type="" [target=""] [preview=""]',intent_category="code_assistant")
def refactor_code(
    file_path: str,
    refactor_type: str,
    target: str = "",
    preview: bool = True
) -> str:
    """
    Refactor code using common patterns.

    Args:
        file_path (str, required): Path to code file
        refactor_type (str, required): Type of refactoring:
            - "extract_function": Extract code block into function
            - "rename": Rename variable/function
            - "simplify": Simplify complex expressions
            - "modernize": Update to modern syntax (f-strings, type hints, etc.)
            - "optimize": Apply performance optimizations
        target (str, optional): What to refactor (function name, line range, etc.)
        preview (bool, optional): Show preview without applying (default: True)

    Returns:
        JSON with:
        - refactor_type: Type of refactoring applied
        - changes: List of changes made
        - preview: Code preview if preview=True
        - applied: Whether changes were applied
        - backup_path: Path to backup if changes applied

    Example:
        refactor_code("app.py", "extract_function", target="lines:45-67")
        refactor_code("legacy.py", "modernize")
        refactor_code("utils.py", "rename", target="old_name:new_name")

    Refactoring types:
        - extract_function: DRY principle, reduce duplication
        - rename: Improve naming clarity
        - simplify: Reduce cognitive complexity
        - modernize: Use latest language features
        - optimize: Performance improvements
    """
    logger.info(f"♻️  [TOOL] refactor_code called: {file_path} ({refactor_type})")
    if not file_path or not Path(file_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "refactor_code", "file_path": file_path})
    valid_types = {"extract_function", "rename", "simplify", "modernize", "optimize"}
    if refactor_type not in valid_types:
        raise MCPToolError(FailureKind.USER_ERROR,
                           f"Invalid refactor_type '{refactor_type}'. Must be one of: {', '.join(valid_types)}",
                           {"tool": "refactor_code", "param": "refactor_type"})
    try:
        return refactor_code_impl(file_path, refactor_type, target, preview)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ refactor_code failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Refactor failed: {e}",
                           {"tool": "refactor_code", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["generate code","write code","create function","create class"],idempotent=True,example='use generate_code: description="" [language=""] [style=""] [include_tests=""] [include_docstrings=""] [framework=""] [output_file=""]',intent_category="code_assistant")
def generate_code(
    description: str,
    language: str = "python",
    style: str = "function",
    include_tests: bool = False,
    include_docstrings: bool = True,
    framework: str = "",
    output_file: str = ""
) -> str:
    """
    Generate code from a natural language description WITH QUALITY FEEDBACK.

    Creates production-ready code following best practices for the target language.
    Now includes automatic quality checks and improvement suggestions.

    Args:
        description (str, required): What the code should do (be specific and detailed)
        language (str, optional): Programming language: "python", "javascript", "typescript", "rust", "go"
        style (str, optional): Code style:
            - "function": Single function
            - "class": Class with methods
            - "module": Complete module/package
            - "script": Standalone script
            - "api_endpoint": REST API endpoint
        include_tests (bool, optional): Generate unit tests (default: False)
        include_docstrings (bool, optional): Include documentation (default: True)
        framework (str, optional): Optional framework: "fastapi", "flask", "react", "express", "actix"
        output_file (str, optional): Optional file path to save generated code

    Returns:
        JSON with:
        - generated_code: The generated code
        - language: Language used
        - style: Code style
        - framework: Framework used (if any)
        - includes_tests: Whether tests were included
        - includes_docs: Whether documentation was included
        - saved_to: File path if saved
        - prompt_used: The prompt sent to generate code
        - status: "success", "needs_improvement", or "low_quality"
        - feedback: Optional improvement suggestions

    Examples:
        generate_code("Calculate factorial recursively", "python", "function")
        generate_code("User authentication manager", "python", "class", include_tests=True)
        generate_code("REST API for todo items", "python", "api_endpoint", framework="fastapi")
        generate_code("React counter component", "javascript", "module", framework="react")

    Description tips:
        - Be specific about inputs/outputs
        - Mention edge cases to handle
        - Specify any constraints or requirements
        - Include examples if helpful

    Good: "Create a function that validates email addresses, returns True/False,
           handles edge cases like missing @ or domain, allows + in local part"

    Bad:  "email validator"
    """
    logger.info(f"✨ [TOOL] generate_code called: {description[:50]}...")
    if not description or not description.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "description must not be empty",
                           {"tool": "generate_code"})

    # Call the original implementation
    result_json = generate_code_impl(description, language, style, include_tests, include_docstrings, framework, output_file)

    # Parse the result
    try:
        result = json.loads(result_json)
    except json.JSONDecodeError:
        # If parsing fails, return as-is
        return result_json

    # ═══════════════════════════════════════════════════════════════
    # NEW: Quality checks and feedback
    # ═══════════════════════════════════════════════════════════════
    generated_code = result.get("generated_code", "")

    # Check 1: Code length (too short might mean incomplete)
    if generated_code and len(generated_code.strip()) < 50:
        result["status"] = "needs_improvement"
        result["feedback"] = {
            "reason": "Generated code is very short (< 50 chars). Description may be too vague.",
            "suggestions": [
                "Provide more detail about what the code should do",
                "Specify input/output types",
                "Mention edge cases to handle",
                "Add context about the use case"
            ],
            "auto_retry": False
        }
        logger.warning(f"⚠️ Generated code too short: {len(generated_code)} chars")
        return json.dumps(result, indent=2)

    # Check 2: Missing requested features
    issues = []

    if include_tests and "def test_" not in generated_code and "it(" not in generated_code:
        issues.append("Tests were requested but none were generated")

    if include_docstrings and language == "python":
        # Check for docstrings in Python
        if '"""' not in generated_code and "'''" not in generated_code:
            issues.append("Docstrings were requested but appear to be missing")

    if framework and framework.lower() not in generated_code.lower():
        issues.append(f"Framework '{framework}' was specified but doesn't appear in the code")

    # Check 3: Description vagueness (single-word descriptions are usually bad)
    desc_words = description.strip().split()
    if len(desc_words) <= 2:
        result["status"] = "needs_improvement"
        result["feedback"] = {
            "reason": f"Description is very brief ({len(desc_words)} words). More detail needed for quality code generation.",
            "suggestions": [
                "Describe what inputs the code should accept",
                "Specify what output it should produce",
                "Mention any constraints or requirements",
                "Provide an example of expected behavior"
            ],
            "example": f'Instead of: "{description}"\nTry: "Create a {style} that {description}, accepting X as input, returning Y, and handling Z edge case"',
            "auto_retry": False
        }
        logger.warning(f"⚠️ Vague description: {description}")
        return json.dumps(result, indent=2)

    # Check 4: Basic syntax check for Python
    if language == "python" and generated_code:
        try:
            ast.parse(generated_code)
            # Syntax is valid
        except SyntaxError as e:
            result["status"] = "needs_improvement"
            result["feedback"] = {
                "reason": f"Generated Python code has syntax errors: {str(e)}",
                "suggestions": [
                    "Try rephrasing the description",
                    "Specify the language constructs to use",
                    "Provide more context about the implementation"
                ],
                "syntax_error": str(e),
                "auto_retry": True  # Auto-retry syntax errors
            }
            logger.error(f"❌ Syntax error in generated code: {e}")
            return json.dumps(result, indent=2)

    # If we found issues but no syntax errors
    if issues:
        result["status"] = "low_quality"
        result["feedback"] = {
            "reason": "Code was generated but some requested features may be missing",
            "issues": issues,
            "suggestions": [
                "Verify the generated code includes all requested features",
                "Consider regenerating with more specific requirements",
                "Explicitly mention each feature in the description"
            ],
            "auto_retry": False
        }
        logger.info(f"ℹ️ Generated code has quality issues: {', '.join(issues)}")
        return json.dumps(result, indent=2)

    # All checks passed
    result["status"] = "success"
    logger.info(f"✅ Generated code passed quality checks")

    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="code_assistant")
@tool_meta(tags=["read","code","ai"],triggers=["analyze project","tech stack","project structure","what languages","dependencies"],idempotent=True,example='use analyze_project [project_path=""] [include_dependencies=""] [include_structure=""] [max_depth=""]',intent_category="code_assistant")
def analyze_project(
        project_path: str = ".",
        include_dependencies: bool = True,
        include_structure: bool = True,
        max_depth: int = 8
) -> str:
    """
    Analyze project structure, tech stack, and dependencies.

    Scans the project to determine:
    - Programming languages used (with file counts and line counts)
    - Frameworks and libraries detected
    - Dependencies from requirements.txt, package.json, etc.
    - Project structure overview
    - Tech stack summary

    Args:
        project_path (str, optional): Root path of project (default: current directory ".")
        include_dependencies (bool, optional): Parse dependency files (default: True)
        include_structure (bool, optional): Include directory tree (default: True)
        max_depth (int, optional): Maximum directory depth to scan (default: 3)

    Returns:
        JSON with:
        - project_name: Name of the project
        - languages: Languages used with file/line counts
        - frameworks: Detected frameworks (FastAPI, LangChain, MCP, etc.)
        - dependencies: Parsed from requirements.txt, package.json
        - file_counts: Count of each file type
        - structure: Directory tree
        - tech_stack: Human-readable tech stack summary

    Examples:
        analyze_project()                                    # Analyze current directory
        analyze_project("/path/to/project")                 # Analyze specific path
        analyze_project(".", max_depth=5)                   # Deeper scan
        analyze_project(".", include_structure=False)       # Skip structure

    Use cases:
        - "What's the tech stack for this project?"
        - "What languages are used in this codebase?"
        - "Show me the project structure"
        - "What dependencies does this project have?"
        - "What packages are in requirements.txt?"
        - "What version of FastAPI is installed?"
        - "Analyze the project I'm working on"
        - "What frameworks are being used?"

    NOTE: This tool covers both dependency listing and structure scanning.
    Use include_dependencies=True (default) for package details, and
    include_structure=True (default) for directory layout.

    IMPORTANT: Always use this tool to answer tech stack questions.
    Never guess or hallucinate the tech stack - scan the actual files.
    """
    logger.info(f"📊 [TOOL] analyze_project called: {project_path}")
    if project_path and project_path != "." and not Path(project_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"Project path does not exist: {project_path}",
                           {"tool": "analyze_project", "project_path": project_path})
    try:
        return analyze_project_impl(project_path, include_dependencies, include_structure, max_depth)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ analyze_project failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Project analysis failed: {e}",
                           {"tool": "analyze_project", "project_path": project_path})



@mcp.tool()
@check_tool_enabled(category="code_assistant")
def list_capabilities(filter_tags: str | None = None) -> str:
    """
    Return the full capability schema for every tool on this server.

    Agents call this to discover what this server can do, what parameters
    each tool accepts, and what constraints apply — without needing the
    client-side CapabilityRegistry.

    Args:
        filter_tags (str, optional): Comma-separated tags to filter by
                                     e.g. "read,search" or "write"

    Returns:
        JSON string with server name, tools array, and total count.
    """
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")

    try:
        from client.capability_registry import (
            _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    import sys as _sys, inspect as _inspect
    _current = _sys.modules[__name__]

    wanted_tags = set(t.strip() for t in filter_tags.split(",") if t.strip()) if filter_tags else None

    tools_out = []
    seen = set()
    for _name, _obj in vars(_current).items():
        if not callable(_obj) or _name.startswith("_") or _name in _INTERNAL_TOOLS:
            continue
        if not hasattr(_obj, "__tool_meta__") and not hasattr(_obj, "_mcp_tool"):
            continue
        if _name in seen:
            continue
        seen.add(_name)

        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue

        sig = _inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname in ("self",):
                continue
            has_default = param.default is not _inspect.Parameter.empty
            ann = param.annotation
            type_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "").replace("Optional[", "").rstrip("]")
                if ann is not _inspect.Parameter.empty else "string"
            )
            params.append({
                "name":     pname,
                "type":     type_str,
                "required": not has_default,
                "default":  None if not has_default else str(param.default),
            })

        tools_out.append({
            "name":         _name,
            "description":  (_obj.__doc__ or "").strip().split("\n")[0],
            "input_schema": params,
            "tags":         tags,
            "rate_limit":   _TOOL_RATE_LIMITS.get(_name),
            "idempotent":   _TOOL_IDEMPOTENT.get(_name, True),
        })

    return json.dumps({
        "server": mcp.name,
        "tools":  tools_out,
        "total":  len(tools_out),
    }, indent=2)

# Skill management tools
skill_registry = None

@mcp.tool()
@check_tool_enabled(category="code_assistant")
def list_skills() -> str:
    """List all available skills for code assistant."""
    logger.info("📚 [TOOL] list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "code-assistant-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "code-assistant-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="code_assistant")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"📖 [TOOL] read_skill called: {skill_name}")

    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)

    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content

    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({
        "error": f"Skill '{skill_name}' not found",
        "available_skills": available
    }, indent=2)


def get_tool_names_from_module():
    """Auto-discover tools from this module"""
    current_module = sys.modules[__name__]
    tool_names = []

    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)

    return tool_names


if __name__ == "__main__":
    # Auto-discover tools
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="code_assistant")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠️  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"📚 {len(skill_registry.skills)} skills loaded")

    mcp.run(transport="stdio")