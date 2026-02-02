"""
Code Assistant Tool Implementation
===================================
Location: tools/code_assistant/tool.py

This is the actual implementation called by servers/code_assistant/server.py
"""

import ast
import json
import shutil
import subprocess
import logging
import re
import os
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ===========================================================================
# LANGUAGE CONFIGURATION
# ===========================================================================

@dataclass
class LanguageConfig:
    name: str
    extensions: List[str]
    comment_single: str
    linter_command: Optional[List[str]] = None
    formatter_command: Optional[List[str]] = None


SUPPORTED_LANGUAGES = {
    "python": LanguageConfig(
        name="python",
        extensions=[".py"],
        comment_single="#",
        linter_command=["ruff", "check", "{file}"],
        formatter_command=["black", "--quiet", "{file}"]
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=[".js", ".jsx"],
        comment_single="//",
        linter_command=["eslint", "{file}"],
        formatter_command=["prettier", "--write", "{file}"]
    ),
    "typescript": LanguageConfig(
        name="typescript",
        extensions=[".ts", ".tsx"],
        comment_single="//",
        linter_command=["eslint", "{file}"],
        formatter_command=["prettier", "--write", "{file}"]
    ),
    "java": LanguageConfig(
        name="java",
        extensions=[".java"],
        comment_single="//",
        linter_command=["checkstyle", "-c", "/google_checks.xml", "{file}"],
        formatter_command=["google-java-format", "--replace", "{file}"]
    ),
    "kotlin": LanguageConfig(
        name="kotlin",
        extensions=[".kt", ".kts"],
        comment_single="//",
        linter_command=["ktlint", "{file}"],
        formatter_command=["ktlint", "-F", "{file}"]
    ),
}


def detect_language(file_path: str) -> Optional[LanguageConfig]:
    """Detect language from file extension"""
    ext = Path(file_path).suffix.lower()
    for lang in SUPPORTED_LANGUAGES.values():
        if ext in lang.extensions:
            return lang
    return None


# ===========================================================================
# PYTHON BUG DETECTOR
# ===========================================================================

class PythonBugDetector:
    @staticmethod
    def analyze(file_path: str) -> List[Dict[str, Any]]:
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()
            tree = ast.parse(code, filename=file_path)

            issues.extend(PythonBugDetector._check_mutable_defaults(tree))
            issues.extend(PythonBugDetector._check_bare_except(tree))
            issues.extend(PythonBugDetector._check_comparisons(tree))
            issues.extend(PythonBugDetector._check_unused_imports(tree, code))

        except SyntaxError as e:
            issues.append({
                "severity": "error",
                "type": "SyntaxError",
                "line": e.lineno or 0,
                "message": f"Syntax error: {e.msg}",
                "suggestion": "Fix syntax error before proceeding"
            })
        return issues

    @staticmethod
    def _check_mutable_defaults(tree) -> List[Dict]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        arg_idx = len(node.args.args) - len(node.args.defaults) + i
                        arg_name = node.args.args[arg_idx].arg
                        issues.append({
                            "severity": "error",
                            "type": "MutableDefault",
                            "line": node.lineno,
                            "message": f"Function '{node.name}' has mutable default argument '{arg_name}='",
                            "suggestion": f"Use None as default, then: if {arg_name} is None: {arg_name} = []",
                            "fix_type": "mutable_default"
                        })
        return issues

    @staticmethod
    def _check_bare_except(tree) -> List[Dict]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append({
                    "severity": "warning",
                    "type": "BareExcept",
                    "line": node.lineno,
                    "message": "Bare 'except:' catches all exceptions including SystemExit",
                    "suggestion": "Use 'except Exception as e:' instead",
                    "fix_type": "bare_except"
                })
        return issues

    @staticmethod
    def _check_comparisons(tree) -> List[Dict]:
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for op, comp in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        if isinstance(comp, (ast.Constant, ast.Num, ast.Str)):
                            issues.append({
                                "severity": "error",
                                "type": "IdentityComparison",
                                "line": node.lineno,
                                "message": "Using 'is' to compare with literal",
                                "suggestion": "Use '==' for value comparison",
                                "fix_type": "identity_comparison"
                            })
        return issues

    @staticmethod
    def _check_unused_imports(tree, code) -> List[Dict]:
        issues = []
        imports = {}
        used_names = set()

        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno
            elif isinstance(node, ast.Name):
                used_names.add(node.id)

        for imp_name, lineno in imports.items():
            if imp_name not in used_names and imp_name in code:
                issues.append({
                    "severity": "info",
                    "type": "UnusedImport",
                    "line": lineno,
                    "message": f"Import '{imp_name}' appears unused",
                    "suggestion": "Remove or comment out",
                    "fix_type": "unused_import"
                })
        return issues


# ===========================================================================
# JAVA BUG DETECTOR
# ===========================================================================

class JavaBugDetector:
    """Simple Java code analyzer using regex patterns"""

    @staticmethod
    def analyze(file_path: str) -> List[Dict[str, Any]]:
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Check for missing @Override
                if JavaBugDetector._is_override_method(line_stripped, i, lines):
                    prev_line = lines[i-2].strip() if i > 1 else ""
                    if "@Override" not in prev_line:
                        issues.append({
                            "severity": "warning",
                            "type": "MissingOverride",
                            "line": i,
                            "message": "Method appears to override but missing @Override annotation",
                            "suggestion": "Add @Override annotation above method"
                        })

                # Check for inefficient string concatenation in loops
                if "+=" in line and "String" in "".join(lines[max(0, i-5):i]):
                    issues.append({
                        "severity": "warning",
                        "type": "InefficientStringConcat",
                        "line": i,
                        "message": "String concatenation in loop is inefficient",
                        "suggestion": "Use StringBuilder instead"
                    })

                # Check for potential null pointer dereference
                if "." in line and "if" not in line and "null" in "".join(lines[max(0, i-3):i]):
                    # Simple heuristic - could be improved
                    if not any(check in line for check in ["!=", "==", "?"]):
                        issues.append({
                            "severity": "info",
                            "type": "PotentialNullPointer",
                            "line": i,
                            "message": "Potential null pointer dereference",
                            "suggestion": "Add null check before dereferencing"
                        })

                # Check for empty catch blocks
                if line_stripped == "} catch" or line_stripped.startswith("} catch("):
                    # Look ahead for empty catch
                    next_idx = i
                    while next_idx < len(lines):
                        next_line = lines[next_idx].strip()
                        if next_line == "{":
                            if next_idx + 1 < len(lines) and lines[next_idx + 1].strip() == "}":
                                issues.append({
                                    "severity": "warning",
                                    "type": "EmptyCatchBlock",
                                    "line": i,
                                    "message": "Empty catch block swallows exceptions",
                                    "suggestion": "Add logging or re-throw exception"
                                })
                            break
                        next_idx += 1

        except Exception as e:
            logger.error(f"Java analysis failed: {e}")
            issues.append({
                "severity": "error",
                "type": "AnalysisError",
                "line": 0,
                "message": f"Failed to analyze: {str(e)}"
            })

        return issues

    @staticmethod
    def _is_override_method(line: str, line_num: int, lines: List[str]) -> bool:
        """Check if this looks like a method that overrides"""
        # Common override patterns
        override_patterns = [
            r'public\s+\w+\s+toString\s*\(',
            r'public\s+boolean\s+equals\s*\(',
            r'public\s+int\s+hashCode\s*\(',
            r'public\s+int\s+compareTo\s*\(',
        ]

        for pattern in override_patterns:
            if re.search(pattern, line):
                return True
        return False


# ===========================================================================
# KOTLIN BUG DETECTOR
# ===========================================================================

class KotlinBugDetector:
    """Simple Kotlin code analyzer using regex patterns"""

    @staticmethod
    def analyze(file_path: str) -> List[Dict[str, Any]]:
        issues = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for i, line in enumerate(lines, 1):
                line_stripped = line.strip()

                # Check for !! (force unwrap) operator
                if "!!" in line_stripped:
                    issues.append({
                        "severity": "warning",
                        "type": "ForceUnwrap",
                        "line": i,
                        "message": "Using !! (force unwrap) can throw NullPointerException",
                        "suggestion": "Use safe call (?.) or let/also/apply instead"
                    })

                # Check for mutable collections when immutable would work
                if "mutableListOf" in line_stripped or "mutableMapOf" in line_stripped:
                    # Check if it's reassigned in next few lines
                    is_modified = False
                    for j in range(i, min(i+10, len(lines))):
                        if ".add(" in lines[j] or ".put(" in lines[j] or ".remove(" in lines[j]:
                            is_modified = True
                            break

                    if not is_modified:
                        issues.append({
                            "severity": "info",
                            "type": "UnnecessaryMutable",
                            "line": i,
                            "message": "Using mutable collection but appears to be immutable",
                            "suggestion": "Use listOf() or mapOf() instead"
                        })

                # Check for Java-style iteration
                if re.search(r'for\s*\(\s*\w+\s+\w+\s*:\s*\w+\s*\)', line_stripped):
                    issues.append({
                        "severity": "info",
                        "type": "NonIdiomatic",
                        "line": i,
                        "message": "Using Java-style for loop",
                        "suggestion": "Use Kotlin's for (item in collection) syntax"
                    })

                # Check for explicit type when type inference works
                if re.search(r'val\s+\w+\s*:\s*\w+\s*=\s*\w+\(', line_stripped):
                    issues.append({
                        "severity": "info",
                        "type": "RedundantTypeAnnotation",
                        "line": i,
                        "message": "Explicit type annotation may be redundant",
                        "suggestion": "Let type inference work: val name = value()"
                    })

                # Check for empty when expression
                if line_stripped.startswith("when") and "{" in line_stripped:
                    next_idx = i
                    while next_idx < len(lines):
                        next_line = lines[next_idx].strip()
                        if next_line == "}":
                            # Check if empty
                            content_lines = [lines[j].strip() for j in range(i, next_idx)]
                            if not any(line for line in content_lines if "->" in line):
                                issues.append({
                                    "severity": "warning",
                                    "type": "EmptyWhen",
                                    "line": i,
                                    "message": "Empty when expression",
                                    "suggestion": "Add cases or remove when"
                                })
                            break
                        next_idx += 1

        except Exception as e:
            logger.error(f"Kotlin analysis failed: {e}")
            issues.append({
                "severity": "error",
                "type": "AnalysisError",
                "line": 0,
                "message": f"Failed to analyze: {str(e)}"
            })

        return issues


# ===========================================================================
# MAIN IMPLEMENTATION FUNCTIONS
# ===========================================================================

def analyze_code_file_impl(file_path: str, language: str, deep_analysis: bool) -> str:
    """Implementation of analyze_code_file"""
    if not Path(file_path).exists():
        return json.dumps({"error": f"File not found: {file_path}"}, indent=2)

    lang = detect_language(file_path) if language == "auto" else SUPPORTED_LANGUAGES.get(language.lower())
    if not lang:
        return json.dumps({"error": "Could not detect language"}, indent=2)

    # Run language-specific analysis
    if lang.name == "python" and deep_analysis:
        issues = PythonBugDetector.analyze(file_path)
    elif lang.name == "java":
        issues = JavaBugDetector.analyze(file_path)
    elif lang.name == "kotlin":
        issues = KotlinBugDetector.analyze(file_path)
    else:
        issues = []

    errors = [i for i in issues if i.get('severity') == 'error']
    warnings = [i for i in issues if i.get('severity') == 'warning']
    info = [i for i in issues if i.get('severity') == 'info']

    return json.dumps({
        "file": file_path,
        "language": lang.name,
        "total_issues": len(issues),
        "errors": len(errors),
        "warnings": len(warnings),
        "info": len(info),
        "fixable": len([i for i in issues if 'fix_type' in i]),
        "issues": issues
    }, indent=2)


def fix_code_file_impl(file_path: str, auto_fix: bool, backup: bool, dry_run: bool) -> str:
    """Implementation of fix_code_file"""
    if not Path(file_path).exists():
        return json.dumps({"error": "File not found"}, indent=2)

    backup_path = None
    if backup and not dry_run:
        backup_path = f"{file_path}.backup"
        shutil.copy2(file_path, backup_path)

    lang = detect_language(file_path)
    if not lang:
        return json.dumps({"error": "Could not detect language"}, indent=2)

    fixes_applied = []

    if lang.name == "python":
        issues = PythonBugDetector.analyze(file_path)
        fixable = [i for i in issues if 'fix_type' in i]

        if not auto_fix:
            return json.dumps({
                "suggestions": [f"Line {i['line']}: {i['suggestion']}" for i in fixable]
            }, indent=2)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        for issue in sorted(fixable, key=lambda x: x['line'], reverse=True):
            idx = issue['line'] - 1
            if 0 <= idx < len(lines):
                orig = lines[idx]

                if issue['fix_type'] == 'bare_except':
                    lines[idx] = orig.replace('except:', 'except Exception as e:')
                    fixes_applied.append(f"Line {issue['line']}: Fixed bare except")

                elif issue['fix_type'] == 'identity_comparison':
                    lines[idx] = orig.replace(' is ', ' == ').replace(' is not ', ' != ')
                    fixes_applied.append(f"Line {issue['line']}: Fixed identity comparison")

        if fixes_applied and not dry_run:
            with open(file_path, 'w') as f:
                f.writelines(lines)

    # Run formatter
    formatted = False
    if lang.formatter_command and not dry_run and fixes_applied:
        try:
            cmd = [arg.format(file=file_path) for arg in lang.formatter_command]
            result = subprocess.run(cmd, capture_output=True, timeout=300)
            formatted = result.returncode == 0
        except:
            pass

    return json.dumps({
        "fixes_applied": len(fixes_applied),
        "details": fixes_applied,
        "backup_path": backup_path,
        "formatted": formatted,
        "dry_run": dry_run
    }, indent=2)


def suggest_improvements_impl(file_path: str, context: str, focus: str) -> str:
    """Implementation of suggest_improvements"""
    if not Path(file_path).exists():
        return json.dumps({"error": "File not found"}, indent=2)

    lang = detect_language(file_path)
    if not lang:
        return json.dumps({"error": "Could not detect language"}, indent=2)

    with open(file_path, 'r') as f:
        code = f.read()

    suggestions = []

    if lang.name == "python":
        if 'print(' in code:
            suggestions.append({
                "type": "best_practice",
                "message": "Use logging instead of print()",
                "reason": "Logging is more flexible for production",
                "priority": "medium"
            })

        if 'import *' in code:
            suggestions.append({
                "type": "best_practice",
                "message": "Avoid wildcard imports",
                "reason": "Makes code unclear and causes naming conflicts",
                "priority": "high"
            })

    elif lang.name == "java":
        if 'System.out.println' in code:
            suggestions.append({
                "type": "best_practice",
                "message": "Use proper logging framework instead of System.out.println",
                "reason": "Logging frameworks provide better control and production debugging",
                "priority": "medium"
            })

        if '+=' in code and 'String' in code:
            suggestions.append({
                "type": "performance",
                "message": "Consider using StringBuilder for string concatenation",
                "reason": "String concatenation with + creates many temporary objects",
                "priority": "medium"
            })

    elif lang.name == "kotlin":
        if '!!' in code:
            suggestions.append({
                "type": "best_practice",
                "message": "Minimize use of !! (force unwrap) operator",
                "reason": "Can throw NullPointerException at runtime",
                "priority": "high"
            })

        if 'mutableListOf' in code or 'mutableMapOf' in code:
            suggestions.append({
                "type": "best_practice",
                "message": "Prefer immutable collections when possible",
                "reason": "Immutable collections are safer and more thread-friendly",
                "priority": "low"
            })

    return json.dumps({
        "file": file_path,
        "language": lang.name,
        "focus": focus,
        "suggestions": suggestions
    }, indent=2)


def explain_code_impl(file_path: str, line_start: Optional[int], line_end: Optional[int], detail_level: str) -> str:
    """Implementation of explain_code"""
    return json.dumps({"message": "Feature coming soon"}, indent=2)


def generate_tests_impl(file_path: str, test_framework: str, coverage_target: str) -> str:
    """Implementation of generate_tests"""
    return json.dumps({"message": "Feature coming soon"}, indent=2)


def refactor_code_impl(file_path: str, refactor_type: str, target: str, preview: bool) -> str:
    """Implementation of refactor_code"""
    return json.dumps({"message": "Feature coming soon"}, indent=2)


def generate_code_impl(
        description: str,
        language: str = "python",
        style: str = "function",
        include_tests: bool = False,
        include_docstrings: bool = True,
        framework: str = "",
        output_file: str = ""
) -> str:
    """
    Generate code from natural language description using LLM.

    Args:
        description: What the code should do
        language: Programming language (python, javascript, typescript, java, kotlin, rust, go)
        style: Code style (function, class, module, script, api_endpoint)
        include_tests: Generate unit tests
        include_docstrings: Include documentation
        framework: Optional framework (fastapi, flask, react, spring, ktor, express)
        output_file: Optional file path to save generated code

    Returns:
        JSON with generated code and metadata
    """
    if not description or not description.strip():
        return json.dumps({
            "error": "Description is required",
            "status": "invalid_input"
        }, indent=2)

    # Build comprehensive prompt
    prompt = _build_code_generation_prompt(
        description=description,
        language=language,
        style=style,
        include_tests=include_tests,
        include_docstrings=include_docstrings,
        framework=framework
    )

    # Generate code using LLM
    try:
        from langchain_ollama import ChatOllama
        import os

        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        llm = ChatOllama(
            model=model_name,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
            temperature=0.3
        )

        response = llm.invoke(prompt)
        generated_code = response.content

        # Clean up markdown code blocks if present
        if "```" in generated_code:
            lines = generated_code.split('\n')
            code_lines = []
            in_code_block = False

            for line in lines:
                if line.startswith('```'):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    code_lines.append(line)

            if code_lines:
                generated_code = '\n'.join(code_lines)

        generated_code = generated_code.strip()
        status = "success"
        note = "Generated using LLM"

    except ImportError:
        # LLM not available, use template
        generated_code = _generate_template_code(description, language.lower(), style, framework)
        status = "template"
        note = "LLM not available. Install langchain-ollama for AI generation."
    except Exception as e:
        # LLM error, fallback to template
        generated_code = _generate_template_code(description, language.lower(), style, framework)
        status = "fallback_template"
        note = f"LLM error: {str(e)}, using template"

    # Build result
    result = {
        "description": description,
        "language": language,
        "style": style,
        "framework": framework if framework else "none",
        "generated_code": generated_code,
        "prompt_used": prompt,
        "includes_tests": include_tests,
        "includes_docs": include_docstrings,
        "status": status,
        "note": note
    }

    # Save to file if requested
    if output_file:
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                f.write(generated_code)

            result["saved_to"] = str(output_path)
            result["file_created"] = True

        except Exception as e:
            result["save_error"] = str(e)
            result["file_created"] = False

    return json.dumps(result, indent=2)


def _build_code_generation_prompt(
        description: str,
        language: str,
        style: str,
        include_tests: bool,
        include_docstrings: bool,
        framework: str
) -> str:
    """Build comprehensive prompt for code generation"""

    prompt_parts = [
        f"Generate {language} code following these specifications:",
        f"\n**Task:** {description}",
        f"\n**Style:** {style}",
        f"**Language:** {language}",
    ]

    if framework:
        prompt_parts.append(f"**Framework:** {framework}")

    # Language-specific best practices
    best_practices = {
        "python": [
            "- Follow PEP 8 style guide",
            "- Use type hints for all parameters and returns",
            "- No mutable default arguments (use None instead)",
            "- Use 'except Exception as e:' not bare except",
            "- Include comprehensive docstrings" if include_docstrings else "- Skip docstrings",
            "- Use f-strings for formatting",
            "- Handle edge cases (None, empty, invalid input)"
        ],
        "javascript": [
            "- Use modern ES6+ syntax",
            "- Use const/let, never var",
            "- Use arrow functions appropriately",
            "- Include JSDoc comments" if include_docstrings else "- Skip JSDoc",
            "- Handle promises with async/await",
            "- Use destructuring when appropriate",
            "- Validate inputs"
        ],
        "typescript": [
            "- Use strict TypeScript with proper types",
            "- Avoid 'any' type - use specific types or generics",
            "- Define interfaces for object shapes",
            "- Use enums for fixed sets of values",
            "- Include proper return types",
            "- Use readonly where applicable",
            "- Handle null/undefined explicitly"
        ],
        "java": [
            "- Follow Java naming conventions (CamelCase for classes, camelCase for methods)",
            "- Use proper access modifiers (private, protected, public)",
            "- Include JavaDoc comments for public methods" if include_docstrings else "- Skip JavaDoc",
            "- Use @Override annotation for overridden methods",
            "- Prefer immutable objects when possible",
            "- Use StringBuilder for string concatenation in loops",
            "- Handle exceptions properly with try-catch",
            "- Close resources with try-with-resources",
            "- Use Optional<T> for nullable return types"
        ],
        "kotlin": [
            "- Follow Kotlin conventions (camelCase, no semicolons)",
            "- Use val for immutable variables, var for mutable",
            "- Use data classes for POJOs",
            "- Prefer nullable types (?) over platform types (!!)",
            "- Use safe call operator (?.) and elvis operator (?:)",
            "- Include KDoc comments for public APIs" if include_docstrings else "- Skip KDoc",
            "- Use when instead of switch",
            "- Leverage extension functions when appropriate",
            "- Use scope functions (let, apply, run, also, with) idiomatically"
        ],
        "rust": [
            "- Follow Rust idioms and conventions",
            "- Use Result<T, E> for error handling",
            "- Implement proper error types",
            "- Use appropriate ownership patterns",
            "- Include comprehensive doc comments (///)" if include_docstrings else "- Skip doc comments",
            "- Use Option<T> for nullable values"
        ],
        "go": [
            "- Follow Go conventions and idioms",
            "- Use proper error handling (return error)",
            "- Include package-level documentation" if include_docstrings else "- Skip documentation",
            "- Use defer for cleanup",
            "- Keep functions focused and small",
            "- Use meaningful variable names"
        ]
    }

    language_lower = language.lower()
    if language_lower in best_practices:
        prompt_parts.append("\n**Best practices to follow:**")
        prompt_parts.extend(best_practices[language_lower])

    # Add test requirements
    if include_tests:
        test_frameworks = {
            "python": "pytest with fixtures and edge cases",
            "javascript": "Jest with describe/it blocks",
            "typescript": "Jest with proper typing",
            "java": "JUnit 5 with @Test annotations and assertions",
            "kotlin": "Kotest with StringSpec or FunSpec style",
            "rust": "Built-in #[test] with assertions",
            "go": "testing package with table-driven tests"
        }

        framework_instruction = test_frameworks.get(language_lower, "appropriate test framework")
        prompt_parts.append(f"\n**Include unit tests using {framework_instruction}:**")
        prompt_parts.append("- Test normal cases")
        prompt_parts.append("- Test edge cases (empty, null, invalid)")
        prompt_parts.append("- Test error conditions")

    # Framework-specific instructions
    if framework:
        framework_instructions = {
            "fastapi": [
                "- Use FastAPI decorators (@app.get, @app.post)",
                "- Use Pydantic models for request/response",
                "- Include proper status codes",
                "- Add error handling with HTTPException"
            ],
            "flask": [
                "- Use Flask decorators (@app.route)",
                "- Return JSON responses with jsonify",
                "- Include error handlers",
                "- Use request.get_json() for POST data"
            ],
            "react": [
                "- Use functional components with hooks",
                "- Use useState for state management",
                "- Use useEffect for side effects",
                "- Include prop validation with PropTypes"
            ],
            "express": [
                "- Use Express middleware patterns",
                "- Include proper error handling",
                "- Use async/await for route handlers",
                "- Return proper HTTP status codes"
            ],
            "spring": [
                "- Use Spring Boot annotations (@RestController, @Service, etc.)",
                "- Use @Autowired for dependency injection",
                "- Include @Valid for request validation",
                "- Return ResponseEntity with proper HTTP status",
                "- Use @ExceptionHandler for error handling"
            ],
            "ktor": [
                "- Use Ktor routing DSL",
                "- Use suspend functions for async operations",
                "- Include proper content negotiation",
                "- Use call.respond() for responses",
                "- Handle exceptions with status pages"
            ]
        }

        if framework.lower() in framework_instructions:
            prompt_parts.append(f"\n**{framework} specific:**")
            prompt_parts.extend(framework_instructions[framework.lower()])

    prompt_parts.append(
        "\n**IMPORTANT:** Return ONLY the code, properly formatted and ready to use. No markdown code blocks, no explanations before or after. Just the raw code.")

    return "\n".join(prompt_parts)


def _generate_template_code(description: str, language: str, style: str, framework: str = "") -> str:
    """Generate template code as fallback (when LLM not available)"""

    language = language.lower()
    style = style.lower()

    # Python templates
    if language == "python":
        if style == "function":
            return f'''def generated_function(param: str) -> str:
    """
    {description}

    Args:
        param: Input parameter

    Returns:
        Processed result

    Examples:
        >>> generated_function("test")
        "test"
    """
    # TODO: Implement actual logic
    return param
'''

        elif style == "class":
            return f'''class GeneratedClass:
    """
    {description}
    """

    def __init__(self, value: str):
        """
        Initialize the class.

        Args:
            value: Initial value
        """
        self.value = value

    def process(self, data: str) -> str:
        """
        Process the data.

        Args:
            data: Data to process

        Returns:
            Processed result
        """
        # TODO: Implement actual logic
        return f"{self.value}: {data}"
'''

        elif style == "api_endpoint" and framework == "fastapi":
            return f'''from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

class RequestModel(BaseModel):
    """Request data model"""
    name: str
    value: str

class ResponseModel(BaseModel):
    """Response data model"""
    result: str
    status: str

@app.post("/endpoint", response_model=ResponseModel)
async def generated_endpoint(request: RequestModel):
    """
    {description}

    Args:
        request: Request data

    Returns:
        Response with result

    Raises:
        HTTPException: If validation fails
    """
    try:
        # TODO: Implement actual logic
        result = f"Processed: {request.name} - {request.value}"
        return ResponseModel(result=result, status="success")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
'''

        elif style == "script":
            return f'''#!/usr/bin/env python3
"""
{description}
"""
import sys
from typing import List

def main(args: List[str]):
    """
    Main entry point.

    Args:
        args: Command line arguments
    """
    # TODO: Implement actual logic
    print(f"Running with args: {args}")

if __name__ == "__main__":
    main(sys.argv[1:])
'''

    # JavaScript/TypeScript templates
    elif language in ["javascript", "typescript"]:
        if "react" in framework.lower() or "component" in style:
            return f'''import React, {{ useState }} from 'react';

/**
 * {description}
 */
function GeneratedComponent() {{
    const [state, setState] = useState('');

    const handleChange = (e) => {{
        setState(e.target.value);
    }};

    return (
        <div>
            <h1>Generated Component</h1>
            <input 
                value={{state}} 
                onChange={{handleChange}}
                placeholder="Enter value"
            />
            <p>{{state}}</p>
        </div>
    );
}}

export default GeneratedComponent;
'''

        elif style == "function":
            return f'''/**
 * {description}
 * 
 * @param {{string}} param - Input parameter
 * @returns {{string}} Processed result
 */
function generatedFunction(param) {{
    // TODO: Implement actual logic
    return param;
}}

export {{ generatedFunction }};
'''

        elif style == "class":
            return f'''/**
 * {description}
 */
class GeneratedClass {{
    constructor(value) {{
        this.value = value;
    }}

    process(data) {{
        // TODO: Implement actual logic
        return `${{this.value}}: ${{data}}`;
    }}
}}

export {{ GeneratedClass }};
'''

    # Java templates
    elif language == "java":
        if style == "function":
            return f'''/**
 * {description}
 * 
 * @param param Input parameter
 * @return Processed result
 */
public static String generatedFunction(String param) {{
    // TODO: Implement actual logic
    return param;
}}
'''

        elif style == "class":
            return f'''/**
 * {description}
 */
public class GeneratedClass {{
    private String value;
    
    /**
     * Constructor
     * 
     * @param value Initial value
     */
    public GeneratedClass(String value) {{
        this.value = value;
    }}
    
    /**
     * Process data
     * 
     * @param data Data to process
     * @return Processed result
     */
    public String process(String data) {{
        // TODO: Implement actual logic
        return value + ": " + data;
    }}
    
    public String getValue() {{
        return value;
    }}
    
    public void setValue(String value) {{
        this.value = value;
    }}
}}
'''

        elif style == "api_endpoint" and framework == "spring":
            return f'''import org.springframework.web.bind.annotation.*;
import org.springframework.http.ResponseEntity;
import javax.validation.Valid;

/**
 * {description}
 */
@RestController
@RequestMapping("/api")
public class GeneratedController {{
    
    @PostMapping("/endpoint")
    public ResponseEntity<ResponseModel> generatedEndpoint(
            @Valid @RequestBody RequestModel request) {{
        
        // TODO: Implement actual logic
        String result = "Processed: " + request.getName() + " - " + request.getValue();
        
        ResponseModel response = new ResponseModel();
        response.setResult(result);
        response.setStatus("success");
        
        return ResponseEntity.ok(response);
    }}
}}

class RequestModel {{
    private String name;
    private String value;
    
    // Getters and setters
    public String getName() {{ return name; }}
    public void setName(String name) {{ this.name = name; }}
    public String getValue() {{ return value; }}
    public void setValue(String value) {{ this.value = value; }}
}}

class ResponseModel {{
    private String result;
    private String status;
    
    // Getters and setters
    public String getResult() {{ return result; }}
    public void setResult(String result) {{ this.result = result; }}
    public String getStatus() {{ return status; }}
    public void setStatus(String status) {{ this.status = status; }}
}}
'''

    # Kotlin templates
    elif language == "kotlin":
        if style == "function":
            return f'''/**
 * {description}
 * 
 * @param param Input parameter
 * @return Processed result
 */
fun generatedFunction(param: String): String {{
    // TODO: Implement actual logic
    return param
}}
'''

        elif style == "class":
            return f'''/**
 * {description}
 * 
 * @property value Initial value
 */
class GeneratedClass(private var value: String) {{
    
    /**
     * Process data
     * 
     * @param data Data to process
     * @return Processed result
     */
    fun process(data: String): String {{
        // TODO: Implement actual logic
        return "$value: $data"
    }}
}}
'''

        elif style == "api_endpoint" and framework == "ktor":
            return f'''import io.ktor.server.application.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*
import io.ktor.http.*

/**
 * {description}
 */
fun Application.configureRouting() {{
    routing {{
        post("/api/endpoint") {{
            val request = call.receive<RequestModel>()
            
            // TODO: Implement actual logic
            val result = "Processed: ${{request.name}} - ${{request.value}}"
            
            val response = ResponseModel(
                result = result,
                status = "success"
            )
            
            call.respond(HttpStatusCode.OK, response)
        }}
    }}
}}

data class RequestModel(
    val name: String,
    val value: String
)

data class ResponseModel(
    val result: String,
    val status: String
)
'''

    # Generic fallback
    else:
        return f'''// {description}
// Language: {language}
// Style: {style}
// TODO: Implement in {language} using {style} style
'''


def _extract_project_description_from_docs(project_root: Path) -> dict:
    """
    Extract project description from README and documentation files.

    Handles common README structures:
    - # Title
      Description immediately after title
    - ## Purpose / ## About / ## Description sections

    Returns:
        dict with 'description' and 'purpose' if found
    """
    result = {
        "description": "",
        "purpose": ""
    }

    # Files to check (in priority order)
    doc_files = [
        'README.md',
        'README.txt',
        'README',
        'readme.md',
        'readme.txt',
        'DESCRIPTION.md',
        'ABOUT.md',
        'docs/README.md',
        'doc/README.md'
    ]

    content = ""

    # Find and read first available doc file
    for doc_file in doc_files:
        file_path = project_root / doc_file
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    break
            except:
                continue

    if not content:
        return result

    lines = content.split('\n')

    # Strategy 1: Look for description after main title
    # Example:
    #   # Project Name
    #   Description here.
    description = None
    title_found = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Skip badges (lines starting with [![ or <img)
        if stripped.startswith('[![') or stripped.startswith('<img'):
            continue

        # Main title (single # at start, not ##)
        if stripped.startswith('# ') and not stripped.startswith('## '):
            title_found = True
            continue

        # If we're after title, get first meaningful line
        if title_found and stripped:
            # Skip if it's another header or badge
            if stripped.startswith('#') or stripped.startswith('!') or stripped.startswith('<'):
                continue

            # This is likely the description
            description = stripped
            break

    # Clean up markdown formatting from description
    if description:
        import re
        # Remove [text](url) -> text
        description = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', description)
        # Remove **bold** -> bold
        description = re.sub(r'\*\*([^\*]+)\*\*', r'\1', description)
        # Remove *italic* -> italic
        description = re.sub(r'\*([^\*]+)\*', r'\1', description)
        description = description.strip()

        result['description'] = description

    # Strategy 2: Look for explicit purpose/description sections
    # Common patterns: ## Purpose, ## Description, ## About, ## What is X
    import re

    purpose_keywords = [
        'purpose', 'description', 'about', 'overview',
        'what is', 'introduction', 'summary'
    ]

    purpose = None

    for i, line in enumerate(lines):
        stripped = line.strip().lower()

        # Check if this is a section header with purpose keywords
        if stripped.startswith('##') or stripped.startswith('###'):
            header_text = stripped.lstrip('#').strip()

            if any(keyword in header_text for keyword in purpose_keywords):
                # Get the next meaningful line(s)
                for j in range(i + 1, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line and not next_line.startswith('#'):
                        # Found content under the purpose section
                        purpose = next_line
                        break

                if purpose:
                    break

    # Clean up purpose
    if purpose:
        purpose = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', purpose)
        purpose = re.sub(r'\*\*([^\*]+)\*\*', r'\1', purpose)
        purpose = re.sub(r'\*([^\*]+)\*', r'\1', purpose)
        purpose = purpose.strip()

        result['purpose'] = purpose

    # If we found description but not purpose, use description as purpose
    if description and not purpose:
        result['purpose'] = description

    return result


def ensure_no_none(obj):
    """
    Recursively replace None with appropriate defaults to prevent Jinja2 errors.

    This is critical for GGUF models which use Jinja2 templates for chat formatting.
    If ANY value in the result is None, Jinja2 will crash with:
    "TypeError: can only concatenate str (not 'NoneType') to str"

    Args:
        obj: Any object (dict, list, str, None, etc.)

    Returns:
        Same structure with None replaced by ""
    """
    if isinstance(obj, dict):
        return {k: ensure_no_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_no_none(item) for item in obj]
    elif obj is None:
        return ""  # Replace None with empty string
    else:
        return obj

def analyze_project_impl(
        project_path: str = ".",
        include_dependencies: bool = True,
        include_structure: bool = True,
        max_depth: int = 6
) -> str:
    """
    Analyze project structure and tech stack.

    Args:
        project_path: Root path of project (default: current directory)
        include_dependencies: Parse requirements.txt, package.json, etc.
        include_structure: Include directory structure
        max_depth: Maximum directory depth to scan

    Returns:
        JSON with project analysis including intro and architecture description
    """
    project_root = Path(project_path).resolve()

    if not project_root.exists():
        return json.dumps({
            "error": f"Project path not found: {project_path}",
            "status": "not_found"
        }, indent=2)

    analysis = {
        "project_root": str(project_root),
        "project_name": project_root.name,
        "project_intro": "",
        "project_purpose": "",
        "architecture": {
            "type": "unknown",
            "patterns": [],
            "layers": {},
            "description": ""
        },
        "languages": {},
        "frameworks": [],
        "dependencies": {},
        "file_counts": {},
        "structure": {},
        "tech_stack": []
    }

    # Scan project files
    file_extensions = defaultdict(int)
    total_lines = defaultdict(int)
    all_dirs = set()
    all_files = []

    try:
        for root, dirs, files in os.walk(project_root):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in {
                '.git', '.venv', 'venv', 'node_modules', '__pycache__',
                '.pytest_cache', '.mypy_cache', 'dist', 'build', '.idea',
                'logs', 'tmp', 'temp'
            }]

            # Check depth
            depth = len(Path(root).relative_to(project_root).parts)
            if depth > max_depth:
                continue

            rel_root = Path(root).relative_to(project_root)
            all_dirs.add(str(rel_root))

            for file in files:
                file_path = Path(root) / file
                ext = file_path.suffix.lower()

                if ext:
                    file_extensions[ext] += 1
                    all_files.append(str(rel_root / file))

                    # Count lines for code files
                    if ext in {'.py', '.js', '.jsx', '.ts', '.tsx', '.rs', '.go', '.java', '.kt', '.c', '.cpp', '.h'}:
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                lines = len(f.readlines())
                                total_lines[ext] += lines
                        except:
                            pass

    except Exception as e:
        return json.dumps({
            "error": f"Failed to scan project: {str(e)}",
            "status": "scan_failed"
        }, indent=2)

    # Determine primary languages
    language_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript (React)',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript (React)',
        '.rs': 'Rust',
        '.go': 'Go',
        '.java': 'Java',
        '.kt': 'Kotlin',
        '.c': 'C',
        '.cpp': 'C++',
        '.rb': 'Ruby',
        '.php': 'PHP'
    }

    for ext, count in sorted(file_extensions.items(), key=lambda x: x[1], reverse=True):
        if ext in language_map:
            lang_name = language_map[ext]
            analysis['languages'][lang_name] = {
                "files": count,
                "lines": total_lines.get(ext, 0),
                "extension": ext
            }

    analysis['file_counts'] = dict(sorted(file_extensions.items(), key=lambda x: x[1], reverse=True)[:20])

    # Detect dependencies and frameworks
    if include_dependencies:
        # Python dependencies
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            python_deps = _parse_requirements(req_file)
            analysis['dependencies']['python'] = python_deps
            analysis['frameworks'].extend(_detect_python_frameworks(python_deps))

        # Node.js dependencies
        package_file = project_root / "package.json"
        if package_file.exists():
            node_deps = _parse_package_json(package_file)
            analysis['dependencies']['node'] = node_deps
            analysis['frameworks'].extend(_detect_node_frameworks(node_deps))

        # Kotlin/Java dependencies (Gradle)
        gradle_deps = _parse_gradle_dependencies(project_root)
        if gradle_deps:
            analysis['dependencies']['gradle'] = gradle_deps
            analysis['frameworks'].extend(_detect_kotlin_java_frameworks(gradle_deps))

        # Kotlin/Java dependencies (Maven)
        maven_deps = _parse_maven_dependencies(project_root)
        if maven_deps:
            analysis['dependencies']['maven'] = maven_deps
            analysis['frameworks'].extend(_detect_kotlin_java_frameworks(maven_deps))

    # ========================================================================
    # ARCHITECTURE ANALYSIS
    # ========================================================================

    dir_names = {Path(d).name.lower() for d in all_dirs if d != '.'}

    # Detect MCP architecture
    if 'servers' in dir_names and 'tools' in dir_names and 'client' in dir_names:
        analysis['architecture']['type'] = 'MCP (Model Context Protocol)'
        analysis['architecture']['patterns'] = [
            'Agent-based architecture',
            'Tool-based extensibility',
            'Server-client pattern'
        ]

        # Count servers and tools
        servers_dir = project_root / 'servers'
        tools_dir = project_root / 'tools'

        server_count = len([d for d in servers_dir.iterdir() if
                            d.is_dir() and not d.name.startswith('.')]) if servers_dir.exists() else 0
        tool_count = len(
            [d for d in tools_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]) if tools_dir.exists() else 0

        analysis['architecture']['layers'] = {
            'client': 'Agent orchestration layer - manages LLM interactions and tool routing',
            'servers': f'MCP server layer - {server_count} servers exposing tools via stdio',
            'tools': f'Tool implementation layer - {tool_count} tool modules with business logic'
        }

        analysis['architecture']['description'] = (
            f"MCP-based multi-agent system with {server_count} specialized servers. "
            f"The client layer orchestrates LLM tool calls through LangGraph, "
            f"routing requests to appropriate MCP servers via stdio transport. "
            f"Each server exposes domain-specific tools (code analysis, knowledge base, etc.) "
            f"with implementations separated into the tools layer for modularity and testability."
        )

    # Detect Android/Kotlin architecture
    elif any('src/main/kotlin' in str(d) or 'src/main/java' in str(d) for d in all_dirs):

        # Check if it's actually Android (not just Kotlin)
        is_android = False

        # Simple check: AndroidManifest.xml exists
        if (project_root / 'AndroidManifest.xml').exists():
            is_android = True
        elif (project_root / 'app' / 'src' / 'main' / 'AndroidManifest.xml').exists():
            is_android = True
        elif (project_root / 'src' / 'main' / 'AndroidManifest.xml').exists():
            is_android = True

        # Check build.gradle for android plugin
        if not is_android:
            for gradle_file in ['build.gradle', 'build.gradle.kts', 'app/build.gradle', 'app/build.gradle.kts']:
                gradle_path = project_root / gradle_file
                if gradle_path.exists():
                    try:
                        content = gradle_path.read_text(encoding='utf-8', errors='ignore')
                        if 'com.android.application' in content or 'com.android.library' in content:
                            is_android = True
                            break
                    except:
                        pass

        # Set architecture type based on what we found
        if is_android:
            analysis['architecture']['type'] = 'Android/Kotlin Application'
        else:
            analysis['architecture']['type'] = 'Kotlin Application'

        analysis['architecture']['patterns'] = []

        # Only do Android-specific detection if it's actually Android
        if is_android:
            # Detect patterns
            has_ui = any(name in dir_names for name in ['ui', 'view', 'activity', 'fragment'])
            has_data = any(name in dir_names for name in ['data', 'repository', 'dao', 'database'])
            has_domain = any(name in dir_names for name in ['domain', 'usecase', 'model'])
            has_di = any(name in dir_names for name in ['di', 'injection', 'module', 'koin', 'hilt'])

            layers = {}

            if has_ui and has_data and has_domain:
                analysis['architecture']['patterns'].append('Clean Architecture')
                layers = {
                    'presentation': 'UI layer with Activities, Fragments, and ViewModels',
                    'domain': 'Business logic layer with use cases and domain models',
                    'data': 'Data layer with repositories and data sources'
                }
            elif has_ui and has_data:
                analysis['architecture']['patterns'].append('MVVM (Model-View-ViewModel)')
                layers = {
                    'presentation': 'Views and ViewModels for UI logic',
                    'data': 'Data repositories and sources'
                }

            if has_di:
                analysis['architecture']['patterns'].append('Dependency Injection')

            # Check for Jetpack Compose
            if any('compose' in str(f).lower() for f in all_files):
                analysis['architecture']['patterns'].append('Jetpack Compose (declarative UI)')

            # Check for Coroutines
            if any('coroutine' in str(f).lower() for f in all_files):
                analysis['architecture']['patterns'].append('Kotlin Coroutines (async)')

            analysis['architecture']['layers'] = layers

            pattern_desc = ' and '.join(analysis['architecture']['patterns']) if analysis['architecture'][
                'patterns'] else 'standard Android patterns'
            analysis['architecture']['description'] = (
                f"Android application using {pattern_desc}. "
                f"Follows Android best practices with clear separation of concerns."
            )
        else:
            # It's Kotlin but NOT Android
            analysis['architecture']['patterns'] = ['Kotlin/JVM application']
            analysis['architecture']['description'] = (
                "Kotlin application (not Android). "
                "Standard JVM or server-side Kotlin project."
            )

    # Detect microservices
    elif 'services' in dir_names or 'microservices' in dir_names:
        analysis['architecture']['type'] = 'Microservices'
        analysis['architecture']['patterns'] = ['Microservices architecture', 'Service-oriented']

        services_dir = project_root / 'services' if (
                    project_root / 'services').exists() else project_root / 'microservices'
        service_names = []
        if services_dir.exists():
            service_names = [d.name for d in services_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]

        analysis['architecture']['layers'] = {
            'services': f'{len(service_names)} independent services',
            'api': 'Service communication layer (REST/gRPC)',
        }

        analysis['architecture']['description'] = (
            f"Microservices architecture with {len(service_names)} independently deployable services. "
            f"Each service owns its domain and communicates via APIs. "
            f"Promotes scalability, fault isolation, and independent deployment."
        )

    # Detect MVC/web framework
    elif all(d in dir_names for d in ['models', 'views', 'controllers']):
        analysis['architecture']['type'] = 'MVC (Model-View-Controller)'
        analysis['architecture']['patterns'] = ['MVC pattern', 'Web application']

        analysis['architecture']['layers'] = {
            'models': 'Data models and business logic',
            'views': 'UI templates and presentation',
            'controllers': 'Request handlers and routing logic'
        }

        analysis['architecture']['description'] = (
            "Classic MVC web application with strict separation of concerns. "
            "Models handle data/business logic, views render UI, controllers "
            "orchestrate request/response flow."
        )

    # Detect layered architecture
    elif any(d in dir_names for d in ['api', 'service', 'repository', 'domain']):
        analysis['architecture']['type'] = 'Layered Architecture'
        analysis['architecture']['patterns'] = ['Layered architecture', 'N-tier']

        layers = {}
        if 'api' in dir_names or 'controllers' in dir_names:
            layers['api'] = 'API/presentation layer - HTTP endpoints and request handling'
        if 'service' in dir_names or 'business' in dir_names:
            layers['service'] = 'Business logic layer - core application logic'
        if 'repository' in dir_names or 'data' in dir_names or 'persistence' in dir_names:
            layers['data'] = 'Data access layer - database operations'
        if 'domain' in dir_names or 'models' in dir_names:
            layers['domain'] = 'Domain layer - business entities and rules'

        analysis['architecture']['layers'] = layers

        analysis['architecture']['description'] = (
            f"Layered architecture with {len(layers)} distinct layers. "
            f"Each layer has specific responsibilities and depends only on layers below it, "
            f"promoting separation of concerns and maintainability."
        )

    else:
        # Generic/monolithic
        analysis['architecture']['type'] = 'Monolithic'
        analysis['architecture']['description'] = (
            "Project follows a straightforward structure without clear architectural patterns. "
            "May benefit from organizing into defined layers as complexity grows."
        )

    # ========================================================================
    # PROJECT INTRO
    # ========================================================================

    # Generate intro based on what we found
    # Extract description from README/docs
    doc_info = _extract_project_description_from_docs(project_root)

    # Use README content if available
    if doc_info.get('description') or doc_info.get('purpose'):
        analysis['project_intro'] = doc_info.get('description') or doc_info.get('purpose') or "Project"
        analysis['project_purpose'] = doc_info.get('purpose') or doc_info.get('description') or "Not specified."
    else:
        # Fallback: Generate intro based on what we found
        primary_lang = list(analysis['languages'].keys())[0] if analysis['languages'] else "Unknown"
        total_files = sum(file_extensions.values())
        total_code_lines = sum(total_lines.values())

        intro_parts = [f"{analysis['project_name']} is a"]

        # Add architecture type
        if analysis['architecture']['type'] != 'unknown':
            intro_parts.append(f"{analysis['architecture']['type']}")

        # Add primary language
        if primary_lang != "Unknown":
            intro_parts.append(f"written primarily in {primary_lang}")

        # Add framework info
        if analysis['frameworks']:
            main_frameworks = analysis['frameworks'][:3]  # Top 3
            intro_parts.append(f"using {', '.join(main_frameworks)}")

        # Add scale
        intro_parts.append(f"with {total_files:,} files")
        if total_code_lines > 0:
            intro_parts.append(f"and ~{total_code_lines:,} lines of code")

        analysis['project_intro'] = ' '.join(intro_parts) + "."
        analysis['project_purpose'] = "Not specified."

    # ========================================================================
    # Build tech stack summary
    # ========================================================================

    tech_stack = []

    # Add languages
    for lang, info in analysis['languages'].items():
        tech_stack.append(f"{lang} ({info['files']} files, {info['lines']:,} lines)")

    # Add frameworks
    tech_stack.extend(analysis['frameworks'])

    # Add key dependencies
    for lang_type, deps in analysis['dependencies'].items():
        key_deps = deps[:5] if isinstance(deps, list) else list(deps.keys())[:5]
        if key_deps:
            tech_stack.append(f"{lang_type.title()} packages: {', '.join(key_deps)}")

    analysis['tech_stack'] = tech_stack

    # Project structure (simplified tree)
    if include_structure:
        analysis['structure'] = _build_directory_tree(project_root, max_depth=2)

    # CRITICAL: Replace ALL None values with empty strings
    # This prevents Jinja2 template errors in GGUF models
    analysis = ensure_no_none(analysis)

    # Double-check critical fields have non-None values
    if not analysis.get('project_intro'):
        analysis['project_intro'] = f"{analysis['project_name']} project"

    if not analysis.get('project_purpose'):
        analysis['project_purpose'] = "Not specified."

    if not analysis['architecture'].get('description'):
        analysis['architecture']['description'] = "Architecture not clearly identified."

    if not analysis['architecture'].get('type'):
        analysis['architecture']['type'] = "unknown"

    return json.dumps(analysis, indent=2)

def get_project_dependencies_impl(project_path: str = ".", dep_type: str = "all") -> str:
    """
    Get detailed project dependencies.

    Args:
        project_path: Root path of project
        dep_type: Type of dependencies ("python", "node", "rust", "all")

    Returns:
        JSON with dependencies
    """
    project_root = Path(project_path).resolve()

    dependencies = {}

    if dep_type in ["python", "all"]:
        req_file = project_root / "requirements.txt"
        if req_file.exists():
            dependencies['python'] = _parse_requirements_detailed(req_file)

    if dep_type in ["node", "all"]:
        package_file = project_root / "package.json"
        if package_file.exists():
            dependencies['node'] = _parse_package_json_detailed(package_file)

    return json.dumps({
        "project_root": str(project_root),
        "dependencies": dependencies
    }, indent=2)


def scan_project_structure_impl(project_path: str = ".", max_depth: int = 3) -> str:
    """
    Get detailed project directory structure.

    Args:
        project_path: Root path of project
        max_depth: Maximum depth to scan

    Returns:
        JSON with directory tree
    """
    project_root = Path(project_path).resolve()

    structure = _build_directory_tree(project_root, max_depth)

    return json.dumps({
        "project_root": str(project_root),
        "structure": structure
    }, indent=2)


# ===========================================================================
# HELPER FUNCTIONS FOR PROJECT ANALYSIS
# ===========================================================================

def _parse_requirements(req_file: Path) -> List[str]:
    """Parse requirements.txt and return package names"""
    packages = []
    try:
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (before ==, >=, etc.)
                    match = re.match(r'^([a-zA-Z0-9\-_]+)', line)
                    if match:
                        packages.append(match.group(1))
    except:
        pass
    return packages


def _parse_requirements_detailed(req_file: Path) -> Dict[str, str]:
    """Parse requirements.txt with versions"""
    packages = {}
    try:
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        packages[name.strip()] = version.strip()
                    elif '>=' in line:
                        name, version = line.split('>=', 1)
                        packages[name.strip()] = f">={version.strip()}"
                    else:
                        packages[line] = "latest"
    except:
        pass
    return packages


def _parse_package_json(package_file: Path) -> List[str]:
    """Parse package.json and return package names"""
    try:
        with open(package_file, 'r') as f:
            data = json.load(f)
            deps = list(data.get('dependencies', {}).keys())
            dev_deps = list(data.get('devDependencies', {}).keys())
            return deps + dev_deps
    except:
        return []


def _parse_package_json_detailed(package_file: Path) -> Dict[str, Any]:
    """Parse package.json with full details"""
    try:
        with open(package_file, 'r') as f:
            data = json.load(f)
            return {
                "dependencies": data.get('dependencies', {}),
                "devDependencies": data.get('devDependencies', {}),
                "scripts": data.get('scripts', {})
            }
    except:
        return {}


def _detect_python_frameworks(packages: List[str]) -> List[str]:
    """Detect Python frameworks from package list"""
    frameworks = []
    framework_map = {
        'fastapi': 'FastAPI',
        'flask': 'Flask',
        'django': 'Django',
        'pytorch': 'PyTorch',
        'tensorflow': 'TensorFlow',
        'langchain': 'LangChain',
        'langchain-ollama': 'LangChain (Ollama)',
        'mcp': 'Model Context Protocol',
        'mcp-use': 'MCP Use',
        'sqlalchemy': 'SQLAlchemy',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'scikit-learn': 'scikit-learn',
        'lancedb': 'LanceDB',
        'sentence-transformers': 'Sentence Transformers',
        'ollama': 'Ollama'
    }

    for pkg in packages:
        pkg_lower = pkg.lower()
        if pkg_lower in framework_map:
            frameworks.append(framework_map[pkg_lower])

    return frameworks


def _detect_node_frameworks(packages: List[str]) -> List[str]:
    """Detect Node.js frameworks from package list"""
    frameworks = []
    framework_map = {
        'react': 'React',
        'vue': 'Vue.js',
        'next': 'Next.js',
        'express': 'Express',
        'nestjs': 'NestJS',
        'typescript': 'TypeScript'
    }

    for pkg in packages:
        pkg_lower = pkg.lower()
        if pkg_lower in framework_map:
            frameworks.append(framework_map[pkg_lower])

    return frameworks


def _build_directory_tree(root_path: Path, max_depth: int, current_depth: int = 0) -> Dict:
    """Build a nested directory tree"""
    if current_depth >= max_depth:
        return {}

    tree = {}

    try:
        items = sorted(root_path.iterdir(), key=lambda x: (not x.is_dir(), x.name))

        for item in items:
            # Skip hidden and ignored directories
            if item.name.startswith('.') or item.name in {
                'node_modules', '__pycache__', 'venv', '.venv', 'logs',
                'tmp', 'temp', 'dist', 'build', '.idea'
            }:
                continue

            if item.is_dir():
                subtree = _build_directory_tree(item, max_depth, current_depth + 1)
                if subtree or current_depth < max_depth - 1:
                    tree[f"{item.name}/"] = subtree
            else:
                tree[item.name] = None  # File (no children)
    except:
        pass

    return tree

def _parse_gradle_dependencies(project_root: Path) -> List[str]:
    """
    Parse Gradle build files to extract dependencies.

    Supports:
    - build.gradle (Groovy DSL)
    - build.gradle.kts (Kotlin DSL)
    - Multiple module projects

    Returns:
        List of dependency strings (e.g., "androidx.compose.ui:ui")
    """
    dependencies = []

    # Gradle files to check
    gradle_files = [
        'build.gradle',
        'build.gradle.kts',
        'app/build.gradle',
        'app/build.gradle.kts',
        'settings.gradle',
        'settings.gradle.kts'
    ]

    # Also find all build.gradle files in subdirectories
    try:
        for gradle_file in project_root.rglob('build.gradle*'):
            if 'build' not in gradle_file.parts and '.gradle' not in gradle_file.parts:
                gradle_files.append(str(gradle_file.relative_to(project_root)))
    except:
        pass

    for gradle_file_path in gradle_files:
        gradle_file = project_root / gradle_file_path
        if not gradle_file.exists():
            continue

        try:
            with open(gradle_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Pattern 1: implementation("group:artifact:version")
            # Pattern 2: implementation "group:artifact:version"
            # Pattern 3: implementation 'group:artifact:version'

            import re

            # Match various dependency declaration formats
            patterns = [
                r'implementation[(\s]+["\']([^"\']+)["\']',
                r'api[(\s]+["\']([^"\']+)["\']',
                r'compileOnly[(\s]+["\']([^"\']+)["\']',
                r'runtimeOnly[(\s]+["\']([^"\']+)["\']',
                r'testImplementation[(\s]+["\']([^"\']+)["\']',
                r'androidTestImplementation[(\s]+["\']([^"\']+)["\']',
                r'kapt[(\s]+["\']([^"\']+)["\']',
                r'ksp[(\s]+["\']([^"\']+)["\']',
            ]

            for pattern in patterns:
                matches = re.findall(pattern, content)
                dependencies.extend(matches)

            # Also look for plugins (these indicate frameworks)
            plugin_patterns = [
                r'id[(\s]+["\']([^"\']+)["\']',
                r'apply\s+plugin:\s*["\']([^"\']+)["\']',
            ]

            for pattern in plugin_patterns:
                matches = re.findall(pattern, content)
                dependencies.extend([f"plugin:{m}" for m in matches])

        except Exception as e:
            logger.debug(f"Failed to parse {gradle_file}: {e}")
            continue

    # Remove duplicates and clean up
    dependencies = list(set(dependencies))
    return dependencies


def _parse_maven_dependencies(project_root: Path) -> List[str]:
    """
    Parse Maven pom.xml files to extract dependencies.

    Returns:
        List of dependency strings (e.g., "org.springframework.boot:spring-boot-starter-web")
    """
    dependencies = []

    # Find all pom.xml files
    pom_files = ['pom.xml']

    try:
        for pom_file in project_root.rglob('pom.xml'):
            if 'target' not in pom_file.parts:
                pom_files.append(str(pom_file.relative_to(project_root)))
    except:
        pass

    for pom_file_path in pom_files:
        pom_file = project_root / pom_file_path
        if not pom_file.exists():
            continue

        try:
            tree = ET.parse(pom_file)
            root = tree.getroot()

            # Handle XML namespaces
            ns = {'m': 'http://maven.apache.org/POM/4.0.0'}

            # Try without namespace first (some pom.xml don't have it)
            deps = root.findall('.//dependency')
            if not deps:
                deps = root.findall('.//m:dependency', ns)

            for dep in deps:
                group_id = dep.find('groupId') or dep.find('m:groupId', ns)
                artifact_id = dep.find('artifactId') or dep.find('m:artifactId', ns)

                if group_id is not None and artifact_id is not None:
                    dep_string = f"{group_id.text}:{artifact_id.text}"
                    dependencies.append(dep_string)

        except Exception as e:
            logger.debug(f"Failed to parse {pom_file}: {e}")
            continue

    # Remove duplicates
    dependencies = list(set(dependencies))
    return dependencies


def _detect_kotlin_java_frameworks(dependencies: List[str]) -> List[str]:
    """
    Detect Kotlin/Java frameworks from Gradle/Maven dependencies.

    Args:
        dependencies: List of dependency strings (e.g., "androidx.compose.ui:ui")

    Returns:
        List of detected framework names
    """
    frameworks = []
    seen = set()

    # Framework detection patterns
    framework_patterns = {
        # Android
        'androidx.compose': 'Jetpack Compose',
        'androidx.navigation': 'Android Navigation',
        'androidx.room': 'Room Database',
        'androidx.lifecycle': 'Android Lifecycle',
        'androidx.work': 'WorkManager',
        'androidx.paging': 'Paging 3',
        'androidx.datastore': 'DataStore',
        'android.arch.lifecycle': 'Android Architecture Components',
        'com.google.android.material': 'Material Design Components',

        # Dependency Injection
        'com.google.dagger': 'Dagger',
        'dagger.hilt': 'Hilt',
        'org.koin': 'Koin',
        'io.insert-koin': 'Koin',
        'javax.inject': 'Javax Inject',

        # Networking
        'com.squareup.retrofit2': 'Retrofit',
        'com.squareup.okhttp3': 'OkHttp',
        'io.ktor': 'Ktor',
        'org.apache.httpcomponents': 'Apache HttpClient',

        # JSON/Serialization
        'com.google.code.gson': 'Gson',
        'com.squareup.moshi': 'Moshi',
        'com.fasterxml.jackson': 'Jackson',
        'org.jetbrains.kotlinx:kotlinx-serialization': 'Kotlin Serialization',

        # Coroutines/Async
        'org.jetbrains.kotlinx:kotlinx-coroutines': 'Kotlin Coroutines',
        'io.reactivex': 'RxJava',
        'io.projectreactor': 'Project Reactor',

        # Spring Framework
        'org.springframework.boot': 'Spring Boot',
        'org.springframework': 'Spring Framework',
        'org.springframework.data': 'Spring Data',
        'org.springframework.security': 'Spring Security',
        'org.springframework.cloud': 'Spring Cloud',

        # Database/ORM
        'org.hibernate': 'Hibernate',
        'org.mybatis': 'MyBatis',
        'com.querydsl': 'QueryDSL',
        'org.jooq': 'jOOQ',
        'com.h2database': 'H2 Database',
        'org.postgresql': 'PostgreSQL Driver',
        'mysql:mysql-connector': 'MySQL Driver',
        'org.mongodb': 'MongoDB Driver',

        # Testing
        'org.junit.jupiter': 'JUnit 5',
        'junit:junit': 'JUnit 4',
        'org.mockito': 'Mockito',
        'io.mockk': 'MockK',
        'io.kotest': 'Kotest',
        'org.assertj': 'AssertJ',
        'com.google.truth': 'Truth',
        'org.robolectric': 'Robolectric',
        'androidx.test.espresso': 'Espresso',

        # Logging
        'org.slf4j': 'SLF4J',
        'ch.qos.logback': 'Logback',
        'org.apache.logging.log4j': 'Log4j',

        # Image Loading
        'com.github.bumptech.glide': 'Glide',
        'io.coil-kt': 'Coil',
        'com.squareup.picasso': 'Picasso',

        # Build/Plugins (from plugin: prefix)
        'plugin:com.android.application': 'Android Application',
        'plugin:com.android.library': 'Android Library',
        'plugin:org.jetbrains.kotlin.android': 'Kotlin Android',
        'plugin:org.jetbrains.kotlin.jvm': 'Kotlin JVM',
        'plugin:org.jetbrains.kotlin.plugin.serialization': 'Kotlin Serialization',
        'plugin:com.google.dagger.hilt.android': 'Hilt',
        'plugin:org.springframework.boot': 'Spring Boot',
        'plugin:io.ktor.plugin': 'Ktor',
        'plugin:kotlin-kapt': 'Kapt (Kotlin Annotation Processing)',
        'plugin:com.google.devtools.ksp': 'KSP (Kotlin Symbol Processing)',
    }

    for dep in dependencies:
        dep_lower = dep.lower()

        # Check each pattern
        for pattern, framework_name in framework_patterns.items():
            if pattern.lower() in dep_lower:
                if framework_name not in seen:
                    frameworks.append(framework_name)
                    seen.add(framework_name)
                break  # Only match once per dependency

    return frameworks