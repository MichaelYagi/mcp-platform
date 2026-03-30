"""
Code Analysis Core Logic
========================
Located at: tools/code_assistant/analyzer.py

This module contains the actual implementation of code analysis.
The MCP server (servers/code_assistant/server.py) calls these functions.
"""

import ast
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


# ============================================================================
# LANGUAGE CONFIGURATION
# ============================================================================

@dataclass
class LanguageConfig:
    """Configuration for a programming language"""
    name: str
    extensions: List[str]
    comment_single: str
    comment_multi_start: Optional[str]
    comment_multi_end: Optional[str]
    linter_command: Optional[List[str]]
    formatter_command: Optional[List[str]]


SUPPORTED_LANGUAGES = {
    "python": LanguageConfig(
        name="python",
        extensions=[".py"],
        comment_single="#",
        comment_multi_start='"""',
        comment_multi_end='"""',
        linter_command=["ruff", "check", "{file}"],
        formatter_command=["black", "{file}"]
    ),
    "javascript": LanguageConfig(
        name="javascript",
        extensions=[".js", ".jsx"],
        comment_single="//",
        comment_multi_start="/*",
        comment_multi_end="*/",
        linter_command=["eslint", "{file}"],
        formatter_command=["prettier", "--write", "{file}"]
    ),
    "typescript": LanguageConfig(
        name="typescript",
        extensions=[".ts", ".tsx"],
        comment_single="//",
        comment_multi_start="/*",
        comment_multi_end="*/",
        linter_command=["eslint", "{file}"],
        formatter_command=["prettier", "--write", "{file}"]
    ),
    "java": LanguageConfig(
        name="java",
        extensions=[".java"],
        comment_single="//",
        comment_multi_start="/*",
        comment_multi_end="*/",
        linter_command=["checkstyle", "-c", "/google_checks.xml", "{file}"],
        formatter_command=["google-java-format", "--replace", "{file}"]
    ),
    "kotlin": LanguageConfig(
        name="kotlin",
        extensions=[".kt", ".kts"],
        comment_single="//",
        comment_multi_start="/*",
        comment_multi_end="*/",
        linter_command=["ktlint", "{file}"],
        formatter_command=["ktlint", "-F", "{file}"]
    ),
}


def detect_language(file_path: str) -> Optional[LanguageConfig]:
    """Detect programming language from file extension"""
    ext = Path(file_path).suffix.lower()
    for lang_config in SUPPORTED_LANGUAGES.values():
        if ext in lang_config.extensions:
            return lang_config
    return None


# ============================================================================
# PYTHON ANALYZER
# ============================================================================

class PythonBugDetector:
    """Detect common Python bugs using AST analysis"""

    @staticmethod
    def analyze_file(file_path: str) -> List[Dict[str, Any]]:
        """Analyze Python file for common bugs"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            tree = ast.parse(code, filename=file_path)

            # Run various checks
            issues.extend(PythonBugDetector._check_mutable_defaults(tree))
            issues.extend(PythonBugDetector._check_bare_except(tree))
            issues.extend(PythonBugDetector._check_comparisons(tree))
            issues.extend(PythonBugDetector._check_unused_imports(tree, code))

        except SyntaxError as e:
            issues.append({
                "severity": "error",
                "type": "SyntaxError",
                "line": e.lineno or 0,
                "column": e.offset or 0,
                "message": str(e.msg),
                "suggestion": "Fix syntax error before proceeding"
            })
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            issues.append({
                "severity": "error",
                "type": "AnalysisError",
                "line": 0,
                "message": f"Failed to analyze: {str(e)}"
            })

        return issues

    @staticmethod
    def _check_mutable_defaults(tree: ast.AST) -> List[Dict]:
        """Check for mutable default arguments (dangerous!)"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for i, default in enumerate(node.args.defaults):
                    if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                        arg_name = node.args.args[-(len(node.args.defaults) - i)].arg
                        issues.append({
                            "severity": "error",
                            "type": "MutableDefault",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "message": f"Function '{node.name}' has mutable default argument '{arg_name}'",
                            "suggestion": f"Use None as default: {arg_name}=None, then inside: if {arg_name} is None: {arg_name} = []",
                            "fix_type": "mutable_default",
                            "function_name": node.name,
                            "arg_name": arg_name
                        })

        return issues

    @staticmethod
    def _check_bare_except(tree: ast.AST) -> List[Dict]:
        """Check for bare except clauses"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append({
                        "severity": "warning",
                        "type": "BareExcept",
                        "line": node.lineno,
                        "column": node.col_offset,
                        "message": "Bare 'except:' catches all exceptions (including SystemExit, KeyboardInterrupt)",
                        "suggestion": "Use 'except Exception as e:' to catch only Exception and its subclasses",
                        "fix_type": "bare_except"
                    })

        return issues

    @staticmethod
    def _check_comparisons(tree: ast.AST) -> List[Dict]:
        """Check for problematic comparisons"""
        issues = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                # Check for 'is' with literals
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, (ast.Is, ast.IsNot)):
                        if isinstance(comparator, (ast.Constant, ast.Num, ast.Str)):
                            literal_value = getattr(comparator, 'value',
                                                    getattr(comparator, 'n', getattr(comparator, 's', '?')))
                            issues.append({
                                "severity": "error",
                                "type": "IdentityComparison",
                                "line": node.lineno,
                                "column": node.col_offset,
                                "message": f"Using 'is' to compare with literal ({literal_value})",
                                "suggestion": "Use '==' for value comparison, 'is' for identity (None, True, False)",
                                "fix_type": "identity_comparison"
                            })

        return issues

    @staticmethod
    def _check_unused_imports(tree: ast.AST, code: str) -> List[Dict]:
        """Check for unused imports (simplified)"""
        issues = []
        imports = {}
        used_names = set()

        # Collect imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    import_name = alias.asname or alias.name.split('.')[0]
                    imports[import_name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    import_name = alias.asname or alias.name
                    imports[import_name] = node.lineno

        # Collect used names
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used_names.add(node.id)
            elif isinstance(node, ast.Attribute):
                # Check first part of attribute access
                if isinstance(node.value, ast.Name):
                    used_names.add(node.value.id)

        # Find unused
        for import_name, lineno in imports.items():
            if import_name not in used_names:
                # Double-check it's not in a string or comment
                if import_name in code:
                    # It's mentioned somewhere, might be intentional
                    continue

                issues.append({
                    "severity": "info",
                    "type": "UnusedImport",
                    "line": lineno,
                    "message": f"Import '{import_name}' appears unused",
                    "suggestion": f"Remove unused import or use it",
                    "fix_type": "unused_import",
                    "import_name": import_name
                })

        return issues


# ============================================================================
# JAVA ANALYZER
# ============================================================================

class JavaBugDetector:
    """Detect common Java bugs using pattern matching"""

    @staticmethod
    def analyze_file(file_path: str) -> List[Dict[str, Any]]:
        """Analyze Java file for common bugs"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                code = ''.join(lines)

            # Run various checks
            issues.extend(JavaBugDetector._check_missing_override(lines))
            issues.extend(JavaBugDetector._check_string_concatenation(lines))
            issues.extend(JavaBugDetector._check_null_pointer(lines))
            issues.extend(JavaBugDetector._check_empty_catch(lines))
            issues.extend(JavaBugDetector._check_system_out(lines))
            issues.extend(JavaBugDetector._check_equals_hashcode(code, lines))

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
    def _check_missing_override(lines: List[str]) -> List[Dict]:
        """Check for missing @Override annotations"""
        issues = []
        override_patterns = [
            r'\bpublic\s+\w+\s+toString\s*\(',
            r'\bpublic\s+boolean\s+equals\s*\(',
            r'\bpublic\s+int\s+hashCode\s*\(',
            r'\bpublic\s+int\s+compareTo\s*\(',
        ]

        for i, line in enumerate(lines, 1):
            for pattern in override_patterns:
                if re.search(pattern, line):
                    # Check if previous line has @Override
                    prev_line = lines[i-2].strip() if i > 1 else ""
                    if "@Override" not in prev_line:
                        issues.append({
                            "severity": "warning",
                            "type": "MissingOverride",
                            "line": i,
                            "message": "Method appears to override but missing @Override annotation",
                            "suggestion": "Add @Override annotation above method declaration"
                        })
                    break

        return issues

    @staticmethod
    def _check_string_concatenation(lines: List[str]) -> List[Dict]:
        """Check for inefficient string concatenation in loops"""
        issues = []

        for i, line in enumerate(lines, 1):
            if '+=' in line and 'String' in ''.join(lines[max(0, i-10):i]):
                # Check if we're in a loop
                context = ''.join(lines[max(0, i-20):i])
                if any(keyword in context for keyword in ['for (', 'while (', 'do {']):
                    issues.append({
                        "severity": "warning",
                        "type": "InefficientStringConcat",
                        "line": i,
                        "message": "String concatenation in loop is inefficient",
                        "suggestion": "Use StringBuilder.append() instead of += for better performance"
                    })

        return issues

    @staticmethod
    def _check_null_pointer(lines: List[str]) -> List[Dict]:
        """Check for potential null pointer dereferences"""
        issues = []

        for i, line in enumerate(lines, 1):
            # Simple heuristic: method call without null check
            if '.' in line and '(' in line:
                # Skip if it's a null check itself
                if any(check in line for check in ['!= null', '== null', 'Objects.requireNonNull']):
                    continue

                # Look for variables that might be null
                context = ''.join(lines[max(0, i-5):i])
                if 'null' in context or 'return null' in context:
                    issues.append({
                        "severity": "info",
                        "type": "PotentialNullPointer",
                        "line": i,
                        "message": "Potential null pointer dereference",
                        "suggestion": "Add null check or use Optional<T> to handle nullable values safely"
                    })

        return issues

    @staticmethod
    def _check_empty_catch(lines: List[str]) -> List[Dict]:
        """Check for empty catch blocks"""
        issues = []

        for i, line in enumerate(lines, 1):
            if 'catch' in line and '(' in line:
                # Look for empty catch block
                j = i
                while j < len(lines):
                    next_line = lines[j].strip()
                    if next_line == '{':
                        if j + 1 < len(lines) and lines[j + 1].strip() == '}':
                            issues.append({
                                "severity": "warning",
                                "type": "EmptyCatchBlock",
                                "line": i,
                                "message": "Empty catch block swallows exceptions silently",
                                "suggestion": "Add logging or re-throw exception, or at minimum add a comment explaining why it's empty"
                            })
                        break
                    elif '{' in next_line:
                        break
                    j += 1

        return issues

    @staticmethod
    def _check_system_out(lines: List[str]) -> List[Dict]:
        """Check for System.out.println usage"""
        issues = []

        for i, line in enumerate(lines, 1):
            if 'System.out.print' in line:
                issues.append({
                    "severity": "info",
                    "type": "SystemOut",
                    "line": i,
                    "message": "Using System.out for logging",
                    "suggestion": "Use a proper logging framework (SLF4J, Log4j2, java.util.logging)"
                })

        return issues

    @staticmethod
    def _check_equals_hashcode(code: str, lines: List[str]) -> List[Dict]:
        """Check if equals is overridden without hashCode"""
        issues = []

        has_equals = 'public boolean equals(' in code
        has_hashcode = 'public int hashCode(' in code

        if has_equals and not has_hashcode:
            # Find line number of equals method
            for i, line in enumerate(lines, 1):
                if 'public boolean equals(' in line:
                    issues.append({
                        "severity": "error",
                        "type": "EqualsWithoutHashCode",
                        "line": i,
                        "message": "equals() overridden but hashCode() is not",
                        "suggestion": "Always override hashCode() when overriding equals() to maintain contract"
                    })
                    break

        return issues


# ============================================================================
# KOTLIN ANALYZER
# ============================================================================

class KotlinBugDetector:
    """Detect common Kotlin anti-patterns"""

    @staticmethod
    def analyze_file(file_path: str) -> List[Dict[str, Any]]:
        """Analyze Kotlin file for common issues"""
        issues = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                code = ''.join(lines)

            # Run various checks
            issues.extend(KotlinBugDetector._check_force_unwrap(lines))
            issues.extend(KotlinBugDetector._check_mutable_collections(lines))
            issues.extend(KotlinBugDetector._check_java_style(lines))
            issues.extend(KotlinBugDetector._check_redundant_types(lines))
            issues.extend(KotlinBugDetector._check_empty_when(lines))
            issues.extend(KotlinBugDetector._check_platform_types(lines))

        except Exception as e:
            logger.error(f"Kotlin analysis failed: {e}")
            issues.append({
                "severity": "error",
                "type": "AnalysisError",
                "line": 0,
                "message": f"Failed to analyze: {str(e)}"
            })

        return issues

    @staticmethod
    def _check_force_unwrap(lines: List[str]) -> List[Dict]:
        """Check for !! (force unwrap) operator"""
        issues = []

        for i, line in enumerate(lines, 1):
            if '!!' in line:
                # Exclude comments
                if '//' in line and line.index('//') < line.index('!!'):
                    continue

                issues.append({
                    "severity": "warning",
                    "type": "ForceUnwrap",
                    "line": i,
                    "message": "Using !! (force unwrap) can throw NullPointerException at runtime",
                    "suggestion": "Use safe call (?.), let/also/apply, or elvis operator (?:) instead"
                })

        return issues

    @staticmethod
    def _check_mutable_collections(lines: List[str]) -> List[Dict]:
        """Check for unnecessary mutable collections"""
        issues = []

        for i, line in enumerate(lines, 1):
            if any(mut in line for mut in ['mutableListOf', 'mutableMapOf', 'mutableSetOf']):
                # Check if collection is actually modified
                is_modified = False
                for j in range(i, min(i+15, len(lines))):
                    if any(op in lines[j] for op in ['.add(', '.put(', '.remove(', '.clear()']):
                        is_modified = True
                        break

                if not is_modified:
                    issues.append({
                        "severity": "info",
                        "type": "UnnecessaryMutable",
                        "line": i,
                        "message": "Using mutable collection but it appears to be immutable",
                        "suggestion": "Use listOf(), mapOf(), or setOf() for immutable collections"
                    })

        return issues

    @staticmethod
    def _check_java_style(lines: List[str]) -> List[Dict]:
        """Check for Java-style syntax in Kotlin"""
        issues = []

        for i, line in enumerate(lines, 1):
            # Java-style for loop
            if re.search(r'for\s*\(\s*\w+\s+\w+\s*:\s*\w+\s*\)', line):
                issues.append({
                    "severity": "info",
                    "type": "NonIdiomatic",
                    "line": i,
                    "message": "Using Java-style for loop syntax",
                    "suggestion": "Use Kotlin's 'for (item in collection)' syntax"
                })

            # Java-style getter/setter
            if re.search(r'\.(get|set)[A-Z]\w+\(', line):
                issues.append({
                    "severity": "info",
                    "type": "NonIdiomatic",
                    "line": i,
                    "message": "Using Java-style getter/setter",
                    "suggestion": "Use Kotlin property access syntax (obj.property instead of obj.getProperty())"
                })

        return issues

    @staticmethod
    def _check_redundant_types(lines: List[str]) -> List[Dict]:
        """Check for redundant explicit type annotations"""
        issues = []

        for i, line in enumerate(lines, 1):
            # val name: Type = Constructor()
            match = re.search(r'val\s+\w+\s*:\s*(\w+)\s*=\s*\1\(', line)
            if match:
                issues.append({
                    "severity": "info",
                    "type": "RedundantTypeAnnotation",
                    "line": i,
                    "message": "Explicit type annotation is redundant",
                    "suggestion": "Let Kotlin's type inference work: val name = Type()"
                })

        return issues

    @staticmethod
    def _check_empty_when(lines: List[str]) -> List[Dict]:
        """Check for empty when expressions"""
        issues = []

        for i, line in enumerate(lines, 1):
            if 'when' in line and '{' in line:
                # Look ahead for empty when
                j = i
                has_cases = False
                while j < len(lines) and j < i + 20:
                    if '->' in lines[j]:
                        has_cases = True
                        break
                    if lines[j].strip() == '}':
                        if not has_cases:
                            issues.append({
                                "severity": "warning",
                                "type": "EmptyWhen",
                                "line": i,
                                "message": "Empty when expression",
                                "suggestion": "Add cases or remove when expression"
                            })
                        break
                    j += 1

        return issues

    @staticmethod
    def _check_platform_types(lines: List[str]) -> List[Dict]:
        """Check for potential platform type issues"""
        issues = []

        for i, line in enumerate(lines, 1):
            # Look for Java interop without null safety
            if 'import java.' in line or 'import javax.' in line:
                # This is just a heuristic
                for j in range(i, min(i+30, len(lines))):
                    if '!!' in lines[j]:
                        # Already flagged by force unwrap check
                        continue
                    # Look for method calls on potentially null Java objects
                    if re.search(r'\w+\.\w+\(', lines[j]) and '?.' not in lines[j]:
                        issues.append({
                            "severity": "info",
                            "type": "PlatformType",
                            "line": j + 1,
                            "message": "Calling method on potentially null Java object",
                            "suggestion": "Use safe call (?.) or explicit null check when working with Java APIs"
                        })
                        break

        return issues


# ============================================================================
# GENERIC ANALYZER (for non-Python)
# ============================================================================

def run_linter(file_path: str, lang_config: LanguageConfig) -> Dict[str, Any]:
    """Run external linter and parse results"""
    if not lang_config.linter_command:
        return {"status": "no_linter", "issues": []}

    try:
        cmd = [arg.format(file=file_path) for arg in lang_config.linter_command]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=Path(file_path).parent
        )

        return {
            "status": "success" if result.returncode == 0 else "issues_found",
            "output": result.stdout,
            "errors": result.stderr,
            "return_code": result.returncode
        }

    except FileNotFoundError:
        return {
            "status": "linter_not_installed",
            "tool": lang_config.linter_command[0],
            "message": f"Linter '{lang_config.linter_command[0]}' not found. Install it first."
        }
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "message": "Linter timed out after 5 minutes"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def analyze_code_file(file_path: str, language: str = "auto") -> Dict[str, Any]:
    """
    Analyze a code file for bugs and issues.

    Args:
        file_path: Path to code file
        language: Language override or "auto"

    Returns:
        Dict with issues, stats, and metadata
    """
    logger.info(f"🔍 Analyzing: {file_path}")

    # Validate file exists
    if not Path(file_path).exists():
        return {
            "error": f"File not found: {file_path}",
            "status": "file_not_found"
        }

    # Detect language
    if language == "auto":
        lang_config = detect_language(file_path)
        if not lang_config:
            return {
                "error": f"Could not detect language for {file_path}",
                "supported_extensions": [ext for cfg in SUPPORTED_LANGUAGES.values() for ext in cfg.extensions],
                "status": "unknown_language"
            }
    else:
        lang_config = SUPPORTED_LANGUAGES.get(language.lower())
        if not lang_config:
            return {
                "error": f"Language '{language}' not supported",
                "supported_languages": list(SUPPORTED_LANGUAGES.keys()),
                "status": "unsupported_language"
            }

    # Run analysis
    try:
        if lang_config.name == "python":
            issues = PythonBugDetector.analyze_file(file_path)
        elif lang_config.name == "java":
            issues = JavaBugDetector.analyze_file(file_path)
        elif lang_config.name == "kotlin":
            issues = KotlinBugDetector.analyze_file(file_path)
        else:
            # Use external linter
            linter_result = run_linter(file_path, lang_config)
            if linter_result['status'] == 'linter_not_installed':
                return {
                    "error": linter_result['message'],
                    "status": "linter_not_available",
                    "language": lang_config.name
                }
            issues = _parse_linter_output(linter_result, lang_config)

        # Categorize
        errors = [i for i in issues if i.get('severity') == 'error']
        warnings = [i for i in issues if i.get('severity') == 'warning']
        info = [i for i in issues if i.get('severity') == 'info']

        return {
            "file": file_path,
            "language": lang_config.name,
            "status": "success",
            "total_issues": len(issues),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info),
            "fixable": len([i for i in issues if 'fix_type' in i]),
            "issues": issues,
            "summary": f"Found {len(errors)} errors, {len(warnings)} warnings, {len(info)} info"
        }

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return {
            "error": str(e),
            "status": "analysis_failed",
            "file": file_path
        }


def _parse_linter_output(linter_result: Dict, lang_config: LanguageConfig) -> List[Dict]:
    """Parse linter output into standardized format"""
    # TODO: Implement language-specific parsing
    # For now, return empty list
    return []