import json
import traceback
from typing import Optional

def fix_bug(
    error_message: str,
    stack_trace: Optional[str] = None,
    code_snippet: Optional[str] = None,
    environment: Optional[str] = None
) -> dict:
    """
    Analyze an error and suggest likely fixes.
    """

    suggestions = []

    if "ModuleNotFoundError" in error_message:
        suggestions.append(
            "Check that the module is installed in the active virtual environment "
            "and that you are using the correct Python interpreter."
        )

    if "unexpected keyword argument" in error_message:
        suggestions.append(
            "The function signature may have changed. Check the installed library version "
            "and update your code to match the current API."
        )

    if "AssertionError" in error_message:
        suggestions.append(
            "An internal invariant failed. Inspect the assertion condition and ensure "
            "inputs match the expected type or structure."
        )

    if "KeyError" in error_message:
        suggestions.append(
            "A dictionary key was missing. Print available keys before accessing the value."
        )

    if not suggestions:
        suggestions.append(
            "Inspect the stack trace carefully and verify input types, function arguments, "
            "and library versions."
        )

    return {
        "summary": "Bug analysis complete",
        "error_message": error_message,
        "environment": environment,
        "suggestions": suggestions,
        "provided_stack_trace": bool(stack_trace),
        "provided_code": bool(code_snippet)
    }
