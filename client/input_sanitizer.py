"""
Input Sanitization Module
Handles various edge cases in user input to prevent LLM parsing issues
"""

import re
import html
import logging

logger = logging.getLogger("mcp_client")


def sanitize_user_input(text: str, preserve_markdown: bool = True) -> str:
    """
    Sanitize user input to prevent LLM parsing issues

    Handles:
    - HTML/XML tags (e.g., <div>, <script>)
    - Special characters that confuse prompt parsers
    - Excessive whitespace
    - Control characters
    - Prompt injection attempts
    - Null bytes

    Args:
        text: Raw user input
        preserve_markdown: Keep markdown formatting (default: True)

    Returns:
        Sanitized text safe for LLM processing
    """
    if not text:
        return text

    original_length = len(text)

    # 1. Remove null bytes and control characters (except newline, tab)
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)

    # 2. Escape HTML entities to prevent tag confusion
    # This handles: <div>, <script>, <!-- comments -->, etc.
    text = html.escape(text, quote=False)  # Don't escape quotes (for markdown)

    # 3. Normalize excessive whitespace (but preserve single newlines)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Replace 3+ newlines with 2 newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace from each line
    text = '\n'.join(line.strip() for line in text.split('\n'))

    # 4. Remove dangerous prompt injection patterns
    # Patterns that try to override system instructions
    injection_patterns = [
        r'(?i)ignore\s+(all\s+)?previous\s+instructions?',
        r'(?i)disregard\s+(all\s+)?previous\s+instructions?',
        r'(?i)forget\s+(all\s+)?previous\s+instructions?',
        r'(?i)you\s+are\s+now\s+a\s+',
        r'(?i)system\s*:\s*',  # Fake system messages
        r'(?i)assistant\s*:\s*',  # Fake assistant messages
        r'(?i)\[INST\]',  # Llama instruction tags
        r'(?i)\[/INST\]',
        r'(?i)<\|im_start\|>',  # ChatML tags
        r'(?i)<\|im_end\|>',
    ]

    for pattern in injection_patterns:
        if re.search(pattern, text):
            logger.warning(f"⚠️  Potential prompt injection detected: {pattern}")
            # Don't remove - just log and continue (user might have legitimate use)

    # 5. Limit length (prevent token overflow attacks)
    max_length = 10000  # ~2500 tokens for most models
    if len(text) > max_length:
        logger.warning(f"⚠️  Input truncated from {len(text)} to {max_length} characters")
        text = text[:max_length] + "\n\n[Input truncated due to length]"

    # 6. If markdown should be preserved, unescape certain patterns
    if preserve_markdown:
        # Restore common markdown that was escaped
        text = text.replace('&ast;&ast;', '**')  # Bold
        text = text.replace('&ast;', '*')  # Italic/lists
        text = text.replace('&#x60;&#x60;&#x60;', '```')  # Code blocks
        text = text.replace('&#x60;', '`')  # Inline code

    # Log if significant changes were made
    if len(text) < original_length * 0.9:
        logger.info(f"📝 Input sanitized: {original_length} → {len(text)} chars")

    return text.strip()


def is_safe_input(text: str) -> tuple[bool, str]:
    """
    Check if input is safe without modifying it

    Returns:
        (is_safe, reason) tuple
    """
    if not text:
        return True, "Empty input"

    # Check for null bytes
    if '\x00' in text:
        return False, "Contains null bytes"

    # Check for excessive length
    if len(text) > 50000:
        return False, f"Too long ({len(text)} chars)"

    # Check for binary data
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return False, "Contains invalid Unicode"

    # Check for obvious injection attempts
    dangerous_patterns = [
        r'(?i)ignore\s+all\s+previous\s+instructions',
        r'(?i)you\s+are\s+now\s+DAN',
        r'(?i)jailbreak',
    ]

    for pattern in dangerous_patterns:
        if re.search(pattern, text):
            return False, f"Potential injection: {pattern}"

    return True, "Safe"


# Quick sanitization for commands (don't escape HTML in commands)
def sanitize_command(text: str) -> str:
    """
    Minimal sanitization for commands (start with :)
    Only removes control characters and excessive whitespace
    """
    if not text:
        return text

    # Remove control characters (except newline, tab)
    text = ''.join(char for char in text if char == '\n' or char == '\t' or ord(char) >= 32)

    # Normalize whitespace
    text = ' '.join(text.split())

    return text.strip()