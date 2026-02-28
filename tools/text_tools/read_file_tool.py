"""
File Reader Tool
Reads any local file and returns its content for LLM analysis.
Supports: CSV, JSON, TXT, MD, PY, JS, TS, YAML, TOML, XML, LOG, and more.
Binary files (images, executables) are rejected gracefully.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("mcp_server")

# Max bytes to read — keeps context window sane for large files
MAX_FILE_BYTES = 100_000  # ~100KB, roughly 25k tokens

# Extensions we'll refuse (binary / not useful as text)
BINARY_EXTENSIONS = {
    ".exe", ".dll", ".so", ".bin", ".dat", ".db", ".sqlite",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".svg",
    ".mp3", ".mp4", ".avi", ".mkv", ".mov",
    ".pdf", ".docx", ".xlsx", ".pptx",  # handled by dedicated tools
}

# Extensions we treat as CSV-like (show row count, column names in summary)
CSV_EXTENSIONS = {".csv", ".tsv"}


def _ingest_file_to_rag(text: str, filename: str) -> None:
    """
    Fire-and-forget: chunk, embed, and insert file content into the RAG store.
    Runs in a daemon thread — never blocks the caller.
    Source is set to "file:<filename>" so it's queryable and identifiable.
    If rag_add is unavailable (import error), logs a warning and continues.
    """
    def _run():
        try:
            from tools.rag.rag_add import rag_add
            source = f"file:{filename}"
            result = rag_add(text, source=source)
            if result.get("success"):
                logger.info(
                    f"📚 RAG ingest complete: {filename} "
                    f"({result.get('chunks_added', 0)} chunks, "
                    f"{result.get('processing_time_seconds', 0):.1f}s)"
                )
            else:
                logger.warning(
                    f"⚠️ RAG ingest partial/failed for {filename}: "
                    f"{result.get('error', 'unknown error')}"
                )
        except ImportError:
            logger.warning("⚠️ rag_add not available — skipping RAG ingest for file read")
        except Exception as e:
            logger.error(f"❌ RAG ingest failed for {filename}: {e}")

    thread = threading.Thread(target=_run, daemon=True, name=f"rag-ingest-{filename}")
    thread.start()
    logger.debug(f"🔄 RAG ingest started in background for: {filename}")


def read_file_tool(file_path: str) -> Dict[str, Any]:
    """
    Read a local file and return its content for analysis.

    Args:
        file_path: Absolute or relative path to the file.
                   IMPORTANT: Always pass the COMPLETE path including any spaces
                   in the filename. Do not truncate at spaces.
                   Windows paths like C:\\Users\\... are accepted and
                   translated to /mnt/c/... for WSL automatically.

    Returns:
        Dict with:
          - success: bool
          - content: file text (possibly truncated)
          - file_name: basename
          - file_type: extension
          - size_bytes: actual file size
          - truncated: bool — True if content was cut
          - row_count: (CSV only) number of data rows
          - columns: (CSV only) list of column names
          - error: (on failure) error message
    """
    # ── Normalise path ────────────────────────────────────────────
    # Strip outer whitespace and any wrapping quotes the LLM may add
    path_str = file_path.strip().strip('"\'').strip()

    # Translate Windows paths to WSL mount points
    if len(path_str) >= 3 and path_str[1] == ':':
        drive = path_str[0].lower()
        rest = path_str[2:].replace('\\', '/')
        path_str = f"/mnt/{drive}{rest}"

    path = Path(path_str)

    # ── Fuzzy fallback for paths with spaces ──────────────────────
    # LLMs sometimes truncate filenames at the first space when the path
    # isn't quoted. If the path doesn't exist but the parent dir does,
    # search for a file whose name starts with the stem we were given.
    if not path.exists() and path.parent.exists():
        stem = path.name  # e.g. "2025-2026"
        candidates = [
            f for f in path.parent.iterdir()
            if f.is_file() and f.name.startswith(stem)
        ]
        if len(candidates) == 1:
            logger.info(
                f"🔍 Fuzzy match: '{path.name}' → '{candidates[0].name}'"
            )
            path = candidates[0]
        elif len(candidates) > 1:
            # Multiple matches — return helpful error listing them
            names = [f.name for f in candidates]
            return {
                "success": False,
                "error": (
                    f"Ambiguous path — multiple files match '{stem}': {names}. "
                    f"Please provide the full filename including any spaces."
                ),
                "file_path": str(path)
            }

    logger.info(f"📂 read_file_tool: {path}")

    # ── Validate ──────────────────────────────────────────────────
    if not path.exists():
        return {
            "success": False,
            "error": f"File not found: {path}",
            "file_path": str(path)
        }

    if not path.is_file():
        return {
            "success": False,
            "error": f"Path is not a file: {path}",
            "file_path": str(path)
        }

    ext = path.suffix.lower()

    if ext in BINARY_EXTENSIONS:
        return {
            "success": False,
            "error": (
                f"Binary file type '{ext}' is not supported for text analysis. "
                f"For Excel files use the spreadsheet tool; for PDFs use the PDF tool."
            ),
            "file_path": str(path)
        }

    size_bytes = path.stat().st_size

    # ── Read content ──────────────────────────────────────────────
    try:
        raw = path.read_bytes()
        # Try UTF-8 first, fall back to latin-1 (covers most Western CSVs)
        try:
            text = raw[:MAX_FILE_BYTES].decode('utf-8')
        except UnicodeDecodeError:
            text = raw[:MAX_FILE_BYTES].decode('latin-1')

        truncated = size_bytes > MAX_FILE_BYTES

    except PermissionError:
        return {
            "success": False,
            "error": f"Permission denied reading: {path}",
            "file_path": str(path)
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error reading file: {e}",
            "file_path": str(path)
        }

    result: Dict[str, Any] = {
        "success": True,
        "file_name": path.name,
        "file_path": str(path),
        "file_type": ext,
        "size_bytes": size_bytes,
        "truncated": truncated,
        "content": text,
    }

    if truncated:
        result["truncation_note"] = (
            f"File is {size_bytes:,} bytes. Only the first "
            f"{MAX_FILE_BYTES:,} bytes are shown. "
            f"Ask to see a specific section if needed."
        )

    # ── Ingest into RAG — fire-and-forget ─────────────────────────
    # Run in a background thread so it never blocks the response.
    # Source format: "file:<filename>" so it's queryable and distinct
    # from conversation turns (which use session_id) and web sources.
    _ingest_file_to_rag(text, path.name)

    # ── CSV extras — column names + row count ─────────────────────
    if ext in CSV_EXTENSIONS:
        try:
            import csv
            import io
            delimiter = '\t' if ext == '.tsv' else ','
            reader = csv.reader(io.StringIO(text), delimiter=delimiter)
            rows = list(reader)
            if rows:
                result["columns"] = rows[0]
                result["row_count"] = len(rows) - 1  # exclude header
                logger.info(
                    f"📊 CSV: {len(rows[0])} columns, {result['row_count']} rows"
                )
        except Exception as csv_err:
            logger.warning(f"⚠️ CSV metadata extraction failed: {csv_err}")

    logger.info(
        f"✅ read_file: {path.name} ({size_bytes:,} bytes, "
        f"truncated={truncated})"
    )
    return result