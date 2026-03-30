import subprocess
import json
from pathlib import Path

# Use your existing directory logic
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent  # Adjust based on where your code lives


def search_code(query, extension=None, directory="."):
    """
    Core search logic for finding text in files.

    Args:
        query: String to search for (case-insensitive)
        directory: Directory to search in
        extension: Optional file extension filter (e.g., 'py', 'js')

    Returns:
        String containing search results or error message
    """
    import os

    matches = []
    ignore_dirs = {".git", "node_modules", "venv", ".venv", "__pycache__", ".mcp_use"}

    try:
        # Ensure we have a valid directory
        if not directory or directory == ".":
            search_path = os.getcwd()
        else:
            search_path = os.path.abspath(directory)

        # Verify directory exists
        if not os.path.exists(search_path):
            return f"Directory does not exist: {search_path}"
        if not os.path.isdir(search_path):
            return f"Path is not a directory: {search_path}"

        count = 0

        for root, dirs, files in os.walk(search_path, topdown=True, onerror=lambda e: None):
            dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith('.')]

            # Limit depth
            try:
                depth = root[len(search_path):].count(os.sep)
                if depth > 10:
                    continue
            except:
                continue

            for fname in files:
                count += 1
                if count > 500:
                    return "\n".join(matches) + "\n[Stopped at 500 files]"

                if fname.startswith('.'):
                    continue
                if extension and not fname.endswith(f".{extension.lstrip('.')}"):
                    continue

                fpath = os.path.join(root, fname)

                try:
                    if os.path.getsize(fpath) > 1048576:  # 1MB
                        continue

                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        for i, line in enumerate(f, 1):
                            if i > 5000:
                                break
                            if query.lower() in line.lower():
                                try:
                                    rel = os.path.relpath(fpath, search_path)
                                except:
                                    rel = fpath
                                matches.append(f"./{rel}:{i}: {line.strip()}")
                                if len(matches) >= 50:
                                    return "\n".join(matches) + "\n[Truncated: 50+ matches]"
                except:
                    pass

        return "\n".join(matches) if matches else f"No matches for '{query}' (searched {count} files)"
    except Exception as e:
        return f"Search error: {type(e).__name__}: {str(e)}"