import os
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

TEXT_EXTENSIONS = {".py", ".js", ".ts", ".kt", ".java", ".json", ".md"}

def scan_directory(path):
    root = Path(path).resolve()
    files = []

    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            ext = Path(name).suffix
            if ext not in TEXT_EXTENSIONS:
                continue

            file_path = Path(dirpath) / name
            try:
                with open(file_path, "r", errors="ignore") as f:
                    lines = sum(1 for _ in f)
            except Exception:
                continue

            files.append({
                "path": str(file_path.relative_to(root)),
                "extension": ext,
                "lines": lines
            })

    result = {
        "root": str(root),
        "total_files": len(files),
        "files": files
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "last_scan.json", "w") as f:
        json.dump(result, f, indent=2)

    return result
