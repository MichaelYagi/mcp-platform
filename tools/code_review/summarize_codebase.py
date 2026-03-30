import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"

def summarize_codebase():
    scan_file = DATA_DIR / "last_scan.json"
    if not scan_file.exists():
        return {"error": "No scan data found"}

    with open(scan_file) as f:
        scan = json.load(f)

    languages = {}
    for f in scan["files"]:
        languages[f["extension"]] = languages.get(f["extension"], 0) + 1

    summary = {
        "summary": "This is a local codebase scanned for structure and organization.",
        "total_files": scan["total_files"],
        "languages": languages,
        "architecture_notes": [
            "File-based organization",
            "Multiple source files detected"
        ],
        "potential_issues": [
            "No automated linting detected",
            "No dependency manifest analysis"
        ]
    }

    return summary
