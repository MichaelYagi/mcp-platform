"""
RAG Storage Tracking
Tracks which Plex items have been ingested
"""

import json
import logging
from pathlib import Path
from typing import Dict, Set

logger = logging.getLogger("mcp_server")

# Storage file for tracking ingested items
STORAGE_FILE = Path(__file__).parent / "ingested_items.json"


def load_ingested_items() -> Dict[str, str]:
    """
    Load dictionary of ingested media IDs with their status

    Returns:
        Dict mapping media_id -> status ("success" or "no_subtitles")
    """
    if not STORAGE_FILE.exists():
        return {}

    try:
        with open(STORAGE_FILE, 'r') as f:
            data = json.load(f)
            return data.get("ingested_items", {})
    except Exception as e:
        logger.error(f"❌ Error loading ingested items: {e}")
        return {}


def save_ingested_items(items: Dict[str, str]):
    """Save dictionary of ingested media IDs with their status"""
    try:
        with open(STORAGE_FILE, 'w') as f:
            json.dump({"ingested_items": items}, f, indent=2)
    except Exception as e:
        logger.error(f"❌ Error saving ingested items: {e}")


def check_if_ingested(media_id: str, skip_no_subtitles: bool = False) -> bool:
    """
    Check if a media item has been ingested

    Args:
        media_id: The media ID to check
        skip_no_subtitles: If True, items with "no_subtitles" status are treated as not ingested
                          (allows re-scanning if subtitles are added later)

    Returns:
        True if already ingested and should be skipped
    """
    ingested = load_ingested_items()

    if media_id not in ingested:
        return False

    status = ingested[media_id]

    # If skip_no_subtitles is True, allow re-checking items that previously had no subtitles
    if skip_no_subtitles and status == "no_subtitles":
        return False

    return True


def mark_as_ingested(media_id: str, status: str = "success"):
    """
    Mark a media item as ingested

    Args:
        media_id: The media ID
        status: Either "success" (has subtitles) or "no_subtitles"
    """
    ingested = load_ingested_items()
    ingested[media_id] = status
    save_ingested_items(ingested)


def get_ingestion_stats() -> Dict[str, int]:
    """Get ingestion statistics"""
    from tools.plex.plex_utils import stream_all_media

    ingested = load_ingested_items()

    # Count by status
    success_count = sum(1 for status in ingested.values() if status == "success")
    no_subtitles_count = sum(1 for status in ingested.values() if status == "no_subtitles")

    total_items = sum(1 for _ in stream_all_media())

    return {
        "total_items": total_items,
        "successfully_ingested": success_count,
        "missing_subtitles": no_subtitles_count,
        "total_processed": len(ingested),
        "remaining": total_items - len(ingested)
    }


def reset_no_subtitle_items():
    """Reset items that were marked as 'no_subtitles' to allow re-scanning"""
    ingested = load_ingested_items()

    # Remove all "no_subtitles" entries
    removed_count = 0
    updated = {}
    for media_id, status in ingested.items():
        if status == "no_subtitles":
            removed_count += 1
        else:
            updated[media_id] = status

    save_ingested_items(updated)
    logger.info(f"🔄 Reset {removed_count} items marked as 'no_subtitles'")
    return removed_count


def reset_ingestion_tracking():
    """Reset all ingestion tracking (for testing)"""
    if STORAGE_FILE.exists():
        STORAGE_FILE.unlink()
    logger.info("🔄 Ingestion tracking reset")