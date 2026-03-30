"""
RAG Diagnostics Tool
Finds incomplete or missing RAG entries
"""

import logging
from typing import Dict, Any, List
from tools.plex.plex_utils import stream_all_media, stream_subtitles
from tools.rag.rag_storage import load_ingested_items, get_ingestion_stats

logger = logging.getLogger("mcp_server")


def diagnose_plex_rag() -> Dict[str, Any]:
    """
    Diagnose RAG database for incomplete entries.

    Returns:
        Dictionary with diagnostic information
    """
    try:
        # Check if Plex tools are disabled before attempting connection
        from tools.tool_control import is_tool_enabled
        if not is_tool_enabled("plex_ingest_batch", "plex"):
            logger.info("⏭️  Plex is disabled — skipping Plex diagnostics")
            try:
                stats = get_ingestion_stats()
            except Exception:
                stats = {"total_items": 0, "successfully_ingested": 0, "missing_subtitles": 0, "total_processed": 0, "remaining": 0}
            return {
                "status": "plex_disabled",
                "message": "Plex tools are disabled (DISABLED_TOOLS). RAG diagnostics require Plex.",
                "ingestion_stats": stats
            }

        logger.info("🔍 Starting RAG diagnostics...")

        ingested_items = load_ingested_items()
        missing_subtitles = []
        not_yet_ingested = []

        # Check all media items
        for media_item in stream_all_media():
            media_id = str(media_item["id"])
            title = media_item["title"]
            media_type = media_item["type"]

            # Check if already ingested
            if media_id in ingested_items:
                status = ingested_items[media_id]
                if status == "no_subtitles":
                    missing_subtitles.append({
                        "title": title,
                        "id": media_id,
                        "type": media_type
                    })
                    logger.debug(f"❌ No subtitles: {title}")
            else:
                # Not yet processed - check if it has subtitles
                subtitle_lines = list(stream_subtitles(media_id))

                if not subtitle_lines:
                    missing_subtitles.append({
                        "title": title,
                        "id": media_id,
                        "type": media_type
                    })
                    logger.debug(f"❌ No subtitles: {title}")
                else:
                    not_yet_ingested.append({
                        "title": title,
                        "id": media_id,
                        "type": media_type,
                        "subtitle_lines": len(subtitle_lines)
                    })
                    logger.debug(f"⏳ Not yet ingested: {title} ({len(subtitle_lines)} subtitle lines)")

        stats = get_ingestion_stats()

        result = {
            "total_items": stats["total_items"],
            "ingested_count": stats["successfully_ingested"],  # Fixed key name
            "missing_subtitles": missing_subtitles,
            "not_yet_ingested": not_yet_ingested,
            "statistics": {
                "items_with_no_subtitles": len(missing_subtitles),
                "items_ready_to_ingest": len(not_yet_ingested),
                "items_successfully_ingested": stats["successfully_ingested"],
                "items_marked_no_subtitles": stats["missing_subtitles"],
                "total_processed": stats["total_processed"],
                "completion_percentage": round((stats["successfully_ingested"] / stats["total_items"] * 100), 2) if stats["total_items"] > 0 else 0
            }
        }

        logger.info(f"📊 Diagnostics complete:")
        logger.info(f"  - Total items: {stats['total_items']}")
        logger.info(f"  - Successfully ingested: {stats['successfully_ingested']}")
        logger.info(f"  - Missing subtitles: {len(missing_subtitles)}")
        logger.info(f"  - Ready to ingest: {len(not_yet_ingested)}")

        return result

    except Exception as e:
        logger.error(f"❌ Diagnostics error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}