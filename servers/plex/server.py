"""
Plex MCP Server
Runs over stdio transport
"""
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

# ═══════════════════════════════════════════════════════════════════
# CHECK PLEX ENVIRONMENT BEFORE IMPORTING TOOLS
# ═══════════════════════════════════════════════════════════════════
import logging

# Create a basic logger for startup checks
startup_logger = logging.getLogger("plex_startup")
startup_logger.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
startup_logger.addHandler(console)

PLEX_URL = os.getenv("PLEX_URL")
PLEX_TOKEN = os.getenv("PLEX_TOKEN")

if not PLEX_URL or not PLEX_TOKEN:
    startup_logger.warning("=" * 70)
    startup_logger.warning("⚠️  PLEX_URL and PLEX_TOKEN not configured")
    startup_logger.warning("   Plex server will start but tools will be unavailable")
    startup_logger.warning("   Set these in .env to enable Plex functionality:")
    startup_logger.warning("   PLEX_URL=http://localhost:32400")
    startup_logger.warning("   PLEX_TOKEN=your_plex_token_here")
    startup_logger.warning("=" * 70)
    PLEX_AVAILABLE = False
else:
    PLEX_AVAILABLE = True
    startup_logger.info("✅ Plex configuration found")

# ═══════════════════════════════════════════════════════════════════
# NOW SAFE TO IMPORT (conditionally)
# ═══════════════════════════════════════════════════════════════════
from typing import Dict, Any, List, Optional
from servers.skills.skill_loader import SkillLoader

import inspect
import json
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response
from client.stop_signal import is_stop_requested
from mcp.server.fastmcp import FastMCP

# Import Plex tools conditionally
if PLEX_AVAILABLE:
    from tools.plex.semantic_media_search import semantic_media_search
    from tools.plex.scene_locator import scene_locator
    from tools.plex.ingest import ingest_next_batch, ingest_batch_parallel_conservative, find_unprocessed_items, process_item_async
    from servers.plex.ml_recommender import get_recommender

# Rest of your logging setup
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers (in case something already configured it)
root_logger.handlers.clear()

# Create formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Create file handler
file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Disable propagation to avoid duplicate logs
logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_plex_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_plex_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

if PLEX_AVAILABLE:
    logger.info("✅ Plex tools available")
else:
    logger.warning("⚠️  Plex tools unavailable - set PLEX_URL and PLEX_TOKEN in .env")

mcp = FastMCP("plex-server")

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTION FOR UNAVAILABLE TOOLS
# ═══════════════════════════════════════════════════════════════════
def plex_unavailable_error():
    """Return error message when Plex is not configured"""
    return {
        "error": "Plex is not configured",
        "message": "Set PLEX_URL and PLEX_TOKEN in .env to enable Plex functionality",
        "help": {
            "PLEX_URL": "http://localhost:32400",
            "PLEX_TOKEN": "Get from Plex Settings > Network > Show Advanced"
        }
    }

# ═══════════════════════════════════════════════════════════════════
# TOOL DEFINITIONS WITH AVAILABILITY CHECKS
# ═══════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="plex")
def semantic_media_search_text(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search for movies and TV shows in the Plex library by title, genre, actor, or description."""
    if not PLEX_AVAILABLE:
        return plex_unavailable_error()

    if not query or not query.strip():
        raise ValueError("semantic_media_search_text called with empty query")
    logger.info(f"🛠 [server] semantic_media_search called with query: {query}, limit: {limit}")
    return semantic_media_search(query=query, limit=limit)


@mcp.tool()
@check_tool_enabled(category="plex")
def scene_locator_tool(media_id: str, query: str, limit: int = 5):
    """Find specific scenes within a movie or TV show using subtitle search."""
    if not PLEX_AVAILABLE:
        return plex_unavailable_error()

    logger.info(f"🛠 [server] scene_locator_tool called with media_id: {media_id}, query: {query}, limit: {limit}")
    return scene_locator(media_id=media_id, query=query, limit=limit)


@mcp.tool()
@check_tool_enabled(category="plex")
def find_scene_by_title(movie_title: str, scene_query: str, limit: int = 5):
    """Find a specific scene in a movie - convenience tool combining search and scene location."""
    if not PLEX_AVAILABLE:
        return plex_unavailable_error()

    logger.info(f"🛠 [server] find_scene_by_title called with movie_title: {movie_title}, query: {scene_query}, limit: {limit}")

    # Step 1: Search for the movie
    search_results = semantic_media_search(query=movie_title, limit=1)

    if not search_results.get("results"):
        return {"error": f"Could not find movie '{movie_title}' in Plex library"}

    # Step 2: Get the ratingKey
    media_id = search_results["results"][0]["id"]
    movie_name = search_results["results"][0]["title"]

    # Step 3: Find the scene
    scenes = scene_locator(media_id=media_id, query=scene_query, limit=limit)

    return {
        "movie": movie_name,
        "media_id": media_id,
        "scenes": scenes
    }


@mcp.tool()
@check_tool_enabled(category="plex")
def plex_find_unprocessed(limit: int = 5, rescan_no_subtitles: bool = False) -> str:
    """Find unprocessed Plex items that need ingestion."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"🔍 [TOOL] plex_find_unprocessed called (limit: {limit})")

    try:
        items = find_unprocessed_items(limit, rescan_no_subtitles)

        # Simplify for multi-agent consumption
        simplified = [
            {
                "id": str(item["id"]),
                "title": item.get("title", "Unknown"),
                "type": item.get("type", "unknown")
            }
            for item in items
        ]

        result = {
            "found_count": len(simplified),
            "items": simplified,
            "message": f"Found {len(simplified)} unprocessed items ready for ingestion"
        }

        logger.info(f"✅ [TOOL] Found {len(simplified)} unprocessed items")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ [TOOL] plex_find_unprocessed failed: {e}")
        return json.dumps({"error": str(e), "found_count": 0, "items": []})


@mcp.tool()
@check_tool_enabled(category="plex")
async def plex_ingest_items(item_ids: str) -> str:
    """Ingest multiple Plex items in parallel (ASYNC) with STOP SIGNAL support."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"🚀 [TOOL] plex_ingest_items called with IDs: {item_ids}")

    # Check stop BEFORE starting
    if is_stop_requested():
        logger.warning("🛑 plex_ingest_items: Stop requested - skipping ingestion")
        return json.dumps({
            "total_processed": 0,
            "ingested_count": 0,
            "skipped_count": 0,
            "stopped": True,
            "message": "Stopped before ingestion started"
        })

    try:
        # Check if using auto mode
        if item_ids.startswith("auto:"):
            limit = int(item_ids.split(":")[1])
            logger.info(f"🔍 Auto mode: finding {limit} unprocessed items")

            if is_stop_requested():
                logger.warning("🛑 Stopped during item discovery")
                return json.dumps({
                    "total_processed": 0,
                    "stopped": True,
                    "message": "Stopped during item discovery"
                })

            media_items = find_unprocessed_items(limit, False)
            if not media_items:
                return json.dumps({
                    "total_processed": 0,
                    "ingested_count": 0,
                    "skipped_count": 0,
                    "message": "No unprocessed items found"
                })
        else:
            # Parse comma-separated IDs
            ids_list = [id.strip() for id in item_ids.split(",") if id.strip()]

            if not ids_list:
                return json.dumps({"error": "No item IDs provided", "total_processed": 0})

            logger.info(f"🔍 Fetching {len(ids_list)} items from Plex")

            from tools.plex.plex_utils import get_plex_server
            plex = get_plex_server()
            media_items = []

            for item_id in ids_list:
                if is_stop_requested():
                    logger.warning(f"🛑 Stopped while fetching items ({len(media_items)} fetched so far)")
                    return json.dumps({
                        "total_processed": 0,
                        "items_fetched": len(media_items),
                        "stopped": True,
                        "message": f"Stopped while fetching items ({len(media_items)}/{len(ids_list)} fetched)"
                    })

                try:
                    item = plex.fetchItem(int(item_id))

                    media_item = {
                        "id": item_id,
                        "title": item.title,
                        "type": item.type,
                        "year": getattr(item, "year", None),
                    }

                    if item.type == "episode":
                        media_item["show_title"] = item.grandparentTitle
                        media_item["season"] = item.parentIndex
                        media_item["episode"] = item.index

                    media_items.append(media_item)
                    logger.info(f"✅ Fetched: {media_item['title']}")

                except Exception as e:
                    logger.error(f"❌ Failed to fetch item {item_id}: {e}")
                    media_items.append({
                        "id": item_id,
                        "title": f"Unknown Item {item_id}",
                        "type": "error",
                        "error": str(e)
                    })

        import asyncio
        import time

        if is_stop_requested():
            logger.warning("🛑 Stopped before processing items")
            return json.dumps({
                "total_processed": 0,
                "items_ready": len(media_items),
                "stopped": True,
                "message": f"Stopped before processing {len(media_items)} items"
            })

        start_time = time.time()
        logger.info(f"🚀 Processing {len(media_items)} items in parallel")

        results = await ingest_batch_parallel_conservative(media_items, target_success_count=limit)

        duration = time.time() - start_time
        stopped = is_stop_requested()

        ingested = [r for r in results if r.get("status") == "success"]
        skipped = [r for r in results if r.get("status") != "success"]

        summary = {
            "total_processed": len(results),
            "ingested_count": len(ingested),
            "skipped_count": len(skipped),
            "ingested": ingested,
            "skipped": skipped,
            "duration": round(duration, 2),
            "mode": "parallel",
            "concurrent_limit": 3,
            "stopped": stopped
        }

        if stopped:
            summary["stop_message"] = "Processing was stopped mid-execution"

        logger.info(f"✅ [TOOL] Batch complete: {len(ingested)} ingested, {len(skipped)} skipped in {duration:.2f}s (stopped={stopped})")
        return json.dumps(summary, indent=2)

    except Exception as e:
        logger.error(f"❌ [TOOL] plex_ingest_items failed: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e), "total_processed": 0})


@mcp.tool()
@check_tool_enabled(category="plex")
async def plex_ingest_single(media_id: str) -> str:
    """Ingest a single Plex item with STOP SIGNAL support."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"💾 [TOOL] plex_ingest_single called for media_id: {media_id}")

    if is_stop_requested():
        logger.warning("🛑 plex_ingest_single: Stop requested - skipping ingestion")
        return json.dumps({
            "title": f"Item {media_id}",
            "id": media_id,
            "status": "stopped",
            "reason": "Stopped before ingestion started"
        })

    try:
        if media_id == "auto" or media_id.startswith("auto"):
            logger.info("🔍 Auto mode: finding 1 unprocessed item")
            items = find_unprocessed_items(1, False)
            if not items:
                return json.dumps({
                    "title": "No items",
                    "id": "none",
                    "status": "error",
                    "reason": "No unprocessed items found"
                })
            media_item = items[0]
        else:
            from tools.plex.plex_utils import get_plex_server
            plex = get_plex_server()

            try:
                item = plex.fetchItem(int(media_id))
                media_item = {
                    "id": media_id,
                    "title": item.title,
                    "type": item.type,
                    "year": getattr(item, "year", None),
                }
            except Exception as e:
                logger.error(f"❌ Failed to fetch item {media_id}: {e}")
                return json.dumps({
                    "title": f"Item {media_id}",
                    "id": media_id,
                    "status": "error",
                    "reason": f"Could not fetch item: {str(e)}"
                })

        if is_stop_requested():
            logger.warning("🛑 Stopped before processing item")
            return json.dumps({
                "title": media_item.get("title", media_id),
                "id": media_id,
                "status": "stopped",
                "reason": "Stopped before processing"
            })

        logger.info(f"📥 Extracting subtitles for: {media_item.get('title', media_id)}")
        result = await process_item_async(media_item)

        logger.info(f"✅ [TOOL] Ingested: {result.get('title', 'unknown')}")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ [TOOL] plex_ingest_single failed for {media_id}: {e}")
        return json.dumps({
            "title": f"Item {media_id}",
            "id": media_id,
            "status": "error",
            "reason": str(e)
        })


@mcp.tool()
@check_tool_enabled(category="plex")
async def plex_ingest_batch(limit: int = 5, rescan_no_subtitles: bool = False) -> str:
    """Ingest the NEXT unprocessed Plex items into RAG (ALL-IN-ONE)."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"🛠 [TOOL] plex_ingest_batch called (limit: {limit})")

    try:
        result = await ingest_next_batch(limit, rescan_no_subtitles)
        logger.info(f"✅ [TOOL] plex_ingest_batch completed")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ [TOOL] plex_ingest_batch failed: {e}")
        import traceback
        traceback.print_exc()
        return json.dumps({"error": str(e)})


@mcp.tool()
@check_tool_enabled(category="plex")
def rag_rescan_no_subtitles() -> str:
    """Reset items that were marked as 'no subtitles' to allow re-scanning."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"🛠 [server] rag_rescan_no_subtitles called")
    from tools.rag.rag_storage import reset_no_subtitle_items
    count = reset_no_subtitle_items()
    return json.dumps({
        "reset_count": count,
        "message": f"Reset {count} items for re-scanning. Run plex_ingest_batch to check them again."
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="plex")
def plex_get_stats() -> str:
    """Get overall Plex ingestion statistics."""
    if not PLEX_AVAILABLE:
        return json.dumps(plex_unavailable_error(), indent=2)

    logger.info(f"📊 [TOOL] plex_get_stats called")

    try:
        from tools.rag.rag_storage import get_ingestion_stats

        stats = get_ingestion_stats()

        result = {
            "total_items": stats["total_items"],
            "successfully_ingested": stats["successfully_ingested"],
            "missing_subtitles": stats["missing_subtitles"],
            "remaining_unprocessed": stats["remaining"],
            "completion_percentage": round(
                (stats["successfully_ingested"] / stats["total_items"] * 100)
                if stats["total_items"] > 0 else 0,
                1
            )
        }

        logger.info(f"📊 [TOOL] Stats: {result['completion_percentage']}% complete")
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ [TOOL] plex_get_stats failed: {e}")
        return json.dumps({"error": str(e)})


skill_registry = None

@mcp.tool()
@check_tool_enabled(category="plex")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "plex-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "plex-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="plex")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called")

    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)

    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content

    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({
        "error": f"Skill '{skill_name}' not found",
        "available_skills": available
    }, indent=2)


def get_tool_names_from_module():
    """Extract all function names from current module (auto-discovers tools)"""
    current_module = sys.modules[__name__]
    tool_names = []

    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name not in ['get_tool_names_from_module', 'plex_unavailable_error']:
                tool_names.append(name)

    return tool_names


# ============================================================================
# ML RECOMMENDATION TOOLS (all with availability checks)
# ============================================================================

@mcp.tool()
@check_tool_enabled(category="plex")
def import_plex_history(limit: int = 50) -> dict:
    """Automatically import your Plex viewing history into the ML recommender"""
    if not PLEX_AVAILABLE:
        return plex_unavailable_error()

    logger.info(f"📥 Importing Plex viewing history (limit: {limit})")

    try:
        from tools.plex.plex_utils import get_plex_server

        plex = get_plex_server()
        recommender = get_recommender()

        # ... rest of your existing import_plex_history code ...
        # (keep all the existing logic)

    except Exception as e:
        logger.error(f"❌ Failed to import Plex history: {e}")
        return {
            "message": f"❌ Error importing Plex history: {str(e)}",
            "imported": 0,
            "error": str(e)
        }


# Apply the same pattern to ALL remaining ML tools:
# - auto_train_from_plex
# - record_viewing
# - train_recommender
# - recommend_content
# - get_recommender_stats
# - reset_recommender
# - auto_recommend_from_plex

# Just add `if not PLEX_AVAILABLE: return plex_unavailable_error()`
# at the start of each function

# ... (rest of your ML tools with the same pattern) ...


if __name__ == "__main__":
    # Auto-extract tool names - NO manual list needed!
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools)
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")

    if PLEX_AVAILABLE:
        logger.info("✅ Plex server starting with full functionality")
    else:
        logger.warning("⚠️  Plex server starting with limited functionality (set PLEX_URL and PLEX_TOKEN)")

    mcp.run(transport="stdio")