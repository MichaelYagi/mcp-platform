"""
Plex Ingestion Tool
Batch embedding generation and database operations for improved performance
"""

import json
import logging
import asyncio
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

from dotenv import load_dotenv

from tools.rag.rag_storage import check_if_ingested, mark_as_ingested, get_ingestion_stats
from tools.rag.rag_vector_db import add_to_rag_batch, flush_batch, embeddings_model
from client.stop_signal import is_stop_requested, clear_stop, get_stop_status
from .plex_utils import stream_all_media, extract_metadata, stream_subtitles, chunk_stream

logger = logging.getLogger("mcp_server")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)
PROGRESS_FILE = PROJECT_ROOT / "data" / "plex_ingest_progress.json"
PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Conservative concurrency limit (safe for most systems)
CONCURRENT_LIMIT = int(os.getenv("CONCURRENT_LIMIT", 1))

# Embedding batch size for parallel generation
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", 10))

# Database flush batch size (chunks per flush)
DB_FLUSH_BATCH_SIZE = int(os.getenv("DB_FLUSH_BATCH_SIZE", 30))

def load_progress() -> Dict[str, bool]:
    """Load ingestion progress from disk"""
    if not PROGRESS_FILE.exists():
        return {}

    try:
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}


def save_progress(progress: Dict[str, bool]) -> None:
    """Save ingestion progress to disk"""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)


# ============================================================================
# Batch Embedding Generation
# ============================================================================

async def generate_embeddings_batch(texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in parallel batches with stop signal support.

    Args:
        texts: List of text chunks to embed
        batch_size: Number of parallel embedding requests (default: 50)

    Returns:
        List of embeddings in same order as input texts

    Raises:
        Exception: If stop signal is received or embedding generation fails
    """
    loop = asyncio.get_event_loop()
    embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info(f"🔮 Generating embeddings for {len(texts)} chunks in batches of {batch_size}...")

    # Process in batches to avoid overwhelming Ollama
    for batch_num, i in enumerate(range(0, len(texts), batch_size), 1):
        # ═══════════════════════════════════════════════════════════
        # STOP CHECK: Before each embedding batch
        # ═══════════════════════════════════════════════════════════
        if is_stop_requested():
            completed = len(embeddings)
            remaining = len(texts) - completed
            logger.info(f"🛑 Embedding generation stopped at batch {batch_num}/{total_batches}")
            logger.info(f"🛑 Generated {completed}/{len(texts)} embeddings ({remaining} stopped)")
            # Return empty to signal clean stop (no partial data)
            return []

        batch = texts[i:i + batch_size]
        batch_size_actual = len(batch)

        # Generate embeddings in parallel using thread pool
        # Use return_exceptions=True to handle individual failures
        tasks = [
            loop.run_in_executor(None, embeddings_model.embed_query, text)
            for text in batch
        ]

        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check for errors and handle them
        for idx, result in enumerate(batch_results):
            # Quick stop check after each embedding completes
            if is_stop_requested():
                completed = len(embeddings)
                remaining = len(texts) - completed
                logger.warning(f"🛑 Embedding generation stopped mid-batch")
                logger.warning(f"🛑 Generated {completed}/{len(texts)} embeddings ({remaining} stopped)")
                raise Exception(f"Embedding generation stopped by user ({completed}/{len(texts)} completed)")

            if isinstance(result, Exception):
                text_preview = batch[idx][:100] + "..." if len(batch[idx]) > 100 else batch[idx]
                logger.error(f"❌ Failed to generate embedding for chunk {i + idx}: {result}")
                logger.error(f"   Text preview: {text_preview}")
                logger.error(f"   Text length: {len(batch[idx])} chars, ~{len(batch[idx].split())} words")
                # Raise the exception to stop processing
                raise Exception(f"Embedding failed for chunk {i + idx}: {result}")
            else:
                embeddings.append(result)

        logger.debug(
            f"📊 Batch {batch_num}/{total_batches}: Generated {batch_size_actual} embeddings (total: {len(embeddings)}/{len(texts)})")

    logger.info(f"✅ Embedding generation complete: {len(embeddings)}/{len(texts)} embeddings")
    return embeddings


# ============================================================================
# PARALLELIZABLE FUNCTIONS (UNCHANGED)
# ============================================================================

def find_unprocessed_items(target_success_count: int, rescan_no_subtitles: bool = False) -> List[Dict[str, Any]]:
    """
    STEP 1: Find unprocessed media items (with buffer for failures)

    Args:
        target_success_count: Target number of SUCCESSFUL ingestions we want
        rescan_no_subtitles: Whether to re-check items with no subtitles

    Returns:
        List of unprocessed media items (up to target * 3 to account for failures)
    """
    unprocessed_items = []
    checked_count = 0

    # Find 3x the target to handle failures/skips
    buffer_multiplier = 3
    search_limit = target_success_count * buffer_multiplier

    logger.info(
        f"🔍 Finding up to {search_limit} unprocessed items (target: {target_success_count} successful, rescan: {rescan_no_subtitles})")

    for media_item in stream_all_media():
        # CHECK STOP SIGNAL during search
        if is_stop_requested():
            logger.warning(f"🛑 Stop requested during search after checking {checked_count} items")
            break

        media_id = str(media_item["id"])
        title = media_item["title"]

        # Check if already ingested
        if check_if_ingested(media_id, skip_no_subtitles=rescan_no_subtitles):
            checked_count += 1
            logger.debug(f"⏭️  [{checked_count}] Already processed: {title}")
            continue

        # Found unprocessed item
        logger.info(f"📍 Found unprocessed: {title}")
        unprocessed_items.append(media_item)

        # Stop when we have enough buffer
        if len(unprocessed_items) >= search_limit:
            logger.info(f"📦 Buffer filled: found {search_limit} items for {target_success_count} target")
            break

    logger.info(
        f"🔍 Found {len(unprocessed_items)} unprocessed items (checked {checked_count + len(unprocessed_items)} total)")
    return unprocessed_items


def extract_subtitles_for_item(media_item: Dict[str, Any]) -> Tuple[str, str, List[str], str]:
    """
    STEP 2: Extract subtitles for a single item (parallelizable)

    NOTE: This function is BLOCKING and runs in a thread pool.
    It CANNOT be interrupted mid-execution.
    Stop checks happen BEFORE and AFTER this function is called.

    Args:
        media_item: Media item dictionary

    Returns:
        Tuple of (media_id, title, subtitle_lines, metadata_text)
    """
    media_id = str(media_item["id"])
    title = media_item["title"]

    logger.info(f"📥 Extracting subtitles for: {title}")

    # Get metadata
    metadata_text = extract_metadata(media_item)

    # Stream subtitles
    subtitle_lines = list(stream_subtitles(media_id))

    if not subtitle_lines:
        logger.warning(f"⚠️  No subtitles found for: {title}")
    else:
        logger.info(f"✅ Extracted {len(subtitle_lines)} subtitle lines for: {title}")

    return media_id, title, subtitle_lines, metadata_text


# ============================================================================
# BATCHED INGESTION FUNCTION
# ============================================================================

async def ingest_item_to_rag(
        media_id: str,
        title: str,
        subtitle_lines: List[str],
        metadata_text: str
) -> Dict[str, Any]:
    """
    Ingest a single item's subtitles into RAG with batched operations.

    Args:
        media_id: Plex media ID
        title: Media title
        subtitle_lines: List of subtitle text lines
        metadata_text: Metadata description

    Returns:
        Dictionary with ingestion results
    """
    if not subtitle_lines:
        mark_as_ingested(media_id, status="no_subtitles")
        return {
            "title": title,
            "id": media_id,
            "subtitle_chunks": 0,
            "subtitle_word_count": 0,
            "status": "no_subtitles",
            "reason": "No subtitles found"
        }

    logger.info(f"💾 Ingesting {title} to RAG...")
    ingestion_start = time.time()

    # ═══════════════════════════════════════════════════════════
    # Step 1: Chunk all text
    # ═══════════════════════════════════════════════════════════
    all_text_chunks = []
    # BGE-large has 512 token limit
    # For very dense subtitle content (HTML, metadata, etc), need aggressive limits
    # Safe estimate: ~2 chars per token for worst case
    # 512 tokens * 2 chars = 1024 chars max
    # Use 1000 chars to be safe
    max_chunk_chars = 1000

    for chunk in chunk_stream(iter(subtitle_lines), chunk_size=1600):
        # Still check stop, but only between chunks (not for each line)
        if is_stop_requested():
            logger.warning(f"🛑 Stopped during chunking of {title}")
            mark_as_ingested(media_id, status="partial")
            return {
                "title": title,
                "id": media_id,
                "subtitle_chunks": 0,
                "subtitle_word_count": 0,
                "status": "stopped",
                "reason": "Stopped during chunking"
            }

        # Validate chunk size - if too large, split it further
        if len(chunk) > max_chunk_chars:
            logger.debug(f"📏 Chunk too large ({len(chunk)} chars), splitting to max {max_chunk_chars}...")
            # Split into smaller chunks by words
            words = chunk.split()
            current_chunk = []
            current_length = 0

            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_chunk_chars:
                    if current_chunk:
                        all_text_chunks.append(" ".join(current_chunk))
                        current_chunk = [word]
                        current_length = word_length
                else:
                    current_chunk.append(word)
                    current_length += word_length

            if current_chunk:
                all_text_chunks.append(" ".join(current_chunk))
        else:
            all_text_chunks.append(chunk)

    logger.info(f"📦 Created {len(all_text_chunks)} text chunks (max {max_chunk_chars} chars each)")

    # ═══════════════════════════════════════════════════════════
    # Step 2: Generate embeddings in parallel
    # ═══════════════════════════════════════════════════════════
    # Final safety check: verify all chunks are within safe limits
    max_safe_chars = 1000
    validated_chunks = []
    for i, chunk in enumerate(all_text_chunks):
        if len(chunk) > max_safe_chars:
            logger.error(f"❌ Chunk {i} still too large ({len(chunk)} chars)! Truncating...")
            # Emergency truncation - should never happen but better than failing
            validated_chunks.append(chunk[:max_safe_chars])
        else:
            validated_chunks.append(chunk)

    if len(validated_chunks) != len(all_text_chunks):
        logger.warning(f"⚠️  Chunk count mismatch after validation!")

    logger.info(f"🔮 Generating embeddings for {len(validated_chunks)} chunks in batches of {EMBEDDING_BATCH_SIZE}...")

    try:
        embeddings = await generate_embeddings_batch(validated_chunks, batch_size=EMBEDDING_BATCH_SIZE)
    except Exception as e:
        # Embedding generation failed or was stopped
        if "stopped by user" in str(e).lower():
            logger.warning(f"🛑 Stopped during embedding generation for {title}")
            mark_as_ingested(media_id, status="partial")
            return {
                "title": title,
                "id": media_id,
                "subtitle_chunks": 0,
                "subtitle_word_count": 0,
                "status": "stopped",
                "reason": "Stopped during embedding generation"
            }
        else:
            # Real error
            logger.error(f"❌ Embedding generation failed for {title}: {e}")
            mark_as_ingested(media_id, status="error")
            return {
                "title": title,
                "id": media_id,
                "subtitle_chunks": 0,
                "subtitle_word_count": 0,
                "status": "error",
                "reason": f"Embedding generation failed: {str(e)}"
            }

    # ═══════════════════════════════════════════════════════════
    # CRITICAL CHECK: Verify embeddings are complete
    # ═══════════════════════════════════════════════════════════
    if not embeddings or len(embeddings) != len(validated_chunks):
        logger.warning(f"🛑 Incomplete embeddings for {title} ({len(embeddings)}/{len(validated_chunks)})")
        mark_as_ingested(media_id, status="partial")
        return {
            "title": title,
            "id": media_id,
            "subtitle_chunks": 0,
            "subtitle_word_count": 0,
            "status": "stopped",
            "reason": f"Incomplete embeddings ({len(embeddings)}/{len(validated_chunks)} generated)"
        }

    # Final stop check before committing to database
    if is_stop_requested():
        logger.warning(f"🛑 Stopped after embedding generation for {title} - not saving to database")
        mark_as_ingested(media_id, status="partial")
        return {
            "title": title,
            "id": media_id,
            "subtitle_chunks": 0,
            "subtitle_word_count": 0,
            "status": "stopped",
            "reason": "Stopped before database write"
        }

    # ═══════════════════════════════════════════════════════════
    # Step 3: Batch database writes
    # ═══════════════════════════════════════════════════════════
    chunks_added = 0
    word_count = 0
    source = f"plex:{media_id}:{title}"

    logger.info(f"💾 Adding {len(validated_chunks)} chunks to RAG database...")

    # Process chunks in batches for database flushing
    for i in range(0, len(validated_chunks), DB_FLUSH_BATCH_SIZE):
        if is_stop_requested():
            logger.warning(f"🛑 Stopped during database write for {title} after {chunks_added} chunks")
            # Flush what we have so far
            flush_batch()
            mark_as_ingested(media_id, status="partial")
            return {
                "title": title,
                "id": media_id,
                "subtitle_chunks": chunks_added,
                "subtitle_word_count": word_count,
                "status": "stopped",
                "reason": f"Stopped during database write after {chunks_added} chunks"
            }

        # Get batch of chunks
        batch_end = min(i + DB_FLUSH_BATCH_SIZE, len(validated_chunks))
        batch_chunks = validated_chunks[i:batch_end]
        batch_embeddings = embeddings[i:batch_end]

        # Add each chunk to the pending batch (fast, in-memory)
        # We need to directly access the module's _pending_chunks list
        import tools.rag.rag_vector_db as rag_db

        for text_chunk, embedding in zip(batch_chunks, batch_embeddings):
            # Manually add to pending batch with pre-computed embedding
            import uuid
            doc = {
                "id": str(uuid.uuid4()),
                "text": text_chunk,
                "embedding": embedding,
                "metadata": {
                    "source": source,
                    "length": len(text_chunk),
                    "word_count": len(text_chunk.split())
                }
            }

            rag_db._pending_chunks.append(doc)

            chunks_added += 1
            word_count += len(text_chunk.split())

        # Flush this batch to database (one write for 30 chunks)
        flush_batch()

        logger.info(
            f"✅ Added batch {i // DB_FLUSH_BATCH_SIZE + 1}/{(len(validated_chunks) + DB_FLUSH_BATCH_SIZE - 1) // DB_FLUSH_BATCH_SIZE} ({batch_end - i} chunks)")

    # ═══════════════════════════════════════════════════════════
    # Handle metadata (small, can be single chunk)
    # ═══════════════════════════════════════════════════════════
    metadata_summary = f"{title} - {metadata_text}"
    if len(metadata_summary) < 1600:
        if is_stop_requested():
            logger.warning(f"🛑 Stopped before adding metadata for {title}")
            mark_as_ingested(media_id, status="partial")
            return {
                "title": title,
                "id": media_id,
                "subtitle_chunks": chunks_added,
                "subtitle_word_count": word_count,
                "status": "stopped",
                "reason": f"Stopped before metadata (ingested {chunks_added} chunks)"
            }

        try:
            # Generate embedding for metadata
            loop = asyncio.get_event_loop()
            metadata_embedding = await loop.run_in_executor(None, embeddings_model.embed_query, metadata_summary)

            import uuid
            import tools.rag.rag_vector_db as rag_db

            doc = {
                "id": str(uuid.uuid4()),
                "text": metadata_summary,
                "embedding": metadata_embedding,
                "metadata": {
                    "source": f"plex:{media_id}:metadata",
                    "length": len(metadata_summary),
                    "word_count": len(metadata_summary.split())
                }
            }

            rag_db._pending_chunks.append(doc)
            chunks_added += 1

            # Flush metadata
            flush_batch()
        except Exception as e:
            logger.warning(f"⚠️  Could not add metadata chunk: {e}")

    mark_as_ingested(media_id, status="success")

    ingestion_duration = time.time() - ingestion_start
    logger.info(f"✅ Ingested: {title} ({chunks_added} chunks, ~{word_count} words) in {ingestion_duration:.1f}s")

    return {
        "title": title,
        "id": media_id,
        "subtitle_chunks": chunks_added,
        "subtitle_word_count": word_count,
        "status": "success",
        "duration": round(ingestion_duration, 1)
    }


# ============================================================================
# Async Pipeline
# ============================================================================

async def process_item_async(media_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single item asynchronously (extract + ingest).
    Includes stop checks before and after each blocking operation.

    Args:
        media_item: Media item to process

    Returns:
        Ingestion result dictionary
    """
    loop = asyncio.get_event_loop()

    try:
        title = media_item.get("title", "Unknown")
        media_id = str(media_item.get("id", "Unknown"))

        # ═══════════════════════════════════════════════════════════
        # STOP CHECK #1: Before starting extraction
        # ═══════════════════════════════════════════════════════════
        if is_stop_requested():
            logger.warning(f"🛑 [STOP CHECK #1] Stopped before extracting: {title}")
            return {
                "title": title,
                "id": media_id,
                "status": "stopped",
                "reason": "Stopped before extraction"
            }

        logger.info(f"📥 Starting extraction for: {title}")
        extraction_start = time.time()

        # Run extraction in thread pool (BLOCKING - cannot interrupt mid-extraction)
        media_id, title, subtitle_lines, metadata_text = await loop.run_in_executor(
            None, extract_subtitles_for_item, media_item
        )

        extraction_duration = time.time() - extraction_start
        logger.info(f"✅ Extraction complete for: {title} ({extraction_duration:.1f}s)")

        # ═══════════════════════════════════════════════════════════
        # STOP CHECK #2: After extraction, before ingestion
        # ═══════════════════════════════════════════════════════════
        if is_stop_requested():
            logger.warning(f"🛑 [STOP CHECK #2] Stopped after extraction, before ingestion: {title}")
            return {
                "title": title,
                "id": media_id,
                "status": "stopped",
                "reason": "Stopped after extraction, before ingestion"
            }

        logger.info(f"💾 Starting ingestion for: {title}")
        ingestion_start = time.time()

        # Run ingestion with batched operations
        result = await ingest_item_to_rag(
            media_id, title, subtitle_lines, metadata_text
        )

        ingestion_duration = time.time() - ingestion_start
        logger.info(f"✅ Ingestion complete for: {title} ({ingestion_duration:.1f}s)")

        # ═══════════════════════════════════════════════════════════
        # STOP CHECK #3: After ingestion (check if it was stopped internally)
        # ═══════════════════════════════════════════════════════════
        if result.get("status") == "stopped":
            logger.warning(f"🛑 [STOP CHECK #3] Ingestion was stopped internally: {title}")

        return result

    except Exception as e:
        logger.error(f"❌ Failed to process item: {e}")
        import traceback
        traceback.print_exc()
        return {
            "title": media_item.get("title", "Unknown"),
            "id": str(media_item.get("id", "Unknown")),
            "status": "error",
            "reason": str(e)
        }


# ============================================================================
# Batch Processing
# ============================================================================

async def ingest_batch_parallel_conservative(
        items: List[Dict[str, Any]],
        target_success_count: int
) -> List[Dict[str, Any]]:
    """
    Process items until target_success_count successful ingestions are completed.

    Stop checks:
    - Before each batch starts
    - After each batch completes
    - When target is reached

    Args:
        items: Pool of media items to process
        target_success_count: How many SUCCESSFUL ingestions we want

    Returns:
        List of all ingestion results (successful + failed + stopped)
    """
    results = []
    successful_count = 0
    items_index = 0
    total_items = len(items)

    logger.info(
        f"🎯 Target: {target_success_count} successful ingestions from pool of {total_items} items ({CONCURRENT_LIMIT} concurrent)")
    overall_start = time.time()

    batch_num = 0

    # Process until we reach target or run out of items
    while successful_count < target_success_count and items_index < total_items:
        batch_num += 1

        # ═══════════════════════════════════════════════════════════
        # STOP CHECK: Before each batch
        # ═══════════════════════════════════════════════════════════
        if is_stop_requested():
            logger.warning(
                f"🛑 [BATCH STOP] Stopped after {successful_count}/{target_success_count} successful ingestions")

            # Mark all remaining items as stopped
            for remaining_idx in range(items_index, total_items):
                remaining_item = items[remaining_idx]
                results.append({
                    "status": "stopped",
                    "title": remaining_item.get("title", "Unknown"),
                    "message": "Stopped before processing",
                })

            break

        # Get next batch
        batch = items[items_index:items_index + CONCURRENT_LIMIT]
        batch_size = len(batch)
        items_index += len(batch)

        logger.info(
            f"⚙️  Processing batch {batch_num} ({batch_size} items)... [{successful_count}/{target_success_count} successful so far]")
        batch_start = time.time()

        # Process this batch in parallel
        batch_results = await asyncio.gather(
            *[process_item_async(item) for item in batch],
            return_exceptions=True
        )

        batch_duration = time.time() - batch_start
        logger.info(f"✅ Batch {batch_num} completed in {batch_duration:.2f}s")

        # Handle results and count successes
        batch_was_stopped = False

        for idx, result in enumerate(batch_results):
            if isinstance(result, Exception):
                logger.error(f"❌ Batch task failed: {result}")
                results.append({
                    "status": "error",
                    "title": batch[idx].get("title", "Unknown"),
                    "reason": str(result)
                })
            else:
                results.append(result)

                # Count successful ingestions
                if result.get("status") == "success":
                    successful_count += 1
                    logger.info(f"✅ Progress: {successful_count}/{target_success_count} successful ingestions")

                    # Check if we hit target
                    if successful_count >= target_success_count:
                        logger.info(f"🎯 Target reached! {successful_count}/{target_success_count} successful")
                        batch_was_stopped = True
                        break

                elif result.get("status") == "stopped":
                    logger.warning(f"🛑 [ITEM STOP] Item '{result.get('title')}' was stopped")
                    batch_was_stopped = True

                elif result.get("status") in ["no_subtitles", "error"]:
                    logger.warning(f"⏭️  Skipped: {result.get('title')} ({result.get('status')})")

        # ═══════════════════════════════════════════════════════════
        # STOP CHECK: After batch or target reached
        # ═══════════════════════════════════════════════════════════
        if batch_was_stopped or is_stop_requested() or successful_count >= target_success_count:
            # Mark any remaining items as stopped (if we haven't processed them yet)
            if items_index < total_items:
                remaining_count = total_items - items_index
                logger.info(f"🛑 Stopping early - marking {remaining_count} remaining items as not attempted")

                for remaining_idx in range(items_index, total_items):
                    remaining_item = items[remaining_idx]
                    results.append({
                        "status": "not_attempted",
                        "title": remaining_item.get("title", "Unknown"),
                        "message": "Target reached before this item"
                    })

            break

        # Brief pause between batches
        if items_index < total_items and successful_count < target_success_count:
            await asyncio.sleep(0.1)

    overall_duration = time.time() - overall_start
    avg_rate = successful_count / overall_duration if overall_duration > 0 else 0

    # Summary
    failed_count = sum(1 for r in results if r.get("status") in ["error", "no_subtitles"])
    stopped_count = sum(1 for r in results if r.get("status") == "stopped")

    logger.info(f"🏁 Parallel ingestion completed:")
    logger.info(f"   - Target: {target_success_count}")
    logger.info(f"   - Successful: {successful_count}")
    logger.info(f"   - Failed/Skipped: {failed_count}")
    logger.info(f"   - Stopped: {stopped_count}")
    logger.info(
        f"   - Total attempted: {len([r for r in results if r.get('status') not in ['not_attempted', 'stopped']])}")
    logger.info(f"   - Duration: {overall_duration:.2f}s ({avg_rate:.2f} items/sec)")

    return results


async def ingest_next_batch(limit: int = 5, rescan_no_subtitles: bool = False) -> Dict[str, Any]:
    """
    Ingest items until LIMIT successful ingestions are completed.

    Includes stop signal handling:
    - Checks stop during item search
    - Checks stop before/after each batch
    - Checks stop when target is reached
    - Reports stop status in results

    Args:
        limit: Number of SUCCESSFUL ingestions to complete (not total attempts)
        rescan_no_subtitles: If True, re-check items that previously had no subtitles

    Returns:
        Dictionary with ingestion results (includes "stopped" flag if stopped)
    """
    try:
        logger.info(f"📥 Starting parallel batch ingestion (target: {limit} successful, rescan: {rescan_no_subtitles})")
        overall_start = time.time()

        # STEP 1: Find unprocessed items (with 3x buffer for failures)
        loop = asyncio.get_event_loop()
        unprocessed_items = await loop.run_in_executor(
            None, find_unprocessed_items, limit, rescan_no_subtitles
        )

        # Check if search was stopped
        if is_stop_requested():
            logger.warning("🛑 Search was stopped - returning early")
            return {
                "target": limit,
                "successful": 0,
                "failed_skipped": 0,
                "stopped": True,
                "stop_reason": "Stopped during item search",
                "duration": time.time() - overall_start,
                "mode": "parallel"
            }

        if not unprocessed_items:
            logger.info("✅ No unprocessed items found")
            stats = get_ingestion_stats()
            return {
                "target": limit,
                "successful": 0,
                "failed_skipped": 0,
                "stopped": False,
                "stats": {
                    "total_items": stats["total_items"],
                    "successfully_ingested": stats["successfully_ingested"],
                    "missing_subtitles": stats["missing_subtitles"],
                    "remaining_unprocessed": stats["remaining"]
                },
                "message": "No unprocessed items found",
                "duration": 0,
                "mode": "parallel"
            }

        # STEP 2 & 3: Process items until target is reached
        logger.info(f"🚀 Processing {len(unprocessed_items)} items with batched operations...")

        results = await ingest_batch_parallel_conservative(
            unprocessed_items,
            target_success_count=limit  # ADDED: Pass target count
        )

        # Categorize results
        successful_items = []
        failed_items = []
        was_stopped = False
        stop_reason = None

        for result in results:
            status = result.get("status")

            if status == "success":
                successful_items.append(result)
            elif status == "stopped":
                was_stopped = True
                stop_reason = result.get("message", "Stopped by user")
                failed_items.append({
                    "title": result.get("title", "Unknown"),
                    "reason": "Stopped before processing"
                })
            elif status == "not_attempted":
                # Don't count as failed - just not attempted because target was reached
                pass
            elif status in ["no_subtitles", "error"]:
                failed_items.append({
                    "title": result.get("title", "Unknown"),
                    "id": result.get("id", "Unknown"),
                    "reason": result.get("reason", status)
                })

        # Get total stats
        stats = get_ingestion_stats()
        overall_duration = time.time() - overall_start

        # Count only items that were actually attempted (not "not_attempted")
        items_attempted = len([r for r in results if r.get("status") not in ["not_attempted", "stopped"]])

        result = {
            "target": limit,
            "successful": len(successful_items),
            "failed_skipped": len(failed_items),
            "total_attempted": items_attempted,
            "target_reached": len(successful_items) >= limit,
            "stopped": was_stopped,
            "stop_reason": stop_reason,
            "stats": {
                "total_items": stats["total_items"],
                "successfully_ingested": stats["successfully_ingested"],
                "missing_subtitles": stats["missing_subtitles"],
                "remaining_unprocessed": stats["remaining"]
            },
            "successful_items": [
                {
                    "title": item["title"],
                    "chunks": item.get("subtitle_chunks", 0),
                    "words": item.get("subtitle_word_count", 0)
                }
                for item in successful_items
            ],
            "failed_items": failed_items,
            "duration": round(overall_duration, 2),
            "mode": "parallel",
            "concurrent_limit": CONCURRENT_LIMIT,
            "embedding_batch_size": EMBEDDING_BATCH_SIZE,
            "db_flush_batch_size": DB_FLUSH_BATCH_SIZE,
            "rate": round(len(successful_items) / overall_duration, 2) if overall_duration > 0 else 0
        }

        # Log final status
        if was_stopped:
            logger.warning(f"🛑 Batch stopped by user:")
            logger.warning(f"   - Reason: {stop_reason}")
            logger.warning(f"   - Successful: {len(successful_items)}/{limit}")
        else:
            logger.info(f"📊 Batch complete:")
            logger.info(f"   - Target: {limit}")
            logger.info(f"   - Successful: {len(successful_items)}")
            logger.info(f"   - Failed/Skipped: {len(failed_items)}")
            logger.info(f"   - Total attempted: {items_attempted}")
            logger.info(f"   - Duration: {overall_duration:.2f}s")
            logger.info(f"   - Rate: {result['rate']:.2f} items/sec")

            if len(successful_items) >= limit:
                logger.info(f"   🎯 Target reached!")

        return result

    except Exception as e:
        logger.error(f"❌ Parallel ingestion error: {e}")
        import traceback
        traceback.print_exc()

        # Check if this was due to a stop
        stop_status = get_stop_status()
        if stop_status["is_stopped"]:
            return {
                "target": limit,
                "successful": 0,
                "error": "Stopped during execution",
                "stopped": True,
                "stop_reason": str(e),
                "mode": "parallel"
            }

        return {
            "target": limit,
            "successful": 0,
            "error": str(e),
            "stopped": False,
            "mode": "parallel"
        }