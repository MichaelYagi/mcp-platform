import os
import logging
from typing import Dict, Any, List, Iterator
from plexapi.server import PlexServer
from pathlib import Path

logger = logging.getLogger("mcp_server")

# Plex connection - using existing env var
BASE_URL = os.getenv("PLEX_URL")
TOKEN = os.getenv("PLEX_TOKEN")

if not TOKEN or not BASE_URL:
    logger.warning("⚠️ PLEX_URL and PLEX_TOKEN must be set in environment")

_plex = None


def get_plex_server():
    """Get or create Plex server connection"""
    global _plex
    if _plex is None:
        _plex = PlexServer(BASE_URL, TOKEN)
    return _plex


def stream_all_media() -> Iterator[Dict[str, Any]]:
    """
    Stream all movies and TV shows from Plex library.

    Yields:
        Dictionary with media information
    """
    try:
        plex = get_plex_server()

        # Get all library sections
        for section in plex.library.sections():
            if section.type in ['movie', 'show']:
                logger.info(f"📚 Scanning section: {section.title}")

                for item in section.all():
                    # For TV shows, process each episode
                    if section.type == 'show':
                        for episode in item.episodes():
                            yield {
                                "id": episode.ratingKey,
                                "title": f"{item.title} - {episode.seasonEpisode} - {episode.title}",
                                "type": "episode",
                                "show_title": item.title,
                                "episode_title": episode.title,
                                "season": episode.seasonNumber,
                                "episode_number": episode.episodeNumber,
                                "summary": episode.summary or "",
                                "year": episode.year,
                            }
                    # For movies
                    else:
                        yield {
                            "id": item.ratingKey,
                            "title": item.title,
                            "type": "movie",
                            "summary": item.summary or "",
                            "year": item.year,
                            "genres": [g.tag for g in item.genres] if hasattr(item, 'genres') else [],
                        }

    except Exception as e:
        logger.error(f"❌ Error streaming media: {e}")
        raise


def extract_metadata(item: Dict[str, Any]) -> str:
    """
    Extract metadata as a text string.

    Args:
        item: Media item dictionary

    Returns:
        Formatted metadata string
    """
    if item["type"] == "episode":
        return f"""
Title: {item['title']}
Show: {item['show_title']}
Episode: {item['episode_title']}
Season {item['season']}, Episode {item['episode_number']}
Year: {item['year']}
Summary: {item.get('summary', '')}
""".strip()
    else:
        genres = ", ".join(item.get('genres', []))
        return f"""
Title: {item['title']}
Type: Movie
Year: {item['year']}
Genres: {genres}
Summary: {item.get('summary', '')}
""".strip()

def stream_subtitles(rating_key: str) -> Iterator[str]:
    """
    Stream subtitle lines for a given media item.

    Selection rules:
    - Only process English subtitle streams
    - Try all English streams until one works
    - If no English streams work, skip the item
    """
    try:
        plex = get_plex_server()
        media = plex.fetchItem(int(rating_key))

        for part in media.iterParts():

            # ---------------------------------------------------------
            # 1. Collect English text subtitle streams only
            # ---------------------------------------------------------
            english_streams = []

            for stream in part.subtitleStreams():
                if stream.streamType != 3:
                    continue

                # Only text-based subtitles
                if stream.codec not in ("srt", "ass", "vtt"):
                    continue

                # Check if English
                lang = (
                        stream.languageCode
                        or stream.language
                        or stream.languageTag
                        or stream.displayTitle
                        or ""
                ).lower()

                if lang in ("eng", "en", "english"):
                    english_streams.append(stream)

            if not english_streams:
                logger.warning(f"⚠️ No English subtitle streams found for: {media.title}")
                return

            # ---------------------------------------------------------
            # 2. Try each English stream until one works
            # ---------------------------------------------------------
            for chosen in english_streams:
                logger.info(f"📝 Trying subtitle stream: {chosen.displayTitle or chosen.title or 'Untitled'}")

                # Try external download (if key exists)
                content = None
                if getattr(chosen, "key", None):
                    try:
                        subtitle_url = plex.url(chosen.key, includeToken=True)
                        import requests
                        response = requests.get(subtitle_url)

                        if response.status_code == 200 and response.text.strip():
                            content = response.text
                            logger.info(f"✅ Downloaded subtitle via key")
                    except Exception as e:
                        logger.warning(f"⚠️ Download failed: {e}")

                # If download worked, parse and return
                if content:
                    for line in parse_srt(content):
                        yield line
                    return  # Success! Exit after first working stream
                else:
                    logger.warning(f"⚠️ Stream has no downloadable content, trying next English stream...")
                    # Continue to next English stream

            # If we got here, no English streams worked
            logger.warning(f"⚠️ No extractable English subtitle content found for: {media.title}")
            return

    except Exception as e:
        logger.error(f"❌ Error streaming subtitles for {rating_key}: {e}")

def parse_srt(content: str) -> List[str]:
    """
    Parse SRT subtitle format.

    Args:
        content: SRT file content

    Returns:
        List of subtitle text lines (without timestamps)
    """
    lines = []
    current_text = []

    for line in content.split('\n'):
        line = line.strip()

        # Skip empty lines
        if not line:
            if current_text:
                lines.append(' '.join(current_text))
                current_text = []
            continue

        # Skip sequence numbers
        if line.isdigit():
            continue

        # Skip timestamp lines
        if '-->' in line:
            continue

        # Collect text
        current_text.append(line)

    # Add last text if any
    if current_text:
        lines.append(' '.join(current_text))

    return lines


def chunk_stream(lines: Iterator[str], chunk_size: int = 1600) -> Iterator[str]:
    """
    Chunk text lines into larger blocks by character count.

    Args:
        lines: Iterator of text lines
        chunk_size: Maximum characters per chunk (default: 4000)

    Yields:
        Text chunks
    """
    current_chunk = []
    current_length = 0

    for line in lines:
        line_length = len(line)

        # If adding this line would exceed limit and we have content, yield chunk
        if current_length + line_length > chunk_size and current_chunk:
            yield ' '.join(current_chunk)
            current_chunk = []
            current_length = 0

        current_chunk.append(line)
        current_length += line_length + 1  # +1 for space

    # Yield remaining chunk
    if current_chunk:
        yield ' '.join(current_chunk)