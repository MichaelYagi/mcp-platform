"""
Environment Variable Display Utility
"""
import os
from typing import Dict, Any


def get_env_display() -> Dict[str, Any]:
    """
    Get current environment variable values for display.
    Masks sensitive tokens.

    Returns:
        Dictionary with categorized env vars
    """

    def mask_token(value: str) -> str:
        """Mask token but handle empty/None"""
        if not value:
            return "(not set)"
        return "*" * len(value)

    env_vars = {
        "plex": {
            "PLEX_URL": os.getenv("PLEX_URL") or "(not set)",
            "PLEX_TOKEN": mask_token(os.getenv("PLEX_TOKEN"))
        },
        "weather": {
            "WEATHER_TOKEN": mask_token(os.getenv("WEATHER_TOKEN"))
        },
        "a2a": {
            "A2A_ENDPOINTS": os.getenv("A2A_ENDPOINTS") or "(not set)",
            "A2A_EXPOSED_TOOLS": os.getenv("A2A_EXPOSED_TOOLS") or "(not set)"
        },
        "ollama": {
            "OLLAMA_TOKEN": mask_token(os.getenv("OLLAMA_TOKEN"))
        },
        "agent": {
            "MAX_MESSAGE_HISTORY": os.getenv("MAX_MESSAGE_HISTORY") or "20"
        },
        "rag_performance": {
            "CONCURRENT_LIMIT": os.getenv("CONCURRENT_LIMIT") or "1",
            "EMBEDDING_BATCH_SIZE": os.getenv("EMBEDDING_BATCH_SIZE") or "10",
            "DB_FLUSH_BATCH_SIZE": os.getenv("DB_FLUSH_BATCH_SIZE") or "30"
        }
    }

    return env_vars


def format_env_display() -> str:
    """
    Format environment variables for display.

    Returns:
        Formatted string (wrapped in code block for web UI)
    """
    env_vars = get_env_display()

    output = []
    output.append("📋 ENVIRONMENT CONFIGURATION")
    output.append("=" * 50)

    # Plex
    output.append("\n🎬 Plex Media Server:")
    output.append(f"   PLEX_URL: {env_vars['plex']['PLEX_URL']}")
    output.append(f"   PLEX_TOKEN: {env_vars['plex']['PLEX_TOKEN']}")

    # Weather
    output.append("\n🌤️  Weather API:")
    output.append(f"   WEATHER_TOKEN: {env_vars['weather']['WEATHER_TOKEN']}")

    # A2A
    output.append("\n🔗 A2A Protocol:")
    output.append(f"   A2A_ENDPOINTS: {env_vars['a2a']['A2A_ENDPOINTS']}")
    output.append(f"   A2A_EXPOSED_TOOLS: {env_vars['a2a']['A2A_EXPOSED_TOOLS']}")

    # Search CLIENT
    output.append("\n🔍 Search Client Web Search:")
    output.append(f"   OLLAMA_TOKEN: {env_vars['ollama']['OLLAMA_TOKEN']}")

    # Agent Config
    output.append("\n🤖 Agent Configuration:")
    output.append(f"   MAX_MESSAGE_HISTORY: {env_vars['agent']['MAX_MESSAGE_HISTORY']}")

    # RAG Performance
    output.append("\n⚡ RAG Performance:")
    output.append(f"   CONCURRENT_LIMIT: {env_vars['rag_performance']['CONCURRENT_LIMIT']}")
    output.append(f"   EMBEDDING_BATCH_SIZE: {env_vars['rag_performance']['EMBEDDING_BATCH_SIZE']}")
    output.append(f"   DB_FLUSH_BATCH_SIZE: {env_vars['rag_performance']['DB_FLUSH_BATCH_SIZE']}")

    output.append("\n" + "=" * 50)

    formatted = "\n".join(output)
    return formatted