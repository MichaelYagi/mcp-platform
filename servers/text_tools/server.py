"""
Text Tools MCP Server
Runs over stdio transport
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader

import inspect
import json
import logging
from pathlib import Path
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response

from mcp.server.fastmcp import FastMCP
from tools.text_tools.split_text import split_text
from tools.text_tools.summarize_chunk import summarize_chunk
from tools.text_tools.merge_summaries import merge_summaries
from tools.text_tools.summarize_text import summarize_text
from tools.text_tools.summarize_direct import summarize_direct
from tools.text_tools.explain_simplified import explain_simplified
from tools.text_tools.concept_contextualizer import concept_contextualizer
from tools.text_tools.read_file_tool import read_file_tool

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
logging.getLogger("mcp_text_tools_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_text_tools_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("text-tools-server")

@mcp.tool()
@check_tool_enabled(category="text_tools")
def split_text_tool(text: str, max_chunk_size: int = 2000) -> str:
    """
    Split long text into manageable chunks for processing.

    Args:
        text (str, required): The text to split
        max_chunk_size (int, optional): Maximum characters per chunk (default: 2000)

    Returns:
        JSON string with:
        - chunks: Array of text segments
        - total_chunks: Number of chunks created
        - original_length: Length of input text

    Use for breaking down large documents before summarization or analysis.
    """
    logger.info(f"🛠 [server] split_text_tool called with text: {text}, max_chunk_size: {max_chunk_size}")
    return json.dumps(split_text(text, max_chunk_size))


@mcp.tool()
@check_tool_enabled(category="text_tools")
def summarize_chunk_tool(chunk: str, style: str = "short") -> str:
    """
    Summarize a single text chunk.

    Args:
        chunk (str, required): Text segment to summarize
        style (str, optional): Summary style - "brief"/"short"/"medium"/"detailed" (default: "short")

    Returns:
        JSON string with:
        - summary: The generated summary
        - original_length: Length of input chunk
        - summary_length: Length of summary
        - compression_ratio: How much text was reduced

    Use for summarizing individual text segments or chunks.
    """
    logger.info(f"🛠 [server] summarize_chunk_tool called with chunk: {chunk}, style: {style}")
    return json.dumps(summarize_chunk(chunk, style))


@mcp.tool()
@check_tool_enabled(category="text_tools")
def merge_summaries_tool(summaries: List[str], style: str = "medium") -> str:
    """
    Combine multiple summaries into one cohesive summary.

    Args:
        summaries (List[str], required): Array of summary texts to merge
        style (str, optional): Output style - "short"/"medium"/"detailed" (default: "medium")

    Returns:
        JSON string with:
        - merged_summary: The combined summary
        - input_count: Number of summaries merged
        - total_input_length: Combined length of inputs
        - output_length: Length of merged summary

    Use for combining chunk summaries into a final document summary.
    """
    logger.info(f"🛠 [server] merge_summaries_tool called with summaries: {summaries}, style: {style}")
    return json.dumps(merge_summaries(summaries, style))


@mcp.tool()
@check_tool_enabled(category="text_tools")
def summarize_text_tool(text: str | None = None,
                        file_path: str | None = None,
                        style: str = "medium") -> str:
    """
    Summarize text from direct input or file.

    Args:
        text (str, optional): Direct text to summarize (mutually exclusive with file_path)
        file_path (str, optional): Path to text file to summarize
        style (str, optional): Summary style - "short"/"medium"/"detailed" (default: "medium")

    Must provide either text OR file_path, not both.

    Returns:
        JSON string with:
        - summary: The generated summary
        - source: "text" or file path
        - original_length: Length of input
        - chunks_processed: Number of chunks if text was split

    Use for comprehensive text summarization from various sources.
    """
    logger.info(f"🛠 [server] summarize_text_tool called with text: {text}, file_path: {file_path}, style: {style}")
    return json.dumps(summarize_text(text, file_path, style))


@mcp.tool()
@check_tool_enabled(category="text_tools")
def summarize_direct_tool(text: str, style: str = "medium") -> str:
    """
    Summarize text in a single LLM call (for shorter texts).

    Args:
        text (str, required): Text to summarize (should be under 4000 characters)
        style (str, optional): Summary style - "short"/"medium"/"detailed" (default: "medium")

    Returns:
        JSON string with:
        - summary: The generated summary
        - style_used: The style applied
        - original_length: Length of input text

    Use for quick summarization of shorter texts without chunking overhead.
    """
    logger.info(f"🛠 [server] summarize_direct_tool called with text: {text}, style: {style}")
    return json.dumps(summarize_direct(text, style))


@mcp.tool()
@check_tool_enabled(category="text_tools")
def explain_simplified_tool(concept: str) -> str:
    """
    Explain complex concepts using the Ladder of Abstraction.

    Args:
        concept (str, required): The concept or term to explain

    Returns:
        JSON string with three explanation levels:
        - analogy: Simple real-world comparison
        - simple_explanation: Plain language explanation
        - technical_definition: Precise technical definition
        - concept: The original concept

    Use when user wants to understand complex topics at multiple levels.
    """
    logger.info(f"🛠 [server] explain_simplified_tool called with concept: {concept}")
    result = explain_simplified(concept)
    return json.dumps(result)


@mcp.tool()
@check_tool_enabled(category="text_tools")
def concept_contextualizer_tool(concept: str) -> str:
    """
    Provide comprehensive context and background for a concept.

    Args:
        concept (str, required): The concept to contextualize

    Returns:
        JSON string with:
        - concept: The concept name
        - definition: Clear definition
        - context: Background and history
        - related_concepts: Connected ideas
        - applications: Real-world uses
        - examples: Concrete examples

    Use when user wants deep understanding with context and connections.
    """
    logger.info(f"🛠 [server] concept_contextualizer_tool called with concept: {concept}")
    result = concept_contextualizer(concept)
    return json.dumps(result)

@mcp.tool()
@check_tool_enabled(category="text_tools")
def read_file_tool_handler(file_path: str) -> str:
    """
    Read any local file and return its contents for analysis or summarization.

    Args:
        file_path (str, required): Absolute path to the file.
            Supports Linux paths (/mnt/c/...) and Windows paths (C:\\Users\\...).
            Supported types: CSV, TSV, TXT, MD, JSON, YAML, TOML, XML, LOG,
                             PY, JS, TS, INI, CFG, CONF, SH and more.

    Returns:
        JSON string with:
        - success: Whether the file was read successfully
        - content: Full file text (up to 100KB)
        - file_name: Filename
        - file_type: Extension
        - size_bytes: File size
        - truncated: True if file exceeded 100KB limit
        - columns: (CSV/TSV only) List of column headers
        - row_count: (CSV/TSV only) Number of data rows

    Use this first when the user provides a file path and wants insights,
    analysis, or a summary of the file contents.
    Chain with summarize_text_tool or summarize_direct_tool for long files.
    """
    logger.info(f"🛠 [server] read_file_tool called: {file_path}")
    result = read_file_tool(file_path)
    return json.dumps(result, indent=2)


skill_registry = None

@mcp.tool()
@check_tool_enabled(category="text_tools")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "text-tools-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "text-tools-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="text_tools")
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
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)

    return tool_names

if __name__ == "__main__":
    # Auto-extract tool names - NO manual list needed!
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="text_tools")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")