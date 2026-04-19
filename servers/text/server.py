"""
Text Tools MCP Server
Runs over stdio transport
"""
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

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
try:
    from client.tool_meta import tool_meta
except Exception:
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator
from tools.text.summarize_text import summarize_text
from tools.text.summarize_direct import summarize_direct
from tools.text.explain_simplified import explain_simplified
from tools.text.concept_contextualizer import concept_contextualizer
from tools.text.read_file_tool import read_file_tool
from tools.text.improve_text import improve_text

from client.search_client import get_search_client

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_text_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_text_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("text-server")


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read","ai"],triggers=["summarize","summarise","summary","tldr"],idempotent=True,example='use summarize_text_tool: [text=""] [file_path=""] [style=""]',text_fields=["summary"])
def summarize_text_tool(text: Optional[str] = None,
                        file_path: Optional[str] = None,
                        style: Optional[str] = "medium") -> str:
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
    style = style if style is not None else "medium"
    logger.info(f"🛠 [server] summarize_text_tool called with text: {text}, file_path: {file_path}, style: {style}")
    return json.dumps(summarize_text(text, file_path, style))


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read","ai"],triggers=["explain","simplify","break down","eli5"],idempotent=True,example='use explain_simplified_tool: concept=""')
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
@check_tool_enabled(category="text")
@tool_meta(tags=["read","ai"],triggers=["contextualize","what is","explain concept"],idempotent=True,example='use concept_contextualizer_tool: concept=""')
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
@check_tool_enabled(category="text")
@tool_meta(tags=["read"],triggers=["read file","open file","load file","analyze file"],idempotent=True,example='use read_file_tool_handler: file_path=""',text_fields=["content"])
def read_file_tool_handler(file_path: str) -> str:
    """
    Read any local file and return its contents for analysis or summarization.

    IMPORTANT: You have direct filesystem access. When the user provides a file
    path, you MUST call this tool immediately. Do NOT ask the user to upload
    the file or say you cannot access it — you can and should read it directly.

    Args:
        file_path (str, required): The COMPLETE path to the file, including any
            spaces in the filename. Do not truncate the path at spaces.
            Supported: Linux (/mnt/c/..., /home/...) and Windows (C:\\Users\\...).
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

    Always call this tool first when the user provides any file path.
    Never ask the user to upload a file if a path has been provided.
    Chain with summarize_text_tool for long files.
    """
    logger.info(f"🛠 [server] read_file_tool called: {file_path}")
    result = read_file_tool(file_path)
    return json.dumps(result, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# improve_text_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(
    tags=["write", "ai"],
    triggers=[
        "improve text", "rewrite", "expand text", "fix grammar", "fix spelling",
        "shorten text", "make it formal", "make it casual", "make it shorter",
        "make it longer", "clean up", "polish", "proofread",
    ],
    idempotent=True,
    example='use improve_text_tool: text="" mode="improve" [instruction=""]',
    text_fields=["result"],
)
def improve_text_tool(
    text: str,
    mode: Optional[str] = "improve",
    instruction: Optional[str] = None,
) -> str:
    """
    Improve, rewrite, or transform text using a local Ollama model.

    Args:
        text (str, required):         The text to process.
        mode (str, optional):         What to do with the text. Options:
                                        expand   — Add detail and depth
                                        improve  — Improve clarity and flow (default)
                                        fix      — Fix grammar/spelling only
                                        shorten  — Remove redundancy, condense
                                        formal   — Rewrite in formal/professional tone
                                        casual   — Rewrite in casual/conversational tone
                                        custom   — Use a custom instruction (requires instruction param)
        instruction (str, optional):  Custom instruction when mode is "custom".
                                      e.g. "Rewrite this as bullet points"

    Returns:
        JSON string with result (str), mode (str), original_length (int),
        result_length (int), and error (str on failure).
    """
    logger.info(f"🛠 [server] improve_text_tool called — mode={mode!r}, len={len(text)}")
    result = improve_text(text, mode or "improve", instruction)
    return json.dumps(result, indent=2)


skill_registry = None


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read","search","external"],triggers=["search web","google","look up","find online","web search"],idempotent=True,example='use web_search_tool: query=""',text_fields=["snippet"])
def web_search_tool(query: str) -> str:
    """
    Search the web for current information using Ollama's web search API.

    Use this for current events, news, stock prices, or any query that needs
    up-to-date information not available in the model's training data.

    Requires OLLAMA_TOKEN in .env (free Ollama account).

    Args:
        query (str, required): The search query

    Returns:
        Formatted list of search results, each with title, URL and summary.
    """
    import asyncio, concurrent.futures
    max_results = 5
    logger.info(f"🛠 [server] web_search_tool called — query={query!r}, max_results={max_results}")

    async def _run():
        client = get_search_client()
        if not client.is_available():
            return "Web search is not available. Set OLLAMA_TOKEN in your .env file."
        return await client.search(query, max_results=max_results)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run()).result()
        else:
            result = loop.run_until_complete(_run())
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, _run()).result()

    if isinstance(result, str):
        return result
    if not result.get("success"):
        return f"Web search failed: {result.get('error', 'Unknown error')}"

    pages = result.get("results", {}).get("webPages", {}).get("value", [])
    if not pages:
        return f'No results found for "{query}".'

    lines = [f'Web search results for "{query}":\n']
    logger.info(f"[web_search_tool] ⏱ starting summarize loop at {time.time():.2f}")
    for i, page in enumerate(pages, 1):
        title = page.get("name", "Untitled")
        url = page.get("url", "")
        content = page.get("summary", "").strip()

        lines.append(f"{i}. {title}")
        lines.append(f"   🔗 {url}")

        if content:
            try:
                summarized = summarize_direct(f"{title}\n\n{content}", style="short")
                description = summarized.get("summary", "").strip()
                if not description or description.startswith(title):
                    description = content[:300]
            except Exception as e:
                logger.warning(f"[web_search_tool] summarize_direct failed: {e}")
                description = content[:300]
            lines.append(f"   📝 {description}")

        lines.append("")

    logger.info(f"[web_search_tool] returning {len(pages)} results")
    return "\n".join(lines)


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read","external"],triggers=["fetch url","get page","read website"],idempotent=True,example='use web_fetch_tool: url=""')
def web_fetch_tool(url: str) -> str:
    """
    Fetch and return the clean text content of a web page.

    Use this to read the full content of a URL found via web_search_tool.
    Requires OLLAMA_TOKEN in .env (free Ollama account).

    Args:
        url (str, required): The full URL to fetch

    Returns:
        Clean text content of the page, truncated at 10,000 characters if needed.
    """
    import asyncio, concurrent.futures
    logger.info(f"🛠 [server] web_fetch_tool called — url={url!r}")

    async def _run():
        client = get_search_client()
        if not client.is_available():
            return "Web fetch is not available. Set OLLAMA_TOKEN in your .env file."
        return await client.fetch_url(url)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                result = pool.submit(asyncio.run, _run()).result()
        else:
            result = loop.run_until_complete(_run())
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            result = pool.submit(asyncio.run, _run()).result()

    if isinstance(result, str):
        return result
    if not result.get("success"):
        return f"Failed to fetch {url}: {result.get('error', 'Unknown error')}"

    output = []
    if result.get("title"):
        output.append(f"# {result['title']}\n")
    output.append(result.get("content", ""))
    return "\n".join(output)


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read","external","ai"],triggers=["summarize url","summarize this page","tldr url"],idempotent=True,example='use summarize_url_tool: url="" [style=""]',text_fields=["summary"])
def summarize_url_tool(url: str, style: Optional[str] = "medium") -> str:
    """
    Fetch and summarize the content of a web page in a single step.

    Args:
        url (str, required): The full URL to fetch and summarize
        style (str, optional): Summary style - "short"/"medium"/"detailed" (default: "medium")

    Returns:
        JSON string with:
        - summary: The generated summary
        - url: The source URL
        - title: Page title if available
        - original_length: Length of fetched content
        - chunks_processed: Number of chunks if text was split

    Use when the user provides a URL and wants a summary without manual chaining.
    Requires OLLAMA_TOKEN in .env (free Ollama account).
    """
    import asyncio, concurrent.futures
    style = style if style is not None else "medium"
    logger.info(f"🛠 [server] summarize_url_tool called — url={url!r}, style={style!r}")

    async def _fetch():
        client = get_search_client()
        if not client.is_available():
            return None, "Web fetch is not available. Set OLLAMA_TOKEN in your .env file."
        result = await client.fetch_url(url)
        if isinstance(result, str):
            return None, result
        if not result.get("success"):
            return None, f"Failed to fetch {url}: {result.get('error', 'Unknown error')}"
        return result, None

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                fetch_result, err = pool.submit(asyncio.run, _fetch()).result()
        else:
            fetch_result, err = loop.run_until_complete(_fetch())
    except RuntimeError:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            fetch_result, err = pool.submit(asyncio.run, _fetch()).result()

    if err:
        return json.dumps({"error": err, "url": url})

    title = fetch_result.get("title", "")
    content = fetch_result.get("content", "")
    full_text = f"{title}\n\n{content}" if title else content

    summary_result = summarize_text(full_text, None, style)
    summary_result["url"] = url
    summary_result["title"] = title
    return json.dumps(summary_result)


@mcp.tool()
@check_tool_enabled(category="text")
@tool_meta(tags=["read"],triggers=["list capabilities","what can you do"],idempotent=True,example="use list_capabilities",intent_category="text")
def list_capabilities(filter_tags: Optional[str] = None) -> str:
    """
    Return the full capability schema for every tool on this server.

    Args:
        filter_tags (str, optional): Comma-separated tags to filter by

    Returns:
        JSON string with server name, tools array, and total count.
    """
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")

    try:
        from client.capability_registry import (
            _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    import sys as _sys, inspect as _inspect
    _current = _sys.modules[__name__]
    wanted_tags = set(t.strip() for t in filter_tags.split(",") if t.strip()) if filter_tags else None

    tools_out = []
    seen = set()
    for _name, _obj in vars(_current).items():
        if not callable(_obj) or _name.startswith("_") or _name in _INTERNAL_TOOLS:
            continue
        _tool_fn = getattr(_current, _name, None)
        if not (hasattr(_tool_fn, "__tool_meta__") or hasattr(_tool_fn, "_mcp_tool")):
            continue
        if _name in seen:
            continue
        seen.add(_name)
        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue
        sig = _inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            has_default = param.default is not _inspect.Parameter.empty
            ann = param.annotation
            type_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "").replace("Optional[", "").rstrip("]")
                if ann is not _inspect.Parameter.empty else "string"
            )
            params.append({"name": pname, "type": type_str, "required": not has_default,
                           "default": None if not has_default else str(param.default)})
        tools_out.append({
            "name": _name,
            "description": (_obj.__doc__ or "").strip().split("\n")[0],
            "input_schema": params,
            "tags": tags,
            "rate_limit": _TOOL_RATE_LIMITS.get(_name),
            "idempotent": _TOOL_IDEMPOTENT.get(_name, True),
        })

    return json.dumps({"server": mcp.name, "tools": tools_out, "total": len(tools_out)}, indent=2)


@mcp.tool()
@check_tool_enabled(category="text")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "text-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "text-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="text")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called")
    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)
    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content
    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({"error": f"Skill '{skill_name}' not found", "available_skills": available}, indent=2)


def get_tool_names_from_module():
    current_module = sys.modules[__name__]
    tool_names = []
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    server_tools = get_tool_names_from_module()
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="text")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")