"""
LangGraph Module with Centralized Pattern Configuration
Handles LangGraph agent creation, routing, and execution
"""
import asyncio
import json
import logging
import operator
import re
import time
from typing import TypedDict, Annotated, Sequence
from .stop_signal import is_stop_requested, clear_stop
from .langsearch_client import get_langsearch_client
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import only router patterns (for router() function)
from client.query_patterns import (
    ROUTER_INGEST_COMMAND, ROUTER_STATUS_QUERY, ROUTER_MULTI_STEP,
    ROUTER_ONE_TIME_INGEST, ROUTER_EXPLICIT_RAG, ROUTER_KNOWLEDGE_QUERY,
    ROUTER_EXCLUDE_MEDIA
)

# Try to import metrics
try:
    from metrics import metrics
    METRICS_AVAILABLE = True
except ImportError:
    try:
        from client.metrics import metrics
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        from collections import defaultdict
        metrics = {
            "agent_runs": 0,
            "agent_errors": 0,
            "agent_times": [],
            "llm_calls": 0,
            "llm_errors": 0,
            "llm_times": [],
            "tool_calls": defaultdict(int),
            "tool_errors": defaultdict(int),
            "tool_times": defaultdict(list),
        }

# ═══════════════════════════════════════════════════════════════════
# CENTRALIZED PATTERN CONFIGURATION
# Add new intents here - no code changes needed!
# ═══════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "general_knowledge": {
        "pattern": (
            r'\bwhat\s+is\s+(a|an|the)\s+\w+\??$'  # "what is a div?"
            r'|\bdefine\s+'
            r'|\bexplain\s+\w+\s*$'  # "explain recursion"
            r'|\bhow\s+does\s+\w+\s+work\??$'
        ),
        "tools": [],  # NO TOOLS - just answer from knowledge
        "priority": 4  # Lower priority than specific intents
    },
    "rag_status": {
        "pattern": (
            r'\bhow\s+many\s+.*(ingested|in\s+rag)\b'
            r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
            r'|\bitems?\s+(have\s+been|were)\s+ingested\b'
            r'|\bcount\s+.*(items?|in\s+rag)\b'
            r'|\btotal\s+.*(items?|in\s+rag)\b'
            r'|\b(show|list|display)\s+rag\b'
            r'|\brag\s+(status|contents?|info|summary|overview|report)\b'
            r'|\bwhat(\'s| is)\s+in\s+(the\s+)?rag\b'
            r'|\bgive\s+me\s+rag\s+(stats|status|info|details)\b'
            r'|\bwhat\s+has\s+been\s+ingested\s+so\s+far\b'
            r'|\bwhat\s+did\s+you\s+ingest\b'
            r'|\bshow\s+me\s+everything\s+in\s+rag\b'
            r'|\brag\s+items\b'
            r'|\brag\s+dump\b'
            r'|\brag\s+data\b'
            r'|\bcurrent\s+rag\s+state\b'
        ),
        "tools": ["rag_status_tool", "rag_diagnose_tool"],
        "priority": 1
    },
    "ingest": {
        "pattern": (
            r'\bingest\b'
            r'|\bprocess\b'
            r'|\badd\s+to\s+rag\b'
            r'|\bindex\b'
            r'|\bvectorize\b'
            r'|\bembed\s+(this|it)\b'
            r'|\bupdate\s+rag\b'
            r'|\brefresh\s+rag\b'
            r'|\bscan\s+plex\b'
            r'|\bprocess\s+next\b'
            r'|\bingest\s+next\b'
            r'|\badd\s+movie\b'
            r'|\badd\s+media\b'
        ),
        "exclude_pattern": (
            r'\bhow\s+many\b'
            r'|\bwhat\s+(has|was)\b'
            r'|\bcount\b'
            r'|\btotal\b'
            r'|\bstatus\b'
            r'|\bwhat(\'s| is)\s+in\s+rag\b'
        ),
        "tools": [
            "plex_ingest_*",
            "plex_find_unprocessed",
            "plex_ingest_single",
            "plex_ingest_batch",
            "rag_add_tool"
        ],
        "priority": 2
    },
    "github_review": {
        "pattern": (
            # GitHub URLs
            r'\\bgithub\\.com/[^\\s]+'
            r'|\\breview.*github'
            r'|\\bclone.*github'
            r'|\\bcheck.*github\\.com'

            # Repository review
            r'|\\breview.*(repo|repository)'
            r'|\\banalyze.*(repo|repository)'
            r'|\\bcheck.*(repo|repository)'

            # Code from URL
            r'|\\breview.*https?://'
            r'|\\banalyze.*https?://'

            # Explicit GitHub references
            r'|\\bgithub\\s+(repo|repository|project)'
            r'|\\bfrom\\s+github'
        ),
        "tools": [
            "github_clone_repo",
            "github_list_files",
            "github_get_file_content",
            "github_cleanup_repo",
            "analyze_project",
            "analyze_code_file",
            "review_code",
            "scan_project_structure"
        ],
        "priority": 2
    },
    "code_assistant": {
        "pattern": (
            # Tech stack queries
            r'\btech\s+stack\b'
            r'|\btechnology\s+stack\b'
            r'|\bwhat.*tech\b'
            r'|\bwhat.*stack\b'
            r'|\bwhat.*technologies\b'
            r'|\bwhat.*languages\b'
            r'|\bwhat.*frameworks?\b'
            r'|\bwhat.*dependencies\b'
            r'|\bproject\s+structure\b'
            r'|\banalyze.*project\b'
            r'|\bscan.*project\b'
            r'|\bshow.*structure\b'
            r'|\blist.*dependencies\b'
            
            # Node.js/Package queries - UPDATED TO CATCH MORE
            r'|\bnode\.?js\s+(packages?|dependencies|modules)\b'
            r'|\bnpm\s+(packages?|dependencies|modules)\b'
            r'|\bpackage\.json\b'
            r'|\bnode\s+(packages?|dependencies|modules)\b'
            r'|\b(about|more|explain).*node\.?js\s+(packages?|dependencies)\b'  # ← ADD THIS
            r'|\btell.*about.*(node|packages?|dependencies)\b'  # ← ADD THIS
            r'|\bmore.*about.*(node|packages?|dependencies)\b'  # ← ADD THIS
            
            # "go into depth" style questions
            r'|\bgo\s+into\s+(depth|detail)\b'
            r'|\bmore\s+detail.*about.*(packages?|dependencies|modules)\b'
            r'|\bin[\s-]?depth.*about.*(packages?|dependencies|modules)\b'
            r'|\belaborate.*on.*(packages?|dependencies|modules)\b'
            r'|\bexpand.*on.*(packages?|dependencies|modules)\b'
            
            # Generic package questions
            r'|\bwhat.*(do|are).*(packages?|dependencies|modules)\b'
            r'|\bexplain.*(packages?|dependencies|modules)\b'
            r'|\bwhat.*do\s+they\s+do\b'
            r'|\bwhat.*are\s+(they|those)\s+(for|used\s+for)\b'
            r'|\bwhat.*they.*used\s+for\b'  # ← ADD THIS
            
            # Code analysis
            r'|\banalyze.*code\b'
            r'|\bcheck.*code\b'
            r'|\breview.*code\b'
            r'|\blint\b'
            
            # File extensions
            r'|\banalyze.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'
            r'|\bcheck.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'
            
            # Bug fixing
            r'|\bfix.*bug\b'
            r'|\bfix.*error\b'
            r'|\bfix.*issue\b'
            r'|\bfix.*code\b'
            
            # Code generation
            r'|\bgenerate.*code\b'
            r'|\bcreate.*(function|class|module|component)\b'
            r'|\bwrite.*(function|class)\b'
        ),
        "tools": [
            "analyze_project",
            "get_project_dependencies",
            "scan_project_structure",
            "analyze_code_file",
            "fix_code_file",
            "suggest_improvements",
            "explain_code",
            "generate_tests",
            "refactor_code",
            "generate_code"
        ],
        "priority": 2
    },
    "location": {
        "pattern": (
            r'\b(my|what\'?s?\s+my)\s+location\b'
            r'|\bwhere\s+am\s+i\b'
            r'|\bcurrent\s+location\b'
            r'|\bwhere\s+do\s+i\s+live\b'
        ),
        "tools": ["get_location_tool"],
        "priority": 3
    },
    "weather": {
        "pattern": (
            r'\bweather\b'
            r'|\btemperature\b'
            r'|\bforecast\b'
            r'|\brain\b'
            r'|\bsnow\b'
            r'|\bwind\b'
            r'|\bconditions\b'
        ),
        "tools": ["get_location_tool", "get_weather_tool"],
        "priority": 3
    },
    "time": {
        "pattern": (
            r'\bwhat\s+time\b'
            r'|\bcurrent\s+time\b'
            r'|\btime\s+now\b'
            r'|\btime\s+is\s+it\b'
        ),
        "tools": ["get_time_tool"],
        "priority": 3
    },
    # ═══════════════════════════════════════════════════════════════
    # CRITICAL FIX: plex_search BEFORE ml_recommendation (priority 2 vs 3)
    # ═══════════════════════════════════════════════════════════════
    "plex_search": {
        "pattern": (
            # Direct search phrases
            r'\b(find|search|look\s+for|show\s+me)\s+.*\b(movie|film|show|media|series)\b'

            # Plot-based searches (CRITICAL for "where hero wins")
            r'|\bmovies?\s+(about|where|with|featuring|in\s+which)\b'
            r'|\bfilms?\s+(about|where|with|featuring|in\s+which)\b'

            # "where X happens" pattern
            r'|\bwhere\s+.*\s+(wins?|loses?|dies|survives|happens|occurs|escapes)\b'

            # Library references
            r'|\bsearch\s+(plex|library|my\s+library|my\s+movies)\b'
            r'|\bfind\s+.*\s+in\s+(plex|library|my\s+library)\b'

            # Scene searches
            r'|\bscene\s+(where|with|from)\b'
            r'|\bfind\s+scene\b'
            r'|\blocate\s+scene\b'

            # Browse/explore (not recommendations)
            r'|\bbrowse\s+my\b'
            r'|\blist\s+.*\s+(movies|films|shows)\b'
        ),
        "tools": [
            "rag_search_tool",
            "semantic_media_search_text",
            "scene_locator_tool",
            "find_scene_by_title"
        ],
        "priority": 2  # HIGHER priority than ml_recommendation!
    },
    "ml_recommendation": {
        "pattern": (
            # Explicit recommendation requests
            r'\brecommend(ation)?s?\b'
            r'|\bsuggest(ion)?s?\b'

            # ML/training specific
            r'|\bml\s+(model|train|recommendation)\b'
            r'|\btrain\s+(model|recommender|recommendation)\b'
            r'|\bauto.?train\b'

            # History management
            r'|\bimport\s+.*\s*history\b'
            r'|\bviewing\s+history\b'
            r'|\bwatch\s+history\b'
            r'|\brecord\s+(viewing|that\s+i\s+watched)\b'

            # Personalized suggestions
            r'|\bwhat\s+should\s+i\s+watch\b'
            r'|\brank\s+(these|movies|shows)\b'
            r'|\bmy\s+best\s+unwatched\b'
            r'|\bunwatched\s+(recommendations|suggestions)\b'

            # Stats/model info
            r'|\brecommender\s+stats\b'
        ),
        "tools": [
            "record_viewing",
            "train_recommender",
            "recommend_content",
            "get_recommender_stats",
            "import_plex_history",
            "auto_train_from_plex",
            "reset_recommender",
            "auto_recommend_from_plex"
        ],
        "priority": 3  # LOWER priority - only matches if plex_search doesn't
    },
    "system": {
        "pattern": (
            r'\bsystem\s+info\b'
            r'|\bhardware\b'
            r'|\b(cpu|gpu|ram)\b'
            r'|\bspecs?\b'
            r'|\bprocesses?\b'
            r'|\bperformance\b'
            r'|\butilization\b'
            r'|\bmemory\s+usage\b'
        ),
        "tools": [
            "get_hardware_specs_tool",
            "get_system_info",
            "list_system_processes",
            "terminate_process"
        ],
        "priority": 3
    },
    "code": {
        "pattern": (
            r'\bcode\b'
            r'|\bscan\s+code\b'
            r'|\bdebug\b'
            r'|\breview\s+code\b'
            r'|\bsummarize\s+code\b'
            r'|\bfix\s+this\s+code\b'
            r'|\bexplain\s+this\s+code\b'
        ),
        "tools": [
            "review_code",
            "summarize_code_file",
            "search_code_in_directory",
            "scan_code_directory",
            "summarize_code",
            "debug_fix"
        ],
        "priority": 3
    },
    "text": {
        "pattern": (
            r'\b(summarize|summary|explain|simplify|break\s+down)\b'
        ),
        "exclude_pattern": r'\bcode\b',
        "tools": [
            "summarize_text_tool",
            "summarize_direct_tool",
            "explain_simplified_tool",
            "concept_contextualizer_tool"
        ],
        "priority": 3
    },
    "todo": {
        "pattern": (
            r'\btodo\b'
            r'|\btask\b'
            r'|\bremind\s+me\b'
            r'|\bmy\s+todos?\b'
            r'|\bmy\s+tasks?\b'
            r'|\badd\s+to\s+my\s+list\b'
            r'|\btask\s+list\b'
        ),
        "tools": [
            "add_todo_item",
            "list_todo_items",
            "search_todo_items",
            "update_todo_item",
            "delete_todo_item",
            "delete_all_todo_items"
        ],
        "priority": 3
    },
    "knowledge": {
        "pattern": (
            r'\bremember\b'
            r'|\bsave\s+this\b'
            r'|\bmake\s+a\s+note\b'
            r'|\bknowledge\s+base\b'
            r'|\bsearch\s+my\s+notes?\b'
            r'|\badd\s+entry\b'
            r'|\bnote\s+this\b'
            r'|\bstore\s+this\b'
        ),
        "tools": [
            "add_entry",
            "list_entries",
            "get_entry",
            "search_entries",
            "search_by_tag",
            "search_semantic",
            "update_entry",
            "delete_entry"
        ],
        "priority": 3
    },
    "a2a": {
        "pattern": (
            r'\ba2a\b'
            r'|\bremote\s+(agent|tools?)\b'
            r'|\bdiscover\s+(agent|tools?)\b'
            r'|\bsend\s+to\s+remote\b'
            r'|\bcall\s+remote\s+tool\b'
            r'|\buse\s+remote\s+agent\b'
            r'|\bconnect\s+to\s+agent\b'
        ),
        "tools": ["send_a2a*", "discover_a2a"],
        "priority": 3
    }
}


class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tools: dict
    llm: object
    ingest_completed: bool
    stopped: bool
    current_model: str

def router(state):
    """
    Route based on what the agent decided to do
    WITH STOP SIGNAL HANDLING AND A2A LOOP PREVENTION
    """
    last_message = state["messages"][-1]
    logger = logging.getLogger("mcp_client")
    logger.info(f"[LangGraph] 🎯 Router: Last message type = {type(last_message).__name__}")

    # Stop signal check
    if is_stop_requested():
        logger.warning(f"🛑 Router: Stop requested - ending graph execution")
        state["stopped"] = True
        return "continue"

    if state.get("stopped", False):
        logger.warning(f"🛑 Router: Execution already stopped - ending")
        return "continue"

    # A2A completion check
    from langchain_core.messages import ToolMessage
    if isinstance(last_message, ToolMessage):
        if hasattr(last_message, 'name') and last_message.name in ["send_a2a", "discover_a2a",
                                                                   "send_a2a_streaming", "send_a2a_batch"]:
            logger.info(f"🛑 Router: {last_message.name} result received - ending execution")
            return "continue"

    # If LLM made tool calls, execute them
    if isinstance(last_message, AIMessage):
        tool_calls = getattr(last_message, "tool_calls", [])
        if tool_calls and len(tool_calls) > 0:
            logger.info(f"[LangGraph] 🎯 Router: Found {len(tool_calls)} tool calls - routing to TOOLS")
            return "tools"

    # Get user's original message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if user_message:
        content = user_message.content

        # Status query check
        if ROUTER_STATUS_QUERY.search(content):
            logger.info(f"🎯 Router: Status query detected - continuing normally")
            return "continue"

        # Ingest routing
        if ROUTER_INGEST_COMMAND.search(content) and not ROUTER_STATUS_QUERY.search(content):
            if not state.get("ingest_completed", False):
                if ROUTER_ONE_TIME_INGEST.search(content):
                    logger.info(f"🎯 Router: ONE-TIME ingest requested")
                    return "ingest"
                if ROUTER_MULTI_STEP.search(content):
                    logger.info(f"🎯 Router: INGEST with multiple steps")
                    return "continue"
                logger.info(f"🎯 Router: INGEST requested")
                return "ingest"
            else:
                logger.info(f"🎯 Router: Ingest already completed")
                return "continue"

        # Explicit RAG requests
        if ROUTER_EXPLICIT_RAG.search(content):
            logger.info(f"🎯 Router: Explicit RAG request")
            return "rag"

    # Default: continue to END
    logger.info(f"[LangGraph] 🎯 Router: Continuing to END")
    return "continue"


async def rag_node(state):
    """Search RAG and provide context to answer the question"""
    logger = logging.getLogger("mcp_client")

    if is_stop_requested():
        logger.warning("🛑 RAG node: Stop requested")
        msg = AIMessage(content="Search cancelled by user.")
        return {"messages": state["messages"] + [msg], "llm": state.get("llm"), "stopped": True}

    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if not user_message:
        logger.error("❌ No user message found in RAG node")
        msg = AIMessage(content="Error: Could not find user's question.")
        return {"messages": state["messages"] + [msg], "llm": state.get("llm")}

    # Find rag_search_tool
    tools_dict = state.get("tools", {})
    rag_search_tool = None
    for tool in tools_dict.values():
        if hasattr(tool, 'name') and tool.name == "rag_search_tool":
            rag_search_tool = tool
            break

    if not rag_search_tool:
        logger.error(f"❌ RAG search tool not found")
        msg = AIMessage(content="RAG search is not available.")
        return {"messages": state["messages"] + [msg], "llm": state.get("llm")}

    try:
        # Call RAG search
        result = await rag_search_tool.ainvoke({"query": user_message.content})

        # Parse results (simplified - add your parsing logic here)
        context = "RAG search results here"

        # Create augmented message with RAG context
        augmented_messages = [
            SystemMessage(content=f"Context from RAG:\n\n{context}"),
            user_message
        ]

        llm = state.get("llm")
        response = await llm.ainvoke(augmented_messages)

        return {"messages": state["messages"] + [response], "llm": state.get("llm")}

    except Exception as e:
        logger.error(f"❌ Error in RAG node: {e}")
        msg = AIMessage(content=f"Error searching knowledge base: {str(e)}")
        return {"messages": state["messages"] + [msg], "llm": state.get("llm")}


def create_langgraph_agent(llm_with_tools, tools):
    """Create and compile the LangGraph agent"""
    logger = logging.getLogger("mcp_client")

    base_llm = llm_with_tools.bound if hasattr(llm_with_tools, 'bound') else llm_with_tools

    async def call_model(state: AgentState):
        if is_stop_requested():
            logger.warning("🛑 call_model: Stop requested")
            empty_response = AIMessage(content="Operation cancelled by user.")
            return {
                "messages": state["messages"] + [empty_response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": True
            }

        messages = state["messages"]
        from langchain_core.messages import ToolMessage

        last_message = messages[-1] if messages else None

        # If formatting tool results, use base LLM
        if isinstance(last_message, ToolMessage):
            logger.info("[LangGraph] 🎯 Formatting tool results")
            start_time = time.time()
            try:
                response = await asyncio.wait_for(
                    base_llm.ainvoke(messages),
                    timeout=300.0
                )
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_calls"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                return {
                    "messages": messages + [response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False)
                }
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_errors"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                logger.error(f"⏱️ LLM call timed out after 5m")

                timeout_message = AIMessage(content="""⏱️ Request timed out after 5 minutes.

    **The model is taking too long to respond.** This usually happens when:
    - The model is processing too many tools (58 tools detected)
    - The query is ambiguous and the model is stuck deciding
    - The model is overloaded

    **Try these solutions:**
    1. Rephrase your question more specifically
    2. Break complex questions into smaller parts
    3. Restart the Ollama service: `ollama restart`

    **Your question:** {question}""".format(question=messages[-1].content if messages else "unknown"))

                return {
                    "messages": messages + [timeout_message],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False)
                }

            except Exception as e:
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_errors"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                logger.error(f"❌ Model call failed: {e}")
                raise

        # Get user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Force LangSearch if explicitly requested
        if user_message:
            user_lower = user_message.lower()

            langsearch_patterns = [
                r'\buse\s+langsearch\b',
                r'\busing\s+langsearch\b',
                r'\bwith\s+langsearch\b',
                r'\blangsearch\s+for\b',
                r'\bvia\s+langsearch\b',
            ]

            should_use_langsearch = any(
                re.search(pattern, user_lower)
                for pattern in langsearch_patterns
            )

            if should_use_langsearch:
                logger.info("[LangGraph] 🎯 FORCED LANGSEARCH: User explicitly requested langsearch")

                langsearch = get_langsearch_client()

                if langsearch.is_available():
                    query = user_message
                    for pattern in langsearch_patterns:
                        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
                    query = query.strip()

                    query = re.sub(r'^,?\s*(who|what|where|when|why|how)\s+', r'\1 ', query, flags=re.IGNORECASE)
                    query = query.strip()

                    if not query:
                        query = user_message

                    logger.info(f"🔍 Searching with langsearch: '{query}'")

                    try:
                        search_result = await langsearch.search(query)

                        if search_result["success"] and search_result["results"]:
                            logger.info("[LangGraph] ✅ LangSearch successful - passing to LLM for processing")
                            search_context = search_result["results"]

                            augmented_prompt = f"""I searched the web using LangSearch and found the following results:

    {search_context}

    Based on these search results, please answer the user's question: "{user_message}"

    Provide a clear, concise answer in English. Extract the most relevant information and present it naturally."""

                            augmented_messages = messages + [HumanMessage(content=augmented_prompt)]
                            response = await base_llm.ainvoke(augmented_messages)

                            return {
                                "messages": messages + [response],
                                "tools": state.get("tools", {}),
                                "llm": state.get("llm"),
                                "ingest_completed": state.get("ingest_completed", False),
                                "stopped": state.get("stopped", False)
                            }
                        else:
                            logger.warning("⚠️ LangSearch returned no results")

                    except Exception as e:
                        logger.error(f"❌ LangSearch failed: {e}")

                else:
                    logger.warning("⚠️ LangSearch not available")
                    error_response = AIMessage(
                        content="LangSearch is not available. Please check configuration."
                    )
                    return {
                        "messages": messages + [error_response],
                        "tools": state.get("tools", {}),
                        "llm": state.get("llm"),
                        "ingest_completed": state.get("ingest_completed", False),
                        "stopped": state.get("stopped", False)
                    }

        # ═══════════════════════════════════════════════════════════
        # CENTRALIZED PATTERN MATCHING
        # ═══════════════════════════════════════════════════════════
        def match_intent(user_message: str, all_tools: list, base_llm, logger, conversation_state):
            """Match user intent using centralized pattern configuration"""

            has_project_context = False
            for msg in reversed(conversation_state.get("messages", [])[-5:]):
                if isinstance(msg, SystemMessage) and "CONVERSATION CONTEXT" in msg.content:
                    has_project_context = True
                    logger.info("[LangGraph] 🎯 Found project context in conversation - using code_assistant")
                    break

            if has_project_context:
                config = INTENT_PATTERNS["code_assistant"]
                filtered_tools = []
                for tool in all_tools:
                    for tool_pattern in config["tools"]:
                        if "*" in tool_pattern:
                            prefix = tool_pattern.replace("*", "")
                            if tool.name.startswith(prefix):
                                filtered_tools.append(tool)
                                break
                        elif tool.name == tool_pattern:
                            filtered_tools.append(tool)
                            break

                if filtered_tools:
                    logger.info(f"   → {len(filtered_tools)} code tools (context-based routing)")
                    return base_llm.bind_tools(filtered_tools), "code_assistant"

            sorted_patterns = sorted(INTENT_PATTERNS.items(), key=lambda x: x[1]["priority"])

            for intent_name, config in sorted_patterns:
                if re.search(config["pattern"], user_message, re.IGNORECASE):
                    if "exclude_pattern" in config:
                        if re.search(config["exclude_pattern"], user_message, re.IGNORECASE):
                            continue

                    logger.info(f"🎯 {intent_name} → filtering tools")

                    filtered_tools = []
                    for tool in all_tools:
                        for tool_pattern in config["tools"]:
                            if "*" in tool_pattern:
                                prefix = tool_pattern.replace("*", "")
                                if tool.name.startswith(prefix):
                                    filtered_tools.append(tool)
                                    break
                            elif tool.name == tool_pattern:
                                filtered_tools.append(tool)
                                break

                    if filtered_tools:
                        logger.info(f"   → {len(filtered_tools)} tools: {[t.name for t in filtered_tools[:5]]}")
                        return base_llm.bind_tools(filtered_tools), intent_name

            logger.info(f"🎯 General query → all {len(all_tools)} tools")
            return base_llm.bind_tools(all_tools), "general"

        # Apply pattern matching
        if user_message:
            all_tools = list(state.get("tools", {}).values())
            llm_to_use, pattern_name = match_intent(
                user_message,
                all_tools,
                base_llm,
                logger,
                state
            )
        else:
            llm_to_use = llm_with_tools

        if hasattr(llm_to_use, 'model'):
            current_model = llm_to_use.model
        elif hasattr(llm_to_use, 'model_name'):
            current_model = llm_to_use.model_name
        elif hasattr(llm_to_use, 'model_path'):
            from pathlib import Path
            current_model = Path(llm_to_use.model_path).stem
        else:
            current_model = "unknown"

        logger.info(f"🧠 Calling LLM with {len(messages)} messages")
        logger.info(f"🤖 Model: {current_model}")

        sanitized_messages = []
        for msg in messages:
            content = msg.content if msg.content is not None else ""
            content = str(content) if not isinstance(content, str) else content

            if isinstance(msg, HumanMessage):
                sanitized_messages.append(HumanMessage(content=content))
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    sanitized_messages.append(AIMessage(content=content, tool_calls=msg.tool_calls))
                else:
                    sanitized_messages.append(AIMessage(content=content))
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if msg.name is not None else "unknown_tool"
                sanitized_messages.append(
                    ToolMessage(content=content, tool_call_id=msg.tool_call_id, name=tool_name))
            elif isinstance(msg, SystemMessage):
                sanitized_messages.append(SystemMessage(content=content))
            else:
                sanitized_messages.append(msg)

        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                llm_to_use.ainvoke(sanitized_messages),
                timeout=300.0
            )
            duration = time.time() - start_time

            if METRICS_AVAILABLE:
                metrics["llm_calls"] += 1
                metrics["llm_times"].append((time.time(), duration))

        except Exception as e:
            # CHECK FOR "DOES NOT SUPPORT TOOLS" ERROR
            error_msg_str = str(e).lower()
            if "does not support tools" in error_msg_str:
                logger.error(f"❌ Model '{current_model}' does not support tool calling")

                all_tools_list = list(state.get("tools", {}).values())
                tool_count = len(all_tools_list)

                error_response = AIMessage(content=f"""❌ **Model Error**: The model '{current_model}' does not support tool calling.

    This model cannot use the {tool_count} tools available in this system.

    **Recommended models with tool support:**
    • `qwen2.5:14b` - Best quality, excellent tools (recommended)
    • `qwen2.5:7b` - Fast, good balance
    • `llama3.1:8b` - Solid general purpose
    • `llama3.2:3b` - Lightweight option
    • `mistral-nemo` - Balanced performance

    **To switch models:**
    Type: `:model qwen2.5:14b`

    **To install a model:**
    Run: `ollama pull qwen2.5:14b-instruct-q4_K_M`

    **Current setup:**
    - Model: {current_model}
    - Tools available: {tool_count}
    - Tool support: ❌ Not supported""")

                return {
                    "messages": messages + [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": current_model
                }

            # Handle other exceptions
            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"❌ Model call failed: {e}")
            raise

        # Continue with normal flow - FALLBACK CHAIN
        try:
            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
            has_content = hasattr(response, 'content') and response.content and response.content.strip()

            if not has_tool_calls and not has_content:
                logger.warning("⚠️ LLM returned blank - trying LangSearch")
                langsearch = get_langsearch_client()

                if langsearch.is_available():
                    search_result = await langsearch.search(user_message)

                    if search_result["success"] and search_result["results"]:
                        logger.info("[LangGraph] ✅ LangSearch successful")
                        search_context = search_result["results"]
                        augmented_prompt = f"""WEB SEARCH RESULTS:
    {search_context}

    Please answer the question using these search results."""

                        retry_messages = messages + [HumanMessage(content=augmented_prompt)]
                        response = await base_llm.ainvoke(retry_messages)
                    else:
                        logger.warning("⚠️ LangSearch failed - using base LLM")
                        response = await asyncio.wait_for(
                            base_llm.ainvoke(messages),
                            timeout=300.0
                        )
                else:
                    logger.warning("⚠️ LangSearch unavailable - using base LLM")
                    response = await asyncio.wait_for(
                        base_llm.ainvoke(messages),
                        timeout=300.0
                    )

            elif not has_tool_calls and has_content:
                needs_current_info = any(word in user_message.lower() for word in [
                    "current", "who is", "latest", "recent", "today", "now"
                ])

                if needs_current_info:
                    logger.info("[LangGraph] 🔍 Trying LangSearch fallback for current info")
                    langsearch = get_langsearch_client()

                    if langsearch.is_available():
                        search_result = await langsearch.search(user_message)

                        if search_result["success"] and search_result["results"]:
                            logger.info("[LangGraph] ✅ LangSearch successful - augmenting")
                            search_context = search_result["results"]
                            augmented_prompt = f"""Previous answer: {response.content}

    However, here are current web search results:
    {search_context}

    Please provide an updated answer using these search results."""

                            retry_messages = messages + [response, HumanMessage(content=augmented_prompt)]
                            response = await base_llm.ainvoke(retry_messages)

            return {
                "messages": messages + [response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": state.get("stopped", False),
                "current_model": current_model
            }

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"⏱️ LLM call timed out after 5m")

            return {
                "messages": messages + [AIMessage(
                    content="⏱️ Request timed out after 5 minutes. Please try:\n\n1. Rephrasing your question\n2. Breaking it into smaller parts\n3. Using a simpler query")],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": state.get("stopped", False),
                "current_model": current_model
            }

        except Exception as e:
            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"❌ Model call failed: {e}")
            raise

    async def ingest_node(state: AgentState):
        """Handle ingestion operations"""
        if is_stop_requested():
            logger.warning("🛑 ingest_node: Stop requested")
            msg = AIMessage(content="Ingestion cancelled by user.")
            return {
                "messages": state["messages"] + [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": True
            }

        tools_dict = state.get("tools", {})
        ingest_tool = None
        for tool in tools_dict.values():
            if hasattr(tool, 'name') and tool.name == "plex_ingest_batch":
                ingest_tool = tool
                break

        if not ingest_tool:
            msg = AIMessage(content="Ingestion tool not available.")
            return {
                "messages": state["messages"] + [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }

        try:
            logger.info("[LangGraph] 📥 Starting ingest operation...")
            result = await ingest_tool.ainvoke({"limit": 5})
            msg = AIMessage(content=f"Ingestion complete: {result}")
            return {
                "messages": state["messages"] + [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }
        except Exception as e:
            logger.error(f"❌ Error in ingest_node: {e}")
            msg = AIMessage(content=f"Ingestion failed: {str(e)}")
            return {
                "messages": state["messages"] + [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }

    async def call_tools_with_stop_check(state: AgentState):
        """Execute tools with stop signal checking"""
        logger = logging.getLogger("mcp_client")

        if is_stop_requested():
            logger.warning("🛑 call_tools: Stop requested")
            empty_response = AIMessage(content="Tool execution cancelled by user.")
            return {
                "messages": state["messages"] + [empty_response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": True
            }

        from langchain_core.messages import ToolMessage
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            logger.warning("⚠️ No tool calls found")
            return state

        # Extract context from system messages for auto-fix
        context = {}
        for msg in reversed(state["messages"][-10:]):
            if isinstance(msg, SystemMessage) and "CONVERSATION CONTEXT" in msg.content:
                match = re.search(r'Active Project:\s*(.+?)(?:\n|$)', msg.content)
                if match:
                    context["project_path"] = match.group(1).strip()
                    logger.info(f"🔧 Context found: {context['project_path']}")
                break

        tool_messages = []
        for tool_call in tool_calls:
            if is_stop_requested():
                logger.warning(f"🛑 Stop requested - halting tool calls")
                break

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            # AUTO-FIX tool arguments
            if context.get("project_path") and tool_name in ["get_project_dependencies", "analyze_code_file", "analyze_project", "scan_project_structure"]:
                if "project_path" not in tool_args:
                    logger.warning(f"🔧 AUTO-FIX: Adding missing project_path → '{context['project_path']}'")
                    tool_args["project_path"] = context["project_path"]
                    tool_call["args"]["project_path"] = context["project_path"]
                elif tool_args.get("project_path") in [".", "./"]:
                    logger.warning(f"🔧 AUTO-FIX: Replacing '.' → '{context['project_path']}'")
                    tool_args["project_path"] = context["project_path"]
                    tool_call["args"]["project_path"] = context["project_path"]

                logger.info(f"🔍 Final tool_args: {tool_args}")

            logger.info(f"🔧 Executing tool: {tool_name}")

            tools_dict = state.get("tools", {})
            tool = tools_dict.get(tool_name)

            if not tool:
                logger.error(f"❌ Tool '{tool_name}' not found")
                error_msg = ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)
                continue

            try:
                tool_start = time.time()
                result = await tool.ainvoke(tool_args)
                tool_duration = time.time() - tool_start

                if METRICS_AVAILABLE:
                    metrics["tool_calls"][tool_name] += 1
                    metrics["tool_times"][tool_name].append((time.time(), tool_duration))

                if isinstance(result, list) and len(result) > 0:
                    if hasattr(result[0], 'text'):
                        result = result[0].text

                result_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(result_msg)
                logger.info(f"✅ Tool {tool_name} completed in {tool_duration:.2f}s")

            except Exception as e:
                logger.error(f"❌ Tool {tool_name} failed: {e}")
                if METRICS_AVAILABLE:
                    metrics["tool_errors"][tool_name] += 1
                error_msg = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)

        return {
            "messages": state["messages"] + tool_messages,
            "tools": state.get("tools", {}),
            "llm": state.get("llm"),
            "ingest_completed": state.get("ingest_completed", False),
            "stopped": is_stop_requested()
        }

    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools_with_stop_check)
    workflow.add_node("rag", rag_node)
    workflow.add_node("ingest", ingest_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "rag": "rag",
            "ingest": "ingest",
            "continue": END
        }
    )
    workflow.add_edge("tools", "agent")
    workflow.add_edge("ingest", END)
    workflow.add_edge("rag", END)

    app = workflow.compile()
    logger.info("[LangGraph] ✅ LangGraph agent compiled successfully")
    return app


async def run_agent(agent, conversation_state, user_message, logger, tools, system_prompt, llm=None, max_history=20):
    """
    Execute the agent with the given user message and track metrics

    CONVERSATION HISTORY DESIGN:
    - The SystemMessage in conversation_state["messages"][0] is the source of truth
    - We preserve it through truncation and never recreate it
    - If no SystemMessage exists, we create one from system_prompt parameter
    """
    start_time = time.time()
    clear_stop()
    logger.info("[LangGraph] ✅ Stop signal cleared for new request")

    try:
        if METRICS_AVAILABLE:
            metrics["agent_runs"] += 1

        # ═══════════════════════════════════════════════════════════════
        # STEP 1: Save the original SystemMessage (if it exists)
        # ═══════════════════════════════════════════════════════════════
        original_system_msg = None
        has_system_msg = (
                conversation_state["messages"]
                and isinstance(conversation_state["messages"][0], SystemMessage)
        )

        if has_system_msg:
            # Preserve the existing SystemMessage (may have enhancements)
            original_system_msg = conversation_state["messages"][0]
            logger.info("[LangGraph] Preserving existing SystemMessage")
        else:
            # No SystemMessage exists - create one from parameter
            original_system_msg = SystemMessage(content=system_prompt)
            conversation_state["messages"].insert(0, original_system_msg)
            logger.info("[LangGraph] Created new SystemMessage from parameter")

        # ═══════════════════════════════════════════════════════════════
        # STEP 2: Add user message and truncate
        # ═══════════════════════════════════════════════════════════════
        conversation_state["messages"].append(HumanMessage(content=user_message))

        # Truncate but keep the system message separate
        system_msg = conversation_state["messages"][0]  # Save it
        other_messages = conversation_state["messages"][1:]  # Everything else

        # Truncate the other messages
        if len(other_messages) > max_history - 1:  # -1 for system message
            other_messages = other_messages[-(max_history - 1):]
            logger.info(f"[LangGraph] Truncated to last {max_history} messages")

        # Reconstruct: system message + truncated messages
        conversation_state["messages"] = [system_msg] + other_messages

        # ═══════════════════════════════════════════════════════════════
        # STEP 3: Run the agent
        # ═══════════════════════════════════════════════════════════════
        logger.info(f"🧠 Starting agent with {len(conversation_state['messages'])} messages")

        tool_registry = {tool.name: tool for tool in tools}

        result = await agent.ainvoke({
            "messages": conversation_state["messages"],
            "tools": tool_registry,
            "llm": llm,
            "ingest_completed": False,
            "stopped": False
        })

        # ═══════════════════════════════════════════════════════════════
        # STEP 4: Update conversation state
        # ═══════════════════════════════════════════════════════════════
        new_messages = result["messages"][len(conversation_state["messages"]):]
        logger.info(f"📨 Agent added {len(new_messages)} new messages")
        conversation_state["messages"].extend(new_messages)

        # ═══════════════════════════════════════════════════════════════
        # STEP 5: Return results
        # ═══════════════════════════════════════════════════════════════
        if METRICS_AVAILABLE:
            duration = time.time() - start_time
            metrics["agent_times"].append((time.time(), duration))

        final_model = result.get("current_model", "unknown")

        if METRICS_AVAILABLE:
            logger.info(f"✅ Agent run completed in {duration:.2f}s (Model: {final_model})")
        else:
            logger.info(f"✅ Agent run completed (Model: {final_model})")

        return {
            "messages": conversation_state["messages"],
            "current_model": final_model
        }

    except ValueError as e:
        # Handle context window overflow gracefully
        error_str = str(e)
        if "exceed context window" in error_str or "Requested tokens" in error_str:
            # Extract token counts from error message
            import re
            match = re.search(r'Requested tokens \((\d+)\) exceed context window of (\d+)', error_str)

            if match:
                requested = int(match.group(1))
                available = int(match.group(2))
                overflow = requested - available
                logger.error(
                    f"❌ Context overflow: {requested} tokens requested, {available} available ({overflow} over)")
            else:
                requested = None
                available = None
                logger.error(f"❌ Context window overflow")

            # Calculate how many messages to drop
            current_msg_count = len(conversation_state["messages"])
            if current_msg_count > 3:  # Keep at least system + user message
                # Drop half the history
                new_limit = max(3, current_msg_count // 2)
                logger.warning(f"⚠️  Auto-recovery: Reducing history from {current_msg_count} to {new_limit} messages")

                # Keep system message + trim history + keep latest user message
                system_msg = conversation_state["messages"][0] if isinstance(conversation_state["messages"][0],
                                                                             SystemMessage) else None
                user_msg = conversation_state["messages"][-1]
                middle_msgs = conversation_state["messages"][1:-1]

                # Take most recent middle messages
                trimmed_middle = middle_msgs[-(new_limit - 2):] if len(middle_msgs) > (new_limit - 2) else middle_msgs

                if system_msg:
                    conversation_state["messages"] = [system_msg] + trimmed_middle + [user_msg]
                else:
                    conversation_state["messages"] = trimmed_middle + [user_msg]

                error_msg = AIMessage(content=f"""⚠️ Context window overflow detected and auto-fixed.

**Issue:** Your conversation ({requested} tokens) exceeded the model's limit ({available} tokens).

**Auto-recovery:** Reduced history from {current_msg_count} to {len(conversation_state['messages'])} messages.

**Suggestions:**
1. Start a new chat
2. `:model qwen2.5:14b` - Switch to larger model (8K tokens)
3. Keep conversations shorter with small models

**You can retry your request now.**""")
            else:
                # Can't trim further - conversation too short but still overflowing
                logger.error(f"❌ Cannot auto-recover: conversation already minimal ({current_msg_count} messages)")
                error_msg = AIMessage(content=f"""❌ Context window overflow - this model is too small for your task.

**Problem:** Even a minimal conversation exceeds this model's {available if available else '?'} token limit.

**Solutions:**
1. Start a new chat session
2. `:model qwen2.5:14b` - Switch to larger model (8K context)
3. Use a model with more capacity

This model cannot handle your current workload.""")

            conversation_state["messages"].append(error_msg)

            if METRICS_AVAILABLE:
                metrics["agent_errors"] += 1
                duration = time.time() - start_time
                metrics["agent_times"].append((time.time(), duration))

            return {"messages": conversation_state["messages"]}

        # Re-raise if not context overflow
        raise

    except Exception as e:
        if METRICS_AVAILABLE:
            metrics["agent_errors"] += 1
            duration = time.time() - start_time
            metrics["agent_times"].append((time.time(), duration))

        logger.exception(f"❌ Unexpected error in agent execution")
        error_msg = AIMessage(content=f"An error occurred: {str(e)}")
        conversation_state["messages"].append(error_msg)
        return {"messages": conversation_state["messages"]}