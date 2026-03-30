"""
prompts.py — Central prompt library for mcp-platform
=====================================================
All LLM prompt strings live here. Static prompts are plain strings.
Dynamic prompts are templates — call .format(**kwargs) at the use site.

Usage
-----
    from client.prompts import VISION_DEFAULT, TOOL_RESULT_SUMMARISE

    # Static
    prompt = VISION_DEFAULT

    # Dynamic (f-string replacement)
    prompt = RESEARCH_SYNTHESIS.format(
        source_count=3,
        sources_list="...",
        combined_content="...",
        query="...",
    )
"""

# ═══════════════════════════════════════════════════════════════════
# VISION / IMAGE DESCRIPTION
# ═══════════════════════════════════════════════════════════════════

# Used by client.py direct dispatch for shashin_random_tool,
# shashin_analyze_tool, and analyze_image_tool when no user
# instruction is provided.
VISION_DEFAULT = (
    "Describe what's happening in this photo warmly and naturally. "
    "Start directly with the scene — no greetings, no preamble."
)

# Appended to VISION_DEFAULT when place/date metadata is available.
# Instructs the model to weave location in naturally, not tack it on.
VISION_LOCATION_INSTRUCTION = (
    " Weave the location into your description naturally — "
    "don't just state it as a fact at the end."
)

# Used by langgraph.py for the agent vision path (image in tool results).
VISION_DESCRIBE_PLAIN = (
    "Describe this image in plain prose. "
    "Do not use JSON, markdown code blocks, or structured formats. "
    "Write naturally as if explaining to someone who cannot see the image."
)

# ═══════════════════════════════════════════════════════════════════
# TOOL RESULT PRESENTATION
# ═══════════════════════════════════════════════════════════════════

# Used by client.py direct dispatch to present tool JSON to the user.
# {tool_name} is replaced at call time.
TOOL_RESULT_PRESENT = (
    "The tool '{tool_name}' returned the following output. "
    "Present the information clearly and completely. "
    "Include all relevant details — do not truncate or over-summarise. "
    "No preamble, no meta-commentary, no advice. "
    "Do not mention the tool name, JSON, or data structures. "
    "Use emojis and symbols exactly as they appear in the data — do not substitute or change them."
)

# ═══════════════════════════════════════════════════════════════════
# LIST-BUILDER SUMMARISATION
# ═══════════════════════════════════════════════════════════════════

# Used by client.py list-builder to summarise individual result items
# (RAG search results, browse items with real text content).
ITEM_SUMMARISE = (
    "Summarise the following in 1-2 sentences. "
    "Be specific and factual. No preamble."
)

# Used by client.py note summariser (search_notes results).
NOTE_SUMMARISE = (
    "Summarise the following note content in 2-3 sentences. "
    "Be specific and factual. Do not add commentary or preamble."
)

# ═══════════════════════════════════════════════════════════════════
# RAG CONTEXT INJECTION
# ═══════════════════════════════════════════════════════════════════

# Used by langgraph.py to inject RAG context as a SystemMessage.
# {context} is replaced at call time.
RAG_CONTEXT = "Context from RAG:\n\n{context}"

# ═══════════════════════════════════════════════════════════════════
# WEB SEARCH RESULT PRESENTATION
# ═══════════════════════════════════════════════════════════════════

# Simple web search augmentation — query + results, single source.
# {query} and {search_context} replaced at call time.
WEB_SEARCH_SIMPLE = (
    'Web search results for: "{query}"\n\n'
    "{search_context}\n\n"
    "Based on these search results, provide a clear answer."
)

# Full web search augmentation — includes user message for context.
# {search_context} and {user_message} replaced at call time.
WEB_SEARCH_WITH_QUESTION = (
    "I searched the web and found the following results:\n\n"
    "{search_context}\n\n"
    'Based on these search results, please answer the user\'s question: "{user_message}"\n\n'
    "Provide a clear, concise answer in English. "
    "Extract the most relevant information and present it naturally."
)

# Minimal web search augmentation used in retry path.
# {search_context} replaced at call time.
WEB_SEARCH_RESULTS = (
    "WEB SEARCH RESULTS:\n"
    "{search_context}\n\n"
    "Please answer the question using these search results."
)

# Web search retry — updates a previous answer with fresh results.
# {previous_answer} and {search_context} replaced at call time.
WEB_SEARCH_UPDATE = (
    "Previous answer: {previous_answer}\n\n"
    "However, here are current web search results:\n"
    "{search_context}\n\n"
    "Please provide an updated answer using these search results."
)

# ═══════════════════════════════════════════════════════════════════
# RESEARCH SYNTHESIS (multi-source fetch)
# ═══════════════════════════════════════════════════════════════════

# Primary research synthesis — full content from multiple sources.
# {source_count}, {sources_list}, {combined_content}, {query} replaced at call time.
RESEARCH_SYNTHESIS = """\
I have fetched content from {source_count} source(s):

{sources_list}

CONTENT FROM ALL SOURCES:
{combined_content}

Question: {query}

**Instructions:**
- Write a comprehensive answer synthesizing information from ALL {source_count} sources
- Cite each source using its ACTUAL URL (listed above)
- Example: "According to the Wikipedia article on Donald Trump (https://...), ..."
- When information comes from multiple sources, note this
- Include a References section listing all {source_count} sources at the end

Your answer:"""

# Intermediate summarisation step — condenses content before retry.
# {source_count}, {combined_content}, {query} replaced at call time.
RESEARCH_CONDENSE = """\
Summarize this content from {source_count} sources concisely, \
keeping key facts relevant to: "{query}"

{combined_content}

Provide a structured summary under 1000 words:"""

# Retry synthesis using already-summarised content.
# {source_count}, {sources_list}, {summarized_content}, {query} replaced at call time.
RESEARCH_RETRY = """\
I have fetched and SUMMARIZED content from {source_count} sources.

Sources:
{sources_list}

SUMMARIZED CONTENT:
{summarized_content}

Question: {query}

**Instructions:**
- Write a comprehensive answer based on the summary
- Cite each source by its URL
- Note this is based on summarized content

Your answer:"""
# ═══════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════

# Default fallback system prompt used when prompts/system_prompt.md is absent.
# Also used as the base when system_prompt.md IS present — HISTORY_AWARENESS
# is appended to the file content in that case.
SYSTEM_PROMPT_DEFAULT = """\
# SYSTEM INSTRUCTION: YOU ARE A TOOL-USING AGENT

CRITICAL RULES:
1. ALWAYS respond in ENGLISH only
2. Read the user's intent carefully before choosing a tool
3. DO NOT make multiple redundant tool calls

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU HAVE FULL ACCESS TO CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT say "I don't have access to history" - YOU DO HAVE ACCESS.
The message list contains the FULL conversation history in chronological order.

WHEN USER ASKS: "what was my last prompt"
→ Look at the most recent HumanMessage before the current one
→ Respond: "Your last prompt was: [exact text]"

Example:
User: "what's the weather?"
You: "Sunny, 22°C"
User: "what was my last prompt"
You: "Your last prompt was: what's the weather?"  ← DO THIS

TOOL SELECTION:

"add to my todo" → use add_todo_item (NOT rag_search_tool)
"remember this" → use rag_add_tool
"find a movie" → use semantic_media_search_text
"using the RAG tool" → use rag_search_tool (ONE search only)

EXAMPLES:

User: "add to my todo due tomorrow, make breakfast"
CORRECT: add_todo_item(title="make breakfast", due_by="[tomorrow date]")
WRONG: rag_search_tool(query="make breakfast") ❌

User: "remember that password is abc123"
CORRECT: rag_add_tool(text="password is abc123", source="notes")
WRONG: add_todo_item(title="password is abc123") ❌

VERIFICATION:
- "add to my todo" = add_todo_item
- "remember" = rag_add_tool
- "find movie" = semantic_media_search_text

Read the user's message carefully and call the RIGHT tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN / SEQUENCE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When analyzing sequences or patterns, you MUST:
1. Check your identified pattern against ALL terms, not just the first few
2. Verify the last few terms match your formula before responding
3. If any terms don't fit, revise your pattern — the sequence may have
   multiple phases with different rules (e.g. +3 for first half, +5 for second)\
"""

# Appended to system_prompt.md content (or SYSTEM_PROMPT_DEFAULT) at startup
# to ensure conversation history awareness regardless of which prompt file is used.
HISTORY_AWARENESS = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU HAVE FULL ACCESS TO CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT say "I don't have access to history" - YOU DO HAVE ACCESS.

WHEN USER ASKS: "what was my last prompt"
→ Look at the most recent HumanMessage before the current one
→ Respond: "Your last prompt was: [exact text]"

Example:
User: "what's the weather?"
You: "Sunny, 22°C"
User: "what was my last prompt"
You: "Your last prompt was: what's the weather?"  ← DO THIS\
"""

# ═══════════════════════════════════════════════════════════════════
# TOOL RESULT FORMATTING (LangGraph path)
# ═══════════════════════════════════════════════════════════════════

# Appended to the system message by langgraph.py when formatting tool
# results, to prevent the LLM substituting its own emojis/symbols for
# those already present in the tool output (e.g. ☂️ instead of ☁️).
TOOL_RESULT_FORMAT_EMOJI = (
    " Use emojis and symbols exactly as they appear in the tool output"
    " — do not substitute or change them."
)

# ═══════════════════════════════════════════════════════════════════
# VISION QUERY FORMAT INSTRUCTION
# ═══════════════════════════════════════════════════════════════════

# Appended to user queries sent to the vision model when the user
# provides an explicit question. Prevents over-formatted, nested,
# HTML-laden responses from vision models like qwen3-vl.
VISION_QUERY_FORMAT = (
    " Answer in plain text only. "
    "Use simple bullet points (- item) if listing things. "
    "No nested bullets, no HTML tags, no markdown headers, "
    "no 'Note:' qualifiers, no preamble."
)