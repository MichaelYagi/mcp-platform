"""
Researcher Agent - Finds and searches Plex items and knowledge base.
Tools are filtered at construction time: rag_search_tool and search_semantic
are only included when the task description signals personal/ingested content.
"""

from typing import Optional, List
from .base_agent import BaseAgent, AgentMessage, MessageType

# Keywords that indicate the query is about personal/ingested content
# and therefore warrants RAG tool access.
_RAG_SIGNAL_WORDS = frozenset([
    "my notes", "what did i", "what have i", "from my", "i stored",
    "in my rag", "knowledge base", "past conversation", "i ingested",
    "i saved", "my research", "i wrote", "i mentioned",
])

# Tool names that should only be available for personal/RAG queries
_RAG_TOOLS = frozenset([
    "rag_search_tool",
    "search_semantic",
    "search_entries",
    "semantic_media_search_text",
])


def _is_rag_query(task_description: str) -> bool:
    """Return True only if the task is clearly about personal/ingested content."""
    lower = task_description.lower()
    return any(signal in lower for signal in _RAG_SIGNAL_WORDS)


class ResearcherAgent(BaseAgent):
    """
    Searches for Plex items, queries knowledge base.
    RAG tools are withheld for general-knowledge queries to prevent the model
    looping on fruitless searches.
    """

    # Full tool list supplied at construction — we filter per-task in execute_task
    _all_tools: list = []

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are a Researcher Agent focused on finding information.

TOOL USAGE — follow these rules strictly:

Use plex_find_unprocessed ONLY when the task is about finding unprocessed Plex media items.

Use rag_search_tool / search_semantic ONLY when the task explicitly asks about:
  - Personal notes or saved content
  - Past conversations
  - Previously ingested documents
  - Queries containing phrases like "my notes", "what did I store", "from my RAG"

For ALL other topics — programming concepts, general knowledge, public information,
technology explanations, current events — answer directly from your own knowledge.
DO NOT call any search tools for general knowledge questions.
Calling RAG for topics like "asyncio", "python", "docker", "AI" wastes time and
returns irrelevant results. Just answer from what you know.

Use get_weather_tool only when weather information is explicitly requested.

When searching for Plex items, use plex_find_unprocessed with a limit parameter.
Return the actual IDs and names of items found."""

        # Store full tool list before filtering
        self._all_tools = list(tools) if tools else []

        super().__init__(
            agent_id=agent_id,
            role="researcher",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def execute_task(self, task_description: str, context: dict = None):
        """Execute research task with RAG tools withheld for general-knowledge queries."""
        is_rag = _is_rag_query(task_description)

        if not is_rag:
            # Filter out RAG tools — give the model no choice but to answer directly
            filtered_tools = [t for t in self._all_tools
                              if getattr(t, 'name', str(t)) not in _RAG_TOOLS]
            original_tools = self.tools
            self.tools = filtered_tools
            self.logger.info(
                f"🔬 Researcher: general-knowledge query — RAG tools withheld "
                f"({len(original_tools) - len(filtered_tools)} tool(s) suppressed)"
            )
            try:
                return await super().execute_task(task_description, context)
            finally:
                self.tools = original_tools  # always restore
        else:
            self.logger.info("🔬 Researcher: RAG query detected — all tools available")
            return await super().execute_task(task_description, context)

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle research requests"""
        if message.message_type == MessageType.REQUEST:
            result = await self.execute_task(
                str(message.content),
                context=message.metadata
            )

            return AgentMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type=MessageType.RESPONSE,
                content=result,
                metadata={"research_completed": True}
            )
        return None