"""
Researcher Agent - Finds and searches Plex items
"""

from typing import Optional
from .base_agent import BaseAgent, AgentMessage, MessageType


class ResearcherAgent(BaseAgent):
    """
    Searches for Plex items, queries knowledge base
    """

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are a Researcher Agent focused on finding information.

Your tools:
- plex_find_unprocessed: Find unprocessed Plex items
- rag_search_tool: Search processed media in knowledge base
- search_semantic: Semantic search across database
- semantic_media_search_text: Search media by description
- get_weather_tool: Get weather information

Always use tools to gather real data. Never make up information.
Be thorough and accurate in your research.

When searching for Plex items, use plex_find_unprocessed with a limit parameter.
Return the actual IDs and names of items found."""

        super().__init__(
            agent_id=agent_id,
            role="researcher",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

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