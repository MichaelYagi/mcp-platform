"""
Analyst Agent - Filters and analyzes data
"""

from typing import Optional
from .base_agent import BaseAgent, AgentMessage, MessageType


class AnalystAgent(BaseAgent):
    """
    Analyzes data, filters results, provides insights
    """

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are an Analyst Agent focused on data filtering and analysis.

Your tools:
- rag_search_tool: Search and analyze knowledge base
- search_entries: Search structured data

Use your tools to gather data, then provide insights and patterns.
Focus on finding trends, anomalies, and meaningful conclusions."""

        super().__init__(
            agent_id=agent_id,
            role="analyst",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle analysis requests"""
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
                metadata={"analysis_completed": True}
            )
        return None