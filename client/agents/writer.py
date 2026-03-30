"""
Writer Agent - Creates summaries and reports
"""

from typing import Optional
from .base_agent import BaseAgent, AgentMessage, MessageType


class WriterAgent(BaseAgent):
    """
    Creates summaries, reports, and written content
    """

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are a Writer Agent focused on clear communication.

Your tools:
- rag_search_tool: Gather information for writing
- search_entries: Find specific details

Gather information using tools, then create well-structured summaries.
Focus on clarity, completeness, and good organization."""

        super().__init__(
            agent_id=agent_id,
            role="writer",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle writing requests"""
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
                metadata={"writing_completed": True}
            )
        return None