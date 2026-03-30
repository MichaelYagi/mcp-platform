"""
Planner Agent - Creates task lists and schedules
"""

from typing import Optional
from .base_agent import BaseAgent, AgentMessage, MessageType


class PlannerAgent(BaseAgent):
    """
    Creates and manages task lists, schedules work
    """

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are a Planner Agent focused on organizing tasks.

Your tools:
- list_todo_items: List existing todo items
- add_todo_item: Add new todo items

Use todo tools to create, manage, and track task lists.
Help users organize their work effectively."""

        super().__init__(
            agent_id=agent_id,
            role="planner",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle planning requests"""
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
                metadata={"planning_completed": True}
            )
        return None