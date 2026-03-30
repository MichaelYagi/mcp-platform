"""
Orchestrator Agent - Plans and coordinates multi-agent workflows
"""

import json
from typing import Dict, List, Any, Optional
from .base_agent import BaseAgent, AgentMessage, MessageType


class OrchestratorAgent(BaseAgent):
    """
    Breaks down complex tasks into subtasks
    Routes work to appropriate specialized agents
    """

    def __init__(self, agent_id: str, llm, logger, message_bus):
        system_prompt = """You are an Orchestrator Agent coordinating multiple specialized agents.

When given a task, analyze it and create an execution plan with subtasks.

Available agents:
- researcher: Lists Plex items, searches knowledge base, gets weather/location info
- analyst: Filters and analyzes data
- planner: Creates task lists and plans
- plex_ingester: Ingests Plex media items into RAG database
- writer: Creates summaries and reports

Respond ONLY with JSON:
{
  "analysis": "Brief analysis of the task",
  "subtasks": [
    {
      "id": "task_1",
      "agent": "researcher",
      "description": "Detailed task description",
      "depends_on": []
    }
  ]
}

If task is simple (no coordination needed), return: {"subtasks": []}

Examples:

User: "Find 5 unprocessed Plex items and ingest them"
{
  "analysis": "Multi-step task: find items, then ingest each",
  "subtasks": [
    {
      "id": "task_1",
      "agent": "researcher",
      "description": "Find 5 unprocessed Plex items and return their IDs",
      "depends_on": []
    },
    {
      "id": "task_2",
      "agent": "plex_ingester",
      "description": "Ingest the items found in task_1",
      "depends_on": ["task_1"]
    }
  ]
}

User: "What's the weather?"
{"subtasks": []}
"""
        super().__init__(
            agent_id=agent_id,
            role="orchestrator",
            llm=llm,
            tools=[],  # No tools needed
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming messages"""
        if message.message_type == MessageType.REQUEST:
            # Another agent asking for coordination
            result = await self.execute_task(str(message.content))
            return AgentMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type=MessageType.RESPONSE,
                content=result,
                metadata={"original_request": message.content}
            )
        return None

    async def create_plan(self, user_request: str) -> Dict[str, Any]:
        """Create execution plan from user request"""
        self.logger.info(f"📋 [{self.agent_id}] Creating plan for: {user_request[:50]}...")

        result = await self.execute_task(f"Create execution plan for: {user_request}")

        # Parse JSON response
        try:
            # Clean response
            result_clean = result.strip()
            if result_clean.startswith("```json"):
                result_clean = result_clean[7:]
            if result_clean.startswith("```"):
                result_clean = result_clean[3:]
            if result_clean.endswith("```"):
                result_clean = result_clean[:-3]

            plan = json.loads(result_clean.strip())
            self.logger.info(f"✅ Plan created with {len(plan.get('subtasks', []))} subtasks")
            return plan
        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️ Failed to parse plan: {e}. Using fallback.")
            return {"subtasks": []}