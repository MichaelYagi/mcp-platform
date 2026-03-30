"""
Base Agent - Foundation for all A2A agents
Handles LLM invocation, tool calling, and inter-agent messaging
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"  # Agent requests another agent's help
    RESPONSE = "response"  # Agent responds to request
    BROADCAST = "broadcast"  # Agent broadcasts to all
    NOTIFICATION = "notification"  # Status update


@dataclass
class AgentMessage:
    """Message passed between agents"""
    from_agent: str
    to_agent: Optional[str]  # None = broadcast
    message_type: MessageType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None


class BaseAgent:
    """
    Base class for all specialized agents
    Provides LLM access, tool calling, and A2A messaging
    """

    def __init__(
            self,
            agent_id: str,
            role: str,
            llm,
            tools: List,
            system_prompt: str,
            logger: logging.Logger,
            message_bus: Optional[Callable] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.system_prompt = system_prompt
        self.logger = logger
        self.message_bus = message_bus  # Callback to send messages to other agents

        # State management
        self.context: Dict[str, Any] = {}
        self.message_history: List[AgentMessage] = []
        self.is_busy = False

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming message - override in subclasses
        Returns response message or None
        """
        raise NotImplementedError("Subclasses must implement process_message")

    async def execute_task(self, task_description: str, context: Dict = None) -> str:
        """
        Execute a task using LLM and tools
        This is the main execution method
        """
        self.is_busy = True
        self.logger.info(f"🤖 [{self.agent_id}] Executing: {task_description[:50]}...")

        try:
            # Build context
            context_str = self._build_context_string(context or {})

            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"{task_description}\n\n{context_str}")
            ]

            # Invoke LLM
            response = await self.llm.ainvoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)

            self.logger.info(f"✅ [{self.agent_id}] Task completed")
            return result

        except Exception as e:
            self.logger.error(f"❌ [{self.agent_id}] Task failed: {e}")
            raise
        finally:
            self.is_busy = False

    async def call_tool(self, tool_name: str, **kwargs) -> Any:
        """Call a tool by name with parameters"""
        if tool_name not in self.tools:
            raise ValueError(f"Tool {tool_name} not available to {self.agent_id}")

        tool = self.tools[tool_name]
        self.logger.info(f"🔧 [{self.agent_id}] Calling tool: {tool_name}")

        # Tools are async-compatible via ainvoke
        result = await tool.ainvoke(kwargs)
        return result

    async def send_message(
            self,
            to_agent: Optional[str],
            message_type: MessageType,
            content: Any,
            metadata: Dict = None
    ):
        """Send message to another agent or broadcast"""
        if not self.message_bus:
            self.logger.warning(f"[{self.agent_id}] No message bus available")
            return

        message = AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            timestamp=time.time()
        )

        self.message_history.append(message)
        await self.message_bus(message)

    def _build_context_string(self, context: Dict) -> str:
        """Build context string from dict"""
        if not context:
            return ""

        parts = []
        for key, value in context.items():
            parts.append(f"{key}: {value}")

        return "Context:\n" + "\n".join(parts)

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "is_busy": self.is_busy,
            "tools": list(self.tools.keys()),
            "messages_sent": len(self.message_history)
        }