"""
Enhanced Base Agent - Foundation for A2A agents with advanced features
Integrates message routing, health monitoring, negotiation, and performance metrics
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
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    NEGOTIATION = "negotiation"


@dataclass
class AgentMessage:
    """Message passed between agents"""
    from_agent: str
    to_agent: Optional[str]
    message_type: MessageType
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[float] = None


class EnhancedBaseAgent:
    """
    Enhanced base class for all specialized agents
    Includes health monitoring, performance tracking, and negotiation support
    """

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm,
        tools: List,
        system_prompt: str,
        logger: logging.Logger,
        message_bus: Optional[Callable] = None,
        health_monitor=None,
        performance_metrics=None,
        negotiation_engine=None
    ):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        self.system_prompt = system_prompt
        self.logger = logger
        self.message_bus = message_bus

        # Advanced features
        self.health_monitor = health_monitor
        self.performance_metrics = performance_metrics
        self.negotiation_engine = negotiation_engine

        # State management
        self.context: Dict[str, Any] = {}
        self.message_history: List[AgentMessage] = []
        self.is_busy = False

        # Register with health monitor
        if self.health_monitor:
            self.health_monitor.register_agent(self.agent_id)

        # Heartbeat task
        self.heartbeat_task = None
        self.start_heartbeat()

    def start_heartbeat(self):
        """Start sending heartbeats to health monitor"""
        if self.health_monitor and not self.heartbeat_task:
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                if self.health_monitor:
                    self.health_monitor.heartbeat(self.agent_id)
                await asyncio.sleep(5.0)  # Heartbeat every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process incoming message - override in subclasses
        Returns response message or None
        """
        raise NotImplementedError("Subclasses must implement process_message")

    async def execute_task(self, task_description: str, context: Dict = None) -> str:
        """
        Execute a task using LLM and tools with full monitoring
        """
        self.is_busy = True
        start_time = time.time()
        task_id = f"task_{int(start_time * 1000)}"

        self.logger.info(f"🤖 [{self.agent_id}] Executing: {task_description[:50]}...")

        # Update health monitor
        if self.health_monitor:
            self.health_monitor.update_resource_usage(
                self.agent_id,
                queue_size=len(self.message_history)
            )

        success = False
        error = None
        tools_used = []
        llm_calls = 0

        try:
            # Build context
            context_str = self._build_context_string(context or {})

            # Create messages
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=f"{task_description}\n\n{context_str}")
            ]

            # Invoke LLM
            llm_calls += 1
            response = await self.llm.ainvoke(messages)
            result = response.content if hasattr(response, 'content') else str(response)

            success = True
            self.logger.info(f"✅ [{self.agent_id}] Task completed")

            return result

        except Exception as e:
            error = str(e)
            self.logger.error(f"❌ [{self.agent_id}] Task failed: {e}")

            # Record error with health monitor
            if self.health_monitor:
                self.health_monitor.record_error(self.agent_id, error)

            raise

        finally:
            self.is_busy = False
            end_time = time.time()
            duration = end_time - start_time

            # Record performance metrics
            if self.performance_metrics:
                from performance_metrics import TaskMetrics

                task_metrics = TaskMetrics(
                    task_id=task_id,
                    agent_id=self.agent_id,
                    task_type=self.role,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    success=success,
                    tools_used=tools_used,
                    llm_calls=llm_calls,
                    tokens_used=0,  # Would need to track from LLM
                    error=error
                )

                self.performance_metrics.record_task(task_metrics)

            # Record with health monitor
            if self.health_monitor:
                self.health_monitor.record_task_completion(
                    self.agent_id,
                    duration,
                    success
                )

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

    async def negotiate_task(self, target_agent: str, task_description: str,
                           urgency: str = "normal") -> Optional[str]:
        """
        Negotiate with another agent to handle a task
        Returns negotiation result or None
        """
        if not self.negotiation_engine:
            self.logger.warning(f"[{self.agent_id}] Negotiation engine not available")
            return None

        from negotiation_engine import NegotiationType

        # Create negotiation proposal
        proposal = self.negotiation_engine.propose(
            initiator=self.agent_id,
            target=target_agent,
            negotiation_type=NegotiationType.TASK_ALLOCATION,
            terms={
                "task_description": task_description,
                "urgency": urgency,
                "offered_compensation": "Will help with your next task"
            }
        )

        self.logger.info(f"🤝 [{self.agent_id}] Proposed negotiation to {target_agent}")

        # Wait for response (with timeout)
        timeout = 10.0
        start_wait = time.time()

        while time.time() - start_wait < timeout:
            status = self.negotiation_engine.get_negotiation_status(proposal.proposal_id)

            if status.status.value == "accepted":
                self.logger.info(f"✅ Negotiation accepted by {target_agent}")
                return "accepted"
            elif status.status.value == "rejected":
                self.logger.info(f"❌ Negotiation rejected by {target_agent}")
                return "rejected"

            await asyncio.sleep(0.5)

        self.logger.warning(f"⏰ Negotiation timed out with {target_agent}")
        return "timeout"

    def _build_context_string(self, context: Dict) -> str:
        """Build context string from dict"""
        if not context:
            return ""

        parts = []
        for key, value in context.items():
            parts.append(f"{key}: {value}")

        return "Context:\n" + "\n".join(parts)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status"""
        status = {
            "agent_id": self.agent_id,
            "role": self.role,
            "is_busy": self.is_busy,
            "tools": list(self.tools.keys()),
            "messages_sent": len(self.message_history)
        }

        # Add health metrics if available
        if self.health_monitor:
            health = self.health_monitor.get_agent_health(self.agent_id)
            if health:
                status["health"] = {
                    "status": health.status.value,
                    "error_count": health.error_count,
                    "queue_size": health.queue_size,
                    "avg_response_time": health.avg_response_time
                }

        # Add performance metrics if available
        if self.performance_metrics:
            perf = self.performance_metrics.get_agent_performance(self.agent_id)
            if perf:
                status["performance"] = {
                    "total_tasks": perf.total_tasks,
                    "success_rate": perf.successful_tasks / perf.total_tasks if perf.total_tasks > 0 else 0,
                    "avg_duration": perf.avg_duration
                }

        return status

    def shutdown(self):
        """Cleanup on shutdown"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()

        if self.health_monitor:
            self.health_monitor.unregister_agent(self.agent_id)