"""
Advanced Message Router for A2A System
Handles sophisticated routing, priority queues, and message protocols
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import heapq


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 0  # System-critical messages
    HIGH = 1  # Urgent requests
    NORMAL = 2  # Standard messages
    LOW = 3  # Background tasks
    BULK = 4  # Batch operations


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "direct"  # Send to specific agent
    BROADCAST = "broadcast"  # Send to all agents
    ROUND_ROBIN = "round_robin"  # Distribute evenly
    LOAD_BALANCED = "balanced"  # Send to least busy
    SKILL_BASED = "skill_based"  # Route by capability


@dataclass
class MessageEnvelope:
    """
    Enhanced message envelope with routing metadata
    """
    message_id: str
    from_agent: str
    to_agent: Optional[str]
    content: Any
    priority: MessagePriority
    routing_strategy: RoutingStrategy
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    timeout: float = 60.0

    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.timestamp < other.timestamp


class MessageRouter:
    """
    Sophisticated message router with multiple routing strategies
    """

    def __init__(self, logger):
        self.logger = logger

        # Priority queues for each agent
        self.agent_queues: Dict[str, List] = defaultdict(list)

        # Message tracking
        self.pending_messages: Dict[str, MessageEnvelope] = {}
        self.completed_messages: Dict[str, Any] = {}

        # Routing state
        self.round_robin_index = 0
        self.agent_registry: Dict[str, Any] = {}

        # Metrics
        self.routing_stats = {
            "total_routed": 0,
            "failed_routes": 0,
            "retries": 0,
            "timeouts": 0
        }

    def register_agent(self, agent_id: str, agent):
        """Register an agent with the router"""
        self.agent_registry[agent_id] = agent
        self.logger.info(f"📡 Registered agent: {agent_id}")

    def unregister_agent(self, agent_id: str):
        """Remove agent from router"""
        if agent_id in self.agent_registry:
            del self.agent_registry[agent_id]
            self.logger.info(f"📡 Unregistered agent: {agent_id}")

    async def route_message(self, envelope: MessageEnvelope) -> bool:
        """
        Route message based on strategy
        Returns True if successfully routed
        """
        self.routing_stats["total_routed"] += 1

        try:
            if envelope.routing_strategy == RoutingStrategy.DIRECT:
                return await self._route_direct(envelope)

            elif envelope.routing_strategy == RoutingStrategy.BROADCAST:
                return await self._route_broadcast(envelope)

            elif envelope.routing_strategy == RoutingStrategy.ROUND_ROBIN:
                return await self._route_round_robin(envelope)

            elif envelope.routing_strategy == RoutingStrategy.LOAD_BALANCED:
                return await self._route_load_balanced(envelope)

            elif envelope.routing_strategy == RoutingStrategy.SKILL_BASED:
                return await self._route_skill_based(envelope)

            else:
                self.logger.warning(f"⚠️ Unknown routing strategy: {envelope.routing_strategy}")
                return False

        except Exception as e:
            self.logger.error(f"❌ Routing failed: {e}")
            self.routing_stats["failed_routes"] += 1
            return False

    async def _route_direct(self, envelope: MessageEnvelope) -> bool:
        """Route directly to specified agent"""
        if not envelope.to_agent:
            self.logger.warning("⚠️ Direct routing requires to_agent")
            return False

        if envelope.to_agent not in self.agent_registry:
            self.logger.warning(f"⚠️ Agent {envelope.to_agent} not found")
            return False

        # Add to agent's priority queue
        heapq.heappush(self.agent_queues[envelope.to_agent], envelope)
        self.pending_messages[envelope.message_id] = envelope

        self.logger.debug(f"📨 Routed message {envelope.message_id} to {envelope.to_agent}")
        return True

    async def _route_broadcast(self, envelope: MessageEnvelope) -> bool:
        """Broadcast to all agents"""
        if not self.agent_registry:
            return False

        for agent_id in self.agent_registry.keys():
            # Create copy for each agent
            agent_envelope = MessageEnvelope(
                message_id=f"{envelope.message_id}_{agent_id}",
                from_agent=envelope.from_agent,
                to_agent=agent_id,
                content=envelope.content,
                priority=envelope.priority,
                routing_strategy=RoutingStrategy.DIRECT,
                timestamp=envelope.timestamp,
                metadata=envelope.metadata.copy()
            )
            heapq.heappush(self.agent_queues[agent_id], agent_envelope)

        self.logger.debug(f"📡 Broadcast message {envelope.message_id} to {len(self.agent_registry)} agents")
        return True

    async def _route_round_robin(self, envelope: MessageEnvelope) -> bool:
        """Distribute messages evenly across agents"""
        if not self.agent_registry:
            return False

        agents = list(self.agent_registry.keys())
        selected_agent = agents[self.round_robin_index % len(agents)]
        self.round_robin_index += 1

        envelope.to_agent = selected_agent
        return await self._route_direct(envelope)

    async def _route_load_balanced(self, envelope: MessageEnvelope) -> bool:
        """Route to least busy agent"""
        if not self.agent_registry:
            return False

        # Find agent with smallest queue
        min_load = float('inf')
        selected_agent = None

        for agent_id, agent in self.agent_registry.items():
            queue_size = len(self.agent_queues[agent_id])
            is_busy = agent.is_busy if hasattr(agent, 'is_busy') else False

            # Calculate load score (queue size + busy penalty)
            load = queue_size + (100 if is_busy else 0)

            if load < min_load:
                min_load = load
                selected_agent = agent_id

        if selected_agent:
            envelope.to_agent = selected_agent
            return await self._route_direct(envelope)

        return False

    async def _route_skill_based(self, envelope: MessageEnvelope) -> bool:
        """Route based on agent capabilities"""
        required_skill = envelope.metadata.get("required_skill")

        if not required_skill:
            # Fallback to load balanced
            return await self._route_load_balanced(envelope)

        # Find agents with required skill
        capable_agents = []
        for agent_id, agent in self.agent_registry.items():
            agent_role = agent.role if hasattr(agent, 'role') else None
            if agent_role == required_skill:
                capable_agents.append(agent_id)

        if not capable_agents:
            self.logger.warning(f"⚠️ No agents with skill: {required_skill}")
            return False

        # Pick least busy capable agent
        min_load = float('inf')
        selected_agent = None

        for agent_id in capable_agents:
            queue_size = len(self.agent_queues[agent_id])
            if queue_size < min_load:
                min_load = queue_size
                selected_agent = agent_id

        if selected_agent:
            envelope.to_agent = selected_agent
            return await self._route_direct(envelope)

        return False

    async def get_next_message(self, agent_id: str) -> Optional[MessageEnvelope]:
        """Get next message for agent (priority-based)"""
        queue = self.agent_queues.get(agent_id, [])

        if queue:
            envelope = heapq.heappop(queue)

            # Check timeout
            if time.time() - envelope.timestamp > envelope.timeout:
                self.logger.warning(f"⏰ Message {envelope.message_id} timed out")
                self.routing_stats["timeouts"] += 1
                return None

            return envelope

        return None

    def mark_complete(self, message_id: str, result: Any):
        """Mark message as completed"""
        if message_id in self.pending_messages:
            envelope = self.pending_messages.pop(message_id)
            self.completed_messages[message_id] = {
                "envelope": envelope,
                "result": result,
                "completed_at": time.time()
            }

    def mark_failed(self, message_id: str, error: str):
        """Mark message as failed and potentially retry"""
        if message_id in self.pending_messages:
            envelope = self.pending_messages[message_id]

            if envelope.retry_count < envelope.max_retries:
                envelope.retry_count += 1
                self.routing_stats["retries"] += 1
                self.logger.info(
                    f"🔄 Retrying message {message_id} (attempt {envelope.retry_count}/{envelope.max_retries})")

                # Re-route with same strategy
                asyncio.create_task(self.route_message(envelope))
            else:
                self.logger.error(f"❌ Message {message_id} failed after {envelope.max_retries} retries: {error}")
                self.pending_messages.pop(message_id)
                self.routing_stats["failed_routes"] += 1

    def get_queue_status(self) -> Dict[str, Any]:
        """Get status of all queues"""
        status = {}
        for agent_id, queue in self.agent_queues.items():
            status[agent_id] = {
                "queue_size": len(queue),
                "pending": len([e for e in queue if e.message_id in self.pending_messages]),
                "priorities": {
                    "critical": len([e for e in queue if e.priority == MessagePriority.CRITICAL]),
                    "high": len([e for e in queue if e.priority == MessagePriority.HIGH]),
                    "normal": len([e for e in queue if e.priority == MessagePriority.NORMAL]),
                    "low": len([e for e in queue if e.priority == MessagePriority.LOW]),
                    "bulk": len([e for e in queue if e.priority == MessagePriority.BULK])
                }
            }
        return status

    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            **self.routing_stats,
            "pending_messages": len(self.pending_messages),
            "completed_messages": len(self.completed_messages),
            "registered_agents": len(self.agent_registry)
        }

    def clear_old_messages(self, max_age_seconds: float = 3600):
        """Clear old completed messages"""
        current_time = time.time()
        to_remove = []

        for msg_id, data in self.completed_messages.items():
            if current_time - data["completed_at"] > max_age_seconds:
                to_remove.append(msg_id)

        for msg_id in to_remove:
            del self.completed_messages[msg_id]

        if to_remove:
            self.logger.debug(f"🧹 Cleaned up {len(to_remove)} old messages")


class MessageProtocol:
    """
    Standard message protocols for agent communication
    """

    @staticmethod
    def create_request(from_agent: str, to_agent: str, action: str, params: Dict = None,
                       priority: MessagePriority = MessagePriority.NORMAL) -> MessageEnvelope:
        """Create a standard request message"""
        import uuid
        return MessageEnvelope(
            message_id=f"req_{uuid.uuid4().hex[:8]}",
            from_agent=from_agent,
            to_agent=to_agent,
            content={
                "type": "request",
                "action": action,
                "params": params or {}
            },
            priority=priority,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time(),
            metadata={"protocol": "request"}
        )

    @staticmethod
    def create_response(original_message: MessageEnvelope, result: Any,
                        success: bool = True) -> MessageEnvelope:
        """Create a response to a request"""
        import uuid
        return MessageEnvelope(
            message_id=f"res_{uuid.uuid4().hex[:8]}",
            from_agent=original_message.to_agent,
            to_agent=original_message.from_agent,
            content={
                "type": "response",
                "original_id": original_message.message_id,
                "result": result,
                "success": success
            },
            priority=original_message.priority,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time(),
            metadata={"protocol": "response", "original_id": original_message.message_id}
        )

    @staticmethod
    def create_negotiation(from_agent: str, to_agent: str, proposal: Dict,
                           priority: MessagePriority = MessagePriority.HIGH) -> MessageEnvelope:
        """Create a negotiation message"""
        import uuid
        return MessageEnvelope(
            message_id=f"neg_{uuid.uuid4().hex[:8]}",
            from_agent=from_agent,
            to_agent=to_agent,
            content={
                "type": "negotiation",
                "proposal": proposal,
                "status": "proposed"
            },
            priority=priority,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time(),
            metadata={"protocol": "negotiation"}
        )

    @staticmethod
    def create_broadcast(from_agent: str, announcement: str,
                         priority: MessagePriority = MessagePriority.LOW) -> MessageEnvelope:
        """Create a broadcast message"""
        import uuid
        return MessageEnvelope(
            message_id=f"brd_{uuid.uuid4().hex[:8]}",
            from_agent=from_agent,
            to_agent=None,
            content={
                "type": "broadcast",
                "announcement": announcement
            },
            priority=priority,
            routing_strategy=RoutingStrategy.BROADCAST,
            timestamp=time.time(),
            metadata={"protocol": "broadcast"}
        )