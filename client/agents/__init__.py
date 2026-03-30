"""
A2A Agent System

Agent-to-Agent communication framework for MCP
Allows agents to coordinate via message passing
"""

from .base_agent import BaseAgent, AgentMessage, MessageType
from .orchestrator import OrchestratorAgent
from .researcher import ResearcherAgent
from .plex_ingester import PlexIngesterAgent
from .analyst import AnalystAgent
from .planner import PlannerAgent
from .writer import WriterAgent

__all__ = [
    'BaseAgent',
    'AgentMessage',
    'MessageType',
    'OrchestratorAgent',
    'ResearcherAgent',
    'PlexIngesterAgent',
    'AnalystAgent',
    'PlannerAgent',
    'WriterAgent'
]