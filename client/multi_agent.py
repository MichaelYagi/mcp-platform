"""
Multi-Agent Execution System (Updated for LangChain 1.2.0)
Uses LangChain create_agent for tool execution
NOW WITH COMPREHENSIVE STOP SIGNAL HANDLING
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
from .stop_signal import is_stop_requested, clear_stop, get_stop_status
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain.agents import create_agent
from .agents.base_agent import AgentMessage, MessageType
from .agents.orchestrator import OrchestratorAgent
from .agents.researcher import ResearcherAgent
from .agents.plex_ingester import PlexIngesterAgent
from .agents.analyst import AnalystAgent
from .agents.planner import PlannerAgent
from .agents.writer import WriterAgent
from .message_router import MessageRouter, MessageProtocol, MessagePriority, RoutingStrategy
from .negotiation_engine import NegotiationEngine
from .health_monitor import HealthMonitor
from .performance_metrics import PerformanceMetrics

class AgentRole(Enum):
    """Defines different agent specializations"""
    ORCHESTRATOR = "orchestrator"
    RESEARCHER = "researcher"
    CODER = "coder"
    ANALYST = "analyst"
    WRITER = "writer"
    PLANNER = "planner"
    PLEX_INGESTER = "plex_ingester"


@dataclass
class AgentTask:
    """Represents a task for an agent"""
    task_id: str
    role: AgentRole
    description: str
    context: Dict[str, Any]
    dependencies: List[str] = None
    result: Optional[Any] = None
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class MultiAgentOrchestrator:
    """
    Orchestrates multiple specialized agents with PROPER tool execution
    Updated for LangChain 1.2.0 WITH STOP SIGNAL HANDLING
    """

    def __init__(self, base_llm, tools, logger: logging.Logger):
        self.base_llm = base_llm
        self.a2a_enabled = False
        self.a2a_agents: Dict[str, Any] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()


        # Handle tools as either list or dict
        if isinstance(tools, dict):
            self.tools = list(tools.values())
        else:
            self.tools = tools

        self.logger = logger

        # Create specialized agent executors
        self.agent_executors = self._create_agent_executors()

        # Task management
        self.tasks: Dict[str, AgentTask] = {}
        self.task_results: Dict[str, Any] = {}

        # A2A
        self.message_router = MessageRouter(logger)
        self.negotiation_engine = NegotiationEngine(logger)
        self.health_monitor = HealthMonitor(logger)
        self.performance_metrics = PerformanceMetrics(logger)

    def update_llm(self, new_llm):
        """
        Update the LLM and recreate all agent executors
        Call this when the model is switched
        """
        self.logger.info(f"🔄 Updating multi-agent LLM and recreating agents")

        # Update base LLM
        self.base_llm = new_llm

        # Recreate all agent executors with new LLM
        self.agent_executors = self._create_agent_executors()

        # Log which model we're using
        if hasattr(new_llm, 'model'):
            model_name = new_llm.model
        elif hasattr(new_llm, 'model_name'):
            model_name = new_llm.model_name
        elif hasattr(new_llm, 'model_path'):
            from pathlib import Path
            model_name = Path(new_llm.model_path).stem
        else:
            model_name = "unknown"

        self.logger.info(f"✅ Multi-agent agents recreated with: {model_name}")

    def _create_agent_executors(self) -> Dict[AgentRole, Dict]:
        """Create agent executors with proper tool calling"""

        executors = {}

        # System prompts for each role
        system_prompts = {
            AgentRole.ORCHESTRATOR: """You are an Orchestrator Agent coordinating multiple specialized agents.
When given a task, create a detailed execution plan with subtasks.
Respond ONLY with JSON in this format:
{{
  "subtasks": [
    {{
      "id": "task_1",
      "role": "researcher",
      "description": "Detailed task description",
      "dependencies": []
    }}
  ]
}}

Available roles: researcher, coder, analyst, writer, planner, plex_ingester

If this is a simple task that doesn't need multiple agents, respond with:
{{"subtasks": []}}""",

            AgentRole.RESEARCHER: """You are a Researcher Agent focused on gathering accurate information.
            ALWAYS use your available tools to search for information.
            Never make up information - use tools to find real data.

            CRITICAL: Call tools ONE AT A TIME. Wait for each result before calling the next.
            NEVER call multiple tools simultaneously.
            NEVER call search_entries, rag_search_tool, or knowledge base tools for GitHub tasks.

            GITHUB REVIEW WORKFLOW — follow this exact sequence:
            1. github_clone_repo(url) → get local_path
            2. analyze_project(project_path=local_path) → get tech stack
            3. github_list_files(local_path=local_path, extensions=["py"]) → get file list
            4. review_code(path=local_path+"/key_file.py") → review important files
            5. github_cleanup_repo(local_path=local_path) → cleanup
            """,

            AgentRole.CODER: """You are a Coder Agent focused on writing quality code.
Use tools when you need to look up code examples or documentation.""",

            AgentRole.ANALYST: """You are an Analyst Agent focused on data analysis and insights.
Use tools to gather data before analyzing.""",

            AgentRole.WRITER: """You are a Writer Agent focused on clear communication.
Use tools to gather information before writing.""",

            AgentRole.PLANNER: """You are a Planner Agent focused on organizing tasks.
Use todo tools to manage and create tasks.""",

            AgentRole.PLEX_INGESTER: """You are a Plex Ingester Agent.
FOR SIMPLE INGESTION (e.g., "Ingest 2 items"):
- Use: plex_ingest_batch(limit=2) - Does everything in one call ✅
FOR COMPLEX WORKFLOWS (e.g., "Find items, then..."):
- Step 1: plex_find_unprocessed(limit=N)
- Step 2: Wait for results
- Step 3: Use real IDs from step 1 in plex_ingest_items
CRITICAL: Never make up item IDs! Only use IDs returned by plex_find_unprocessed."""
        }

        for role in AgentRole:
            # Get tools for this role
            role_tools = self._get_tools_for_role(role)

            if not role_tools and role != AgentRole.ORCHESTRATOR:
                # No tools, store None
                executors[role] = None
                self.logger.info(f"✅ Created {role.value} agent (no tools)")
                continue

            # Create agent with LangChain 1.2.0 API
            try:
                agent = create_agent(
                    self.base_llm,
                    role_tools
                )

                # Store both agent and system prompt
                executors[role] = {
                    "agent": agent,
                    "system_prompt": system_prompts.get(role, "You are a helpful AI assistant."),
                    "tools": role_tools
                }

                self.logger.info(f"✅ Created {role.value} agent with {len(role_tools)} tools")

            except Exception as e:
                self.logger.error(f"❌ Failed to create {role.value} agent: {e}")
                executors[role] = None

        return executors

    def _get_tools_for_role(self, role: AgentRole) -> List:
        """Get appropriate tools for each agent role"""

        role_tools = {
            AgentRole.ORCHESTRATOR: [],

            AgentRole.RESEARCHER: [
                "rag_search_tool",
                "semantic_media_search_text",
                "get_weather_tool",
                "github_clone_repo",
                "github_list_files",
                "github_get_file_content",
                "github_cleanup_repo",
                # Code review tools — needed for GitHub review workflow
                "analyze_project",
                "analyze_code_file",
                "review_code",
                "scan_project_structure",
            ],

            AgentRole.CODER: [
                "rag_search_tool",
                # Code analysis tools
                "analyze_project",
                "analyze_code_file",
                "review_code",
                "scan_project_structure",
                "get_project_dependencies",
                # GitHub file access
                "github_get_file_content"
            ],

            AgentRole.ANALYST: [
                "rag_search_tool", "search_entries",
            ],

            AgentRole.WRITER: [
                "rag_search_tool", "search_entries",
            ],

            AgentRole.PLANNER: [
                "list_todo_items", "add_todo_item",
            ],

            # Plex Ingester with granular tools
            AgentRole.PLEX_INGESTER: [
                # Granular tools for multi-agent orchestration
                "plex_find_unprocessed",  # STEP 1: Find items
                "plex_ingest_items",  # STEP 2: Batch parallel (recommended)
                "plex_ingest_single",  # STEP 3: Single item (max parallelization)
                "plex_get_stats",  # STEP 4: Get statistics

                # Original all-in-one (for simple queries)
                "plex_ingest_batch",  # Original combined tool

                # Supporting tools
                "rag_search_tool",  # Check what's already ingested
            ],
        }

        tool_names = role_tools.get(role, [])

        # Filter available tools
        available_tools = []
        for tool in self.tools:
            if hasattr(tool, 'name') and tool.name in tool_names:
                available_tools.append(tool)

        return available_tools

    def enable_a2a(self):
        """Enable Agent-to-Agent communication with advanced features"""
        if self.a2a_enabled:
            return

        self.logger.info("🔗 Initializing advanced A2A system...")

        # Start health monitoring
        asyncio.create_task(self.health_monitor.start_monitoring(check_interval=5.0))
        self.logger.info("💚 Health monitoring started")

        # Message bus callback with routing
        async def message_bus(message: AgentMessage):
            """Route messages through the advanced router"""
            from client.message_router import MessageEnvelope, MessagePriority

            # Determine priority from message metadata
            priority_map = {
                "critical": MessagePriority.CRITICAL,
                "high": MessagePriority.HIGH,
                "normal": MessagePriority.NORMAL,
                "low": MessagePriority.LOW,
                "bulk": MessagePriority.BULK
            }
            priority = priority_map.get(
                message.metadata.get("priority", "normal"),
                MessagePriority.NORMAL
            )

            # Determine routing strategy
            if message.to_agent:
                strategy = RoutingStrategy.DIRECT
            elif message.metadata.get("broadcast"):
                strategy = RoutingStrategy.BROADCAST
            elif message.metadata.get("load_balanced"):
                strategy = RoutingStrategy.LOAD_BALANCED
            else:
                strategy = RoutingStrategy.DIRECT

            envelope = MessageEnvelope(
                message_id=f"msg_{int(time.time() * 1000)}_{id(message)}",
                from_agent=message.from_agent,
                to_agent=message.to_agent,
                content=message.content,
                priority=priority,
                routing_strategy=strategy,
                timestamp=message.timestamp or time.time(),
                metadata=message.metadata
            )

            await self.message_router.route_message(envelope)

        # Create specialized A2A agents
        tools_map = {
            "researcher": self._get_tools_for_role(AgentRole.RESEARCHER),
            "analyst": self._get_tools_for_role(AgentRole.ANALYST),
            "planner": self._get_tools_for_role(AgentRole.PLANNER),
            "plex_ingester": self._get_tools_for_role(AgentRole.PLEX_INGESTER),
            "writer": self._get_tools_for_role(AgentRole.WRITER),
        }

        # Orchestrator (no tools needed)
        self.a2a_agents["orchestrator"] = OrchestratorAgent(
            agent_id="orchestrator_1",
            llm=self.base_llm,
            logger=self.logger,
            message_bus=message_bus
        )

        # Register with systems
        self.message_router.register_agent("orchestrator_1", self.a2a_agents["orchestrator"])
        self.health_monitor.register_agent("orchestrator_1")

        # Specialized agents
        self.a2a_agents["researcher"] = ResearcherAgent(
            agent_id="researcher_1",
            llm=self.base_llm,
            tools=tools_map["researcher"],
            logger=self.logger,
            message_bus=message_bus
        )
        self.message_router.register_agent("researcher_1", self.a2a_agents["researcher"])
        self.health_monitor.register_agent("researcher_1")

        self.a2a_agents["analyst"] = AnalystAgent(
            agent_id="analyst_1",
            llm=self.base_llm,
            tools=tools_map["analyst"],
            logger=self.logger,
            message_bus=message_bus
        )
        self.message_router.register_agent("analyst_1", self.a2a_agents["analyst"])
        self.health_monitor.register_agent("analyst_1")

        self.a2a_agents["planner"] = PlannerAgent(
            agent_id="planner_1",
            llm=self.base_llm,
            tools=tools_map["planner"],
            logger=self.logger,
            message_bus=message_bus
        )
        self.message_router.register_agent("planner_1", self.a2a_agents["planner"])
        self.health_monitor.register_agent("planner_1")

        self.a2a_agents["writer"] = WriterAgent(
            agent_id="writer_1",
            llm=self.base_llm,
            tools=tools_map["writer"],
            logger=self.logger,
            message_bus=message_bus
        )
        self.message_router.register_agent("writer_1", self.a2a_agents["writer"])
        self.health_monitor.register_agent("writer_1")

        self.a2a_agents["plex_ingester"] = PlexIngesterAgent(
            agent_id="plex_ingester_1",
            llm=self.base_llm,
            tools=tools_map["plex_ingester"],
            logger=self.logger,
            message_bus=message_bus
        )
        self.message_router.register_agent("plex_ingester_1", self.a2a_agents["plex_ingester"])
        self.health_monitor.register_agent("plex_ingester_1")

        self.a2a_enabled = True
        self.logger.info(f"✅ A2A system initialized with {len(self.a2a_agents)} agents")
        self.logger.info("✨ Advanced features: routing, health monitoring, metrics, negotiation")

    def disable_a2a(self):
        """Disable A2A system and cleanup"""
        self.a2a_enabled = False

        # Stop health monitoring
        if self.health_monitor:
            asyncio.create_task(self.health_monitor.stop_monitoring())

        # Cleanup agents
        self.a2a_agents.clear()

        self.logger.info("🔗 A2A system disabled")

    async def execute_a2a(self, user_request: str) -> str:
        """
        Execute using A2A system with advanced features
        """
        if not self.a2a_enabled:
            self.enable_a2a()

        self.logger.info(f"🔗 A2A execution started: {user_request}")
        start_time = time.time()

        try:
            # Step 1: Orchestrator creates plan
            orchestrator = self.a2a_agents["orchestrator"]
            plan = await orchestrator.create_plan(user_request)

            # Check stop
            if is_stop_requested():
                return "A2A execution stopped during planning"

            subtasks = plan.get("subtasks", [])

            if not subtasks:
                # Simple task - use single agent
                self.logger.info("📌 Simple task - using single A2A agent")
                result = await self._a2a_single_agent(user_request)

                # Record metrics for simple task
                from client.performance_metrics import TaskMetrics
                task_metrics = TaskMetrics(
                    task_id=f"simple_{int(start_time)}",
                    agent_id="single_agent",
                    task_type="simple",
                    start_time=start_time,
                    end_time=time.time(),
                    duration=time.time() - start_time,
                    success=True,
                    tools_used=[],
                    llm_calls=1,
                    tokens_used=0
                )
                self.performance_metrics.record_task(task_metrics)

                return result

            # Step 2: Execute subtasks with tracking
            self.logger.info(f"🎭 Executing {len(subtasks)} subtasks via A2A")
            results = {}

            for subtask in subtasks:
                # Check stop before each subtask
                if is_stop_requested():
                    self.logger.warning("🛑 A2A execution stopped")
                    break

                task_id = subtask["id"]
                agent_role = subtask["agent"]
                description = subtask["description"]
                depends_on = subtask.get("depends_on", [])

                # Build context from dependencies
                context = {"user_request": user_request}
                for dep_id in depends_on:
                    if dep_id in results:
                        context[f"result_{dep_id}"] = results[dep_id]

                agent = self.a2a_agents.get(agent_role)
                if not agent:
                    self.logger.warning(f"⚠️ Agent {agent_role} not found")
                    results[task_id] = f"Agent {agent_role} not available"
                    continue

                # Execute task with tracking
                self.logger.info(f"▶️  Executing {task_id} with {agent_role}")
                task_start = time.time()

                try:
                    result = await agent.execute_task(description, context)
                    results[task_id] = result
                    success = True
                    error = None
                    self.logger.info(f"✅ {task_id} completed")

                except Exception as e:
                    results[task_id] = f"Error: {str(e)}"
                    success = False
                    error = str(e)
                    self.logger.error(f"❌ {task_id} failed: {e}")

                    # Record error with health monitor
                    self.health_monitor.record_error(agent.agent_id, error)

                task_end = time.time()
                task_duration = task_end - task_start

                # Record performance metrics
                from client.performance_metrics import TaskMetrics
                task_metrics = TaskMetrics(
                    task_id=task_id,
                    agent_id=agent.agent_id,
                    task_type=agent_role,
                    start_time=task_start,
                    end_time=task_end,
                    duration=task_duration,
                    success=success,
                    tools_used=list(agent.tools.keys()) if hasattr(agent, 'tools') else [],
                    llm_calls=1,
                    tokens_used=0,
                    error=error
                )
                self.performance_metrics.record_task(task_metrics)

                # Record with health monitor
                self.health_monitor.record_task_completion(
                    agent.agent_id,
                    task_duration,
                    success
                )

                # Update resource usage
                queue_size = len(agent.message_history) if hasattr(agent, 'message_history') else 0
                self.health_monitor.update_resource_usage(
                    agent.agent_id,
                    queue_size=queue_size
                )

            # Step 3: Aggregate results
            if is_stop_requested():
                return f"A2A execution stopped. Partial results:\n{self._format_results(results)}"

            final_result = await self._aggregate_results(user_request, results)

            duration = time.time() - start_time
            self.logger.info(f"✅ A2A execution completed in {duration:.2f}s")

            return final_result

        except Exception as e:
            self.logger.error(f"❌ A2A execution failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _a2a_single_agent(self, user_request: str) -> str:
        """Execute simple task with single A2A agent"""
        request_lower = user_request.lower()

        if "plex" in request_lower or "ingest" in request_lower:
            agent = self.a2a_agents["plex_ingester"]
        elif "analyze" in request_lower:
            agent = self.a2a_agents["analyst"]
        elif "plan" in request_lower or "todo" in request_lower:
            agent = self.a2a_agents["planner"]
        elif "write" in request_lower or "summary" in request_lower:
            agent = self.a2a_agents["writer"]
        else:
            agent = self.a2a_agents["researcher"]

        self.logger.info(f"📌 Using {agent.role} agent")

        # Execute with tracking
        task_start = time.time()
        try:
            result = await agent.execute_task(user_request)
            success = True
            error = None
        except Exception as e:
            result = f"Error: {str(e)}"
            success = False
            error = str(e)
            self.health_monitor.record_error(agent.agent_id, error)

        task_duration = time.time() - task_start

        # Record metrics
        from client.performance_metrics import TaskMetrics
        task_metrics = TaskMetrics(
            task_id=f"single_{int(task_start)}",
            agent_id=agent.agent_id,
            task_type=agent.role,
            start_time=task_start,
            end_time=time.time(),
            duration=task_duration,
            success=success,
            tools_used=list(agent.tools.keys()) if hasattr(agent, 'tools') else [],
            llm_calls=1,
            tokens_used=0,
            error=error
        )
        self.performance_metrics.record_task(task_metrics)

        self.health_monitor.record_task_completion(agent.agent_id, task_duration, success)

        return result

    def _format_results(self, results: Dict[str, Any]) -> str:
        """Format partial results"""
        output = []
        for task_id, result in results.items():
            result_str = str(result)[:100]
            output.append(f"{task_id}: {result_str}...")
        return "\n".join(output)

    def get_a2a_status(self) -> Dict[str, Any]:
        """Get comprehensive A2A system status"""
        if not self.a2a_enabled:
            return {"enabled": False}

        agent_statuses = {}
        for name, agent in self.a2a_agents.items():
            status = agent.get_status()

            # Add health info
            health = self.health_monitor.get_agent_health(agent.agent_id)
            if health:
                status["health"] = {
                    "status": health.status.value,
                    "error_count": health.error_count,
                    "avg_response_time": health.avg_response_time
                }

            # Add performance info
            perf = self.performance_metrics.get_agent_performance(agent.agent_id)
            if perf:
                status["performance"] = {
                    "total_tasks": perf.total_tasks,
                    "success_rate": perf.successful_tasks / perf.total_tasks if perf.total_tasks > 0 else 0,
                    "avg_duration": perf.avg_duration
                }

            agent_statuses[name] = status

        status = {
            "enabled": True,
            "agents": agent_statuses,
            "message_queue_size": self.message_queue.qsize()
        }

        # Add advanced system stats
        status["routing"] = self.message_router.get_routing_stats()
        status["health_summary"] = self.health_monitor.get_health_summary()
        status["performance_summary"] = self.performance_metrics.get_comparative_stats()
        status["negotiation_stats"] = self.negotiation_engine.get_statistics()

        return status

    async def execute(self, user_request: str, skill_context: str = None, session_id: int = None, rag_search_fn=None) -> dict:
        """
        Main entry point for multi-agent execution
        NOW WITH STOP SIGNAL HANDLING
        NOW RETURNS DICT with current_model
        """

        self.logger.info(f"🎭 Multi-agent execution started: {user_request}")
        start_time = time.time()

        # Get current model name from base_llm
        if hasattr(self.base_llm, 'model'):
            current_model = self.base_llm.model
        elif hasattr(self.base_llm, 'model_name'):
            current_model = self.base_llm.model_name
        elif hasattr(self.base_llm, 'model_path'):
            from pathlib import Path
            current_model = Path(self.base_llm.model_path).stem
        else:
            current_model = "MCP Error"

        try:
            # Step 1: Create execution plan
            plan = await self._create_execution_plan(user_request, skill_context=skill_context)

            # Check stop after planning
            if is_stop_requested():
                self.logger.warning("🛑 Stop requested after creating plan - aborting execution")
                return {
                    "response": "Execution stopped by user before tasks could begin.",
                    "current_model": current_model,
                    "multi_agent": True,
                    "stopped": True
                }

            if not plan:
                self.logger.info("📊 Simple query detected, falling back to single agent")
                fallback_response = await self._fallback_single_agent(user_request)
                return {
                    "response": fallback_response,
                    "current_model": current_model,
                    "multi_agent": True
                }

            # Step 2: Execute tasks
            results = await self._execute_tasks(plan, skill_context=skill_context)

            # Check if execution was stopped
            if results.get("_stopped", False):
                stopped_message = results.get("_stopped_message", "Stopped by user")
                self.logger.warning(f"🛑 Multi-agent execution stopped: {stopped_message}")
                return {
                    "response": f"🛑 **Execution stopped:** {stopped_message}",
                    "current_model": current_model,
                    "multi_agent": True,
                    "stopped": True
                }

            # Step 3: Aggregate results
            final_response = await self._aggregate_results(user_request, results, session_id=session_id, rag_search_fn=rag_search_fn)
            duration = time.time() - start_time
            self.logger.info(f"✅ Multi-agent execution completed in {duration:.2f}s")

            return {
                "response": final_response,
                "current_model": current_model,
                "multi_agent": True
            }

        except Exception as e:
            self.logger.error(f"❌ Multi-agent execution failed: {e}, falling back to single agent")
            import traceback
            traceback.print_exc()
            fallback_response = await self._fallback_single_agent(user_request)
            return {
                "response": fallback_response,
                "current_model": current_model,
                "multi_agent": True
            }

    async def _create_execution_plan(self, user_request: str, skill_context: str = None) -> Optional[List[AgentTask]]:
        """Use orchestrator to create execution plan"""

        skill_section = ""
        if skill_context:
            skill_section = f"""
        IMPORTANT: A relevant skill has been provided. Follow its workflow exactly when creating the plan.

        {skill_context}

        """

        self.logger.info("📋 Creating execution plan...")

        # Check stop before planning
        if is_stop_requested():
            self.logger.warning("🛑 Stop requested - skipping plan creation")
            return None

        # Orchestrator has no tools, use base LLM
        planning_prompt = f"""{skill_section}Given this user request: "{user_request}"

        Create an execution plan by breaking it into subtasks.

        Role selection guide:
        - researcher: Gather information, search web, GitHub clone+list+analyze
        - coder: Write code, review code files, fix bugs  
        - analyst: Analyze data, compare results
        - writer: Write reports, summaries
        - planner: Manage todos and tasks
        - plex_ingester: Ingest Plex media

        For GitHub repository REVIEW requests, use this plan:
        {{
          "subtasks": [
            {{"id": "task_1", "role": "researcher", "description": "Clone the repo, analyze project structure, list Python files, review key source files using review_code, then cleanup", "dependencies": []}}
          ]
        }}

        Respond ONLY with JSON...
        """

        try:
            response = await self.base_llm.ainvoke([
                SystemMessage(content="You are an Orchestrator Agent coordinating multiple specialized agents."),
                HumanMessage(content=planning_prompt)
            ])

            # Parse JSON response
            import json
            import re

            content = response.content.strip()

            # Extract JSON if wrapped in markdown
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            # Check if content is empty
            if not content:
                self.logger.warning("⚠️ Orchestrator returned empty response, treating as simple task")
                return None

            # Try parsing with better error handling
            try:
                plan_data = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.warning(f"⚠️ Orchestrator returned invalid JSON: {e}")
                self.logger.debug(f"   Content: {content[:200]}")
                self.logger.info("   Falling back to single-agent mode")
                return None

            subtasks = plan_data.get("subtasks", [])

            if not subtasks:
                self.logger.info("📋 Simple task, using single agent")
                return None

            # Convert to AgentTask objects
            tasks = []
            for i, subtask in enumerate(subtasks):
                role_str = subtask.get("role", "researcher")
                try:
                    role = AgentRole(role_str)
                except ValueError:
                    self.logger.warning(f"⚠️  Unknown role '{role_str}', defaulting to researcher")
                    role = AgentRole.RESEARCHER

                task = AgentTask(
                    task_id=subtask.get("id", f"task_{i}"),
                    role=role,
                    description=subtask.get("description", ""),
                    context={"user_request": user_request},
                    dependencies=subtask.get("dependencies", [])
                )
                tasks.append(task)
                self.tasks[task.task_id] = task

            self.logger.info(f"📋 Created plan with {len(tasks)} subtasks")
            for task in tasks:
                self.logger.info(f"  - {task.task_id}: {task.role.value} - {task.description[:50]}...")

            return tasks

        except Exception as e:
            self.logger.error(f"❌ Failed to create plan: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _execute_tasks(self, tasks: List[AgentTask], skill_context: str = None) -> Dict[str, Any]:
        """
        Execute tasks respecting dependencies
        WITH COMPREHENSIVE STOP SIGNAL CHECKING
        """

        self.logger.info(f"⚙️ Executing {len(tasks)} tasks...")

        completed = set()
        results = {}

        while len(completed) < len(tasks):
            # ═══════════════════════════════════════════════════════════
            # CHECK STOP SIGNAL BEFORE EACH BATCH OF TASKS
            # ═══════════════════════════════════════════════════════════
            if is_stop_requested():
                self.logger.warning(f"🛑 Multi-agent execution stopped after {len(completed)}/{len(tasks)} tasks")
                results["_stopped"] = True
                results["_stopped_message"] = f"Stopped after completing {len(completed)} of {len(tasks)} tasks"
                break

            ready_tasks = [
                task for task in tasks
                if task.task_id not in completed
                and all(dep in completed for dep in task.dependencies)
            ]

            if not ready_tasks:
                self.logger.error("❌ Dependency deadlock detected")
                break

            self.logger.info(f"⚙️ Executing {len(ready_tasks)} parallel tasks...")

            task_coroutines = [
                self._execute_single_task(task, results, skill_context=skill_context)
                for task in ready_tasks
            ]

            task_results = await asyncio.gather(*task_coroutines, return_exceptions=True)

            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    self.logger.error(f"❌ Task {task.task_id} failed: {result}")
                    results[task.task_id] = f"Error: {str(result)}"
                else:
                    results[task.task_id] = result
                    self.logger.info(f"✅ Task {task.task_id} completed")

                completed.add(task.task_id)

            # ═══════════════════════════════════════════════════════════
            # CHECK STOP SIGNAL AFTER COMPLETING BATCH
            # ═══════════════════════════════════════════════════════════
            if is_stop_requested():
                self.logger.warning(f"🛑 Stop detected after batch completion")
                results["_stopped"] = True
                results["_stopped_message"] = f"Stopped after completing {len(completed)} of {len(tasks)} tasks"
                break

        return results

    async def _execute_single_task(self, task: AgentTask, previous_results: Dict, skill_context: str = None) -> str:
        """
        Execute a single agent task WITH TOOL EXECUTION
        NOW WITH STOP SIGNAL CHECK BEFORE EXECUTION
        """

        # ═══════════════════════════════════════════════════════════
        # CHECK STOP BEFORE STARTING TASK
        # ═══════════════════════════════════════════════════════════
        if is_stop_requested():
            self.logger.warning(f"🛑 Task {task.task_id} ({task.role.value}) stopped before execution")
            task.status = "stopped"
            return f"Task stopped before execution"

        task.status = "running"
        task.start_time = time.time()

        self.logger.info(f"🤖 {task.role.value} executing: {task.description[:50]}...")

        try:
            agent_info = self.agent_executors.get(task.role)

            # Build context from previous results
            context_info = ""
            for dep_id in task.dependencies:
                if dep_id in previous_results:
                    context_info += f"\n\nOUTPUT FROM {dep_id} (use this data directly):\n{previous_results[dep_id]}\n"
                    context_info += f"IMPORTANT: Extract any IDs, titles, or structured data from the above and use them in your tool calls. Do not invent new IDs.\n"

            # Create input for agent
            task_input = f"""Task: {task.description}

User's original request: {task.context.get('user_request', '')}
{context_info}

Complete this task using your available tools."""

            # Execute with tools
            if agent_info:
                agent = agent_info["agent"]
                system_prompt = agent_info["system_prompt"]

                self.logger.info(f"🔧 Running {task.role.value} with tool execution enabled...")

                # Build messages with system prompt
                skill_section = ""
                if skill_context:
                    skill_section = f"""
                RELEVANT SKILL - FOLLOW THIS WORKFLOW EXACTLY:
                {skill_context}

                CRITICAL: Use ONLY the tools listed in the skill workflow above.
                Do NOT call search_entries, rag_search_tool, or any knowledge base tools.
                Call tools ONE AT A TIME. Wait for each result before calling the next.
                """

                messages = [
                    SystemMessage(content=system_prompt + skill_section),
                    HumanMessage(content=task_input)
                ]

                # Invoke agent with messages
                result = await agent.ainvoke({"messages": messages})

                # ═══════════════════════════════════════════════════════════
                # CHECK STOP AFTER AGENT EXECUTION
                # ═══════════════════════════════════════════════════════════
                if is_stop_requested():
                    self.logger.warning(f"🛑 Task {task.task_id} stopped after agent execution")
                    task.status = "stopped"
                    task.end_time = time.time()
                    return f"Task stopped after execution"

                # Extract output from last message
                last_message = result["messages"][-1]
                output = last_message.content if hasattr(last_message, 'content') else str(last_message)

            else:
                # No tools, use base LLM
                self.logger.info(f"💬 Running {task.role.value} without tools...")
                response = await self.base_llm.ainvoke([
                    SystemMessage(content=f"You are a {task.role.value} agent."),
                    HumanMessage(content=task_input)
                ])
                output = response.content

            task.result = output
            task.status = "completed"
            task.end_time = time.time()

            duration = task.end_time - task.start_time
            self.logger.info(f"✅ {task.role.value} completed in {duration:.2f}s")

            return output

        except Exception as e:
            task.status = "failed"
            task.end_time = time.time()
            self.logger.error(f"❌ Task {task.task_id} failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    async def _aggregate_results(self, user_request: str, results: Dict[str, Any], session_id: int = None, rag_search_fn=None) -> str:
        """Aggregate results from all agents"""

        self.logger.info("📊 Aggregating results...")

        # Check if any results indicate stop
        if results.get("_stopped", False):
            return results.get("_stopped_message", "Execution stopped")

        # Check stop before aggregation
        if is_stop_requested():
            self.logger.warning("🛑 Stop requested - skipping result aggregation")
            return "Result aggregation stopped by user."

        results_summary = ""
        for task_id, result in results.items():
            # Skip metadata keys
            if task_id.startswith("_"):
                continue

            task = self.tasks.get(task_id)
            if task:
                results_summary += f"\n\n### {task.role.value.title()} ({task_id}):\n{result}"

        rag_context = ""
        if session_id and rag_search_fn:
            try:
                relevant = rag_search_fn(session_id, user_request, top_k=3, min_score=0.4)
                if relevant:
                    rag_context = "\n\nRELEVANT CONVERSATION HISTORY:\n"
                    for turn in relevant:
                        rag_context += f"{turn.get('text', '')[:600]}\n\n"
            except Exception as e:
                self.logger.warning(f"⚠️ RAG lookup failed in aggregator: {e}")

        aggregation_prompt = f"""User's original request: "{user_request}"
        {rag_context}
        Results from specialized agents:
        {results_summary}

        Synthesize these results into a coherent, final response that directly answers the user's request.
        If conversation history is provided above, use it to answer questions about prior exchanges.
        Focus on clarity and completeness."""

        response = await self.base_llm.ainvoke([
            SystemMessage(content="You are synthesizing results from multiple agents. Create a clear, unified response."),
            HumanMessage(content=aggregation_prompt)
        ])

        return response.content

    async def _fallback_single_agent(self, user_request: str) -> str:
        """Fallback to single agent with tool execution"""

        self.logger.info("🔄 Using single-agent fallback mode")

        # Check stop before fallback
        if is_stop_requested():
            self.logger.warning("🛑 Stop requested - skipping single-agent fallback")
            return "Single-agent execution stopped by user."

        # Choose best agent based on keywords
        request_lower = user_request.lower()

        if "plex" in request_lower or "ingest" in request_lower or "subtitle" in request_lower:
            agent_role = AgentRole.PLEX_INGESTER
        elif "code" in request_lower:
            agent_role = AgentRole.CODER
        elif "analyze" in request_lower:
            agent_role = AgentRole.ANALYST
        elif "write" in request_lower:
            agent_role = AgentRole.WRITER
        elif "plan" in request_lower or "todo" in request_lower:
            agent_role = AgentRole.PLANNER
        else:
            agent_role = AgentRole.RESEARCHER

        self.logger.info(f"📌 Selected {agent_role.value} agent for single-agent execution")

        agent_info = self.agent_executors.get(agent_role)

        if agent_info:
            agent = agent_info["agent"]
            system_prompt = agent_info["system_prompt"]

            self.logger.info(f"🔧 Running {agent_role.value} with tool execution enabled...")

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_request)
            ]

            result = await agent.ainvoke({"messages": messages})

            # Check stop after execution
            if is_stop_requested():
                self.logger.warning("🛑 Single-agent execution stopped")
                return "Single-agent execution stopped by user."

            last_message = result["messages"][-1]
            return last_message.content if hasattr(last_message, 'content') else str(last_message)
        else:
            # No tools, use base LLM
            response = await self.base_llm.ainvoke([
                SystemMessage(content=f"You are a {agent_role.value} agent."),
                HumanMessage(content=user_request)
            ])
            return response.content


async def should_use_multi_agent(user_request: str) -> bool:
    """Determine if a request should use multi-agent execution"""

    import logging
    import re
    from pathlib import Path

    logger = logging.getLogger("mcp_client")

    request_lower = user_request.lower()
    logger.info(f"🔍 Checking multi-agent for: {request_lower[:100]}")

    # ═══════════════════════════════════════════════════════════════
    # Check model size - Disable for tiny models
    # This check must come BEFORE any trigger logic
    # ═══════════════════════════════════════════════════════════════
    try:
        # Read current model from last_model.txt
        last_model_file = Path(__file__).parent / "last_model.txt"

        if last_model_file.exists():
            current_model = last_model_file.read_text().strip().lower()

            # Extract model size using pattern matching
            # Pattern: look for number followed by 'b' (billions) after colon or dash
            # Examples: qwen2.5:32b-instruct → 32, llama3.1:8b → 8
            size_pattern = r':(\d+)b(?:-|instruct|$|\s)'
            match = re.search(size_pattern, current_model)

            if match:
                size_b = int(match.group(1))
                logger.debug(f"Detected model size: {size_b}B parameters")

                if size_b < 3:
                    logger.warning(f"⚠️ Multi-agent DISABLED: Model too small ({size_b}B < 3B params)")
                    logger.info(f"   💡 Use qwen2.5:7b or larger for multi-agent")
                    return False
            else:
                # Fallback: Check for tiny model names if pattern matching fails
                tiny_models = [
                    "tinyllama",
                    "tiny-llama",
                    "phi-1",
                    "phi-2",
                    "gemma-2b",
                    "stablelm-2b",
                    "qwen2-1.5b",
                    "llama-1b",
                ]

                for tiny in tiny_models:
                    if tiny in current_model:
                        logger.warning(f"⚠️ Multi-agent DISABLED: {current_model} is too small/slow")
                        logger.info(f"   💡 Use qwen2.5:7b or larger for multi-agent")
                        return False

    except Exception as e:
        # If model check fails, continue with multi-agent decision
        logger.debug(f"Could not check model size: {e}")

    # ═══════════════════════════════════════════════════════════════
    # Multi-agent triggers
    # ═══════════════════════════════════════════════════════════════

    # Repository review is multi-step: clone → analyze → review → cleanup
    if re.search(r'github\.com/[^\s]+', user_request):
        logger.info(f"✅ Multi-agent triggered by: GitHub URL")
        return True

    multi_step_indicators = [
        " and then ", " then ", " after that ", " next ",
        "first.*then", "research.*analyze", "find.*summarize",
        "gather.*create", "search.*write", "ingest.*and.*",
    ]

    for indicator in multi_step_indicators:
        if re.search(indicator, request_lower):
            logger.info(f"✅ Multi-agent triggered by: {indicator}")
            return True

    complex_keywords = [
        "comprehensive", "detailed analysis", "full report",
        "research and", "analyze and", "compare and"
    ]

    if any(keyword in request_lower for keyword in complex_keywords):
        logger.info(f"✅ Multi-agent triggered by keyword")
        return True

    if len(user_request.split()) > 30:
        logger.info(f"✅ Multi-agent triggered by length: {len(user_request.split())} words")
        return True

    logger.info(f"❌ Multi-agent NOT triggered")
    return False