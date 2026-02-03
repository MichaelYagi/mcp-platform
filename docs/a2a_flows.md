# AGENT-TO-AGENT (A2A) ARCHITECTURE DIAGRAM
For mcp_a2a repository

## HIGH-LEVEL ARCHITECTURE

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           MCP A2A SYSTEM                                │
    │                                                                         │
    │  ┌──────────────┐         ┌──────────────┐         ┌──────────────┐     │
    │  │   CLIENT     │◄───────►│   MESSAGE    │◄───────►│   REMOTE     │     │
    │  │   (Local)    │         │    ROUTER    │         │   A2A SERVER │     │
    │  │              │         │  (Optional)  │         │   (HTTP/WS)  │     │
    │  └──────────────┘         └──────────────┘         └──────────────┘     │
    │         │                                                   │           │
    │         │                                                   │           │
    │         ▼                                                   ▼           │
    │  ┌──────────────┐                                   ┌──────────────┐    │
    │  │  LOCAL MCP   │                                   │  REMOTE MCP  │    │
    │  │   SERVERS    │                                   │   SERVERS    │    │
    │  │              │                                   │              │    │
    │  │ • RAG        │                                   │ • Tools      │    │
    │  │ • Plex       │                                   │ • Services   │    │
    │  │ • Code       │                                   │ • APIs       │    │
    │  │ • Weather    │                                   │              │    │
    │  └──────────────┘                                   └──────────────┘    │
    └─────────────────────────────────────────────────────────────────────────┘

## DETAILED A2A FLOW
USER QUERY → CLIENT → A2A DECISION → EXECUTION

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. USER SUBMITS QUERY                                                   │
    └─────────────────────────────────────────────────────────────────────────┘
                                  ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. CLIENT.PY - run_agent_wrapper()                                      │
    │                                                                         │
    │    ┌─────────────────────────────────────────────────────────────┐      │
    │    │ Should use multi-agent? (should_use_multi_agent())          │      │
    │    │  • Length > 30 words?                                       │      │
    │    │  • Multi-step indicators? ("and then", "after that")        │      │
    │    │  • Complex keywords? ("comprehensive", "detailed")          │      │
    │    └─────────────────────────────────────────────────────────────┘      │
    │                              ▼                                          │
    │    ┌─────────────────────────────────────────────────────────────┐      │
    │    │ YES: Multi-agent capable                                    │      │
    │    │ Check A2A vs Standard Multi-agent                           │      │
    │    └─────────────────────────────────────────────────────────────┘      │
    │                    ▼                           ▼                        │
    │         ┌──────────────────┐      ┌──────────────────────┐              │
    │         │   A2A ENABLED?   │      │  STANDARD MULTI-AGENT│              │
    │         │                  │      │                      │              │
    │         │ A2A_STATE[       │      │ orchestrator.execute()│             │
    │         │   "enabled"]     │      │                      │              │
    │         └──────────────────┘      └──────────────────────┘              │
    └─────────────────────────────────────────────────────────────────────────┘
                        ▼
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3. A2A EXECUTION PATH                                                   │
    │                                                                         │
    │    orchestrator.execute_a2a(user_request)                               │
    │                              ▼                                          │
    │    ┌─────────────────────────────────────────────────────────────┐      │
    │    │ STEP 1: Orchestrator Creates Plan                           │      │
    │    │  • OrchestratorAgent.create_plan()                          │      │
    │    │  • Breaks request into subtasks                             │      │
    │    │  • Assigns agents: researcher, analyst, planner, etc.       │      │
    │    └─────────────────────────────────────────────────────────────┘      │
    │                              ▼                                          │
    │    ┌─────────────────────────────────────────────────────────────┐      │
    │    │ STEP 2: Execute Subtasks                                    │      │
    │    │  • Each agent executes via execute_task()                   │      │
    │    │  • Agents communicate via message_bus                       │      │
    │    │  • Messages routed through MessageRouter                    │      │
    │    │  • Health monitored by HealthMonitor                        │      │
    │    └─────────────────────────────────────────────────────────────┘      │
    │                              ▼                                          │
    │    ┌─────────────────────────────────────────────────────────────┐      │
    │    │ STEP 3: Aggregate Results                                   │      │
    │    │  • Combine all task results                                 │      │
    │    │  • Return unified response                                  │      │
    │    └─────────────────────────────────────────────────────────────┘      │
    └─────────────────────────────────────────────────────────────────────────┘

## A2A AGENT ARCHITECTURE

### A2A AGENT INTERNAL STRUCTURE

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                           BASE AGENT                                    │
    │                      (client/agents/base_agent.py)                      │
    │                                                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ PROPERTIES:                                                      │   │
    │  │  • agent_id: str                                                 │   │
    │  │  • role: str                                                     │   │
    │  │  • llm: ChatModel                                                │   │
    │  │  • tools: Dict[str, Tool]                                        │   │ 
    │  │  • message_bus: Callable                                         │   │
    │  │  • message_history: List[AgentMessage]                           │   │
    │  │  • state: Dict[str, Any]                                         │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ METHODS:                                                         │   │
    │  │  • execute_task(description, context) → str                      │   │
    │  │  • send_message(to_agent, content, metadata)                     │   │
    │  │  • receive_message(message)                                      │   │
    │  │  • get_status() → Dict                                           │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘
                                    ▼
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
    ┌──────────────────┐                          ┌──────────────────┐
    │ ORCHESTRATOR     │                          │ SPECIALIZED      │
    │ AGENT            │                          │ AGENTS           │
    │                  │                          │                  │
    │ • No tools       │                          │ • Researcher     │
    │ • Creates plans  │                          │ • Analyst        │
    │ • Coordinates    │                          │ • Planner        │
    └──────────────────┘                          │ • Writer         │
                                                  │ • PlexIngester   │
                                                  └──────────────────┘

## MESSAGE ROUTING SYSTEM

### MESSAGE FLOW ARCHITECTURE

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                        MESSAGE ROUTER                                   │
    │                   (client/message_router.py)                            │
    │                                                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ ROUTING STRATEGIES:                                              │   │
    │  │                                                                  │   │
    │  │  1. DIRECT      - Send to specific agent                         │   │
    │  │  2. BROADCAST   - Send to all agents                             │   │
    │  │  3. ROUND_ROBIN - Distribute load evenly                         │   │
    │  │  4. LOAD_BALANCED - Send to least busy agent                     │   │
    │  │  5. PRIORITY    - Route by message priority                      │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ MESSAGE PRIORITIES:                                              │   │
    │  │  • CRITICAL (0) - Immediate processing                           │   │
    │  │  • HIGH (1)     - Next in queue                                  │   │
    │  │  • NORMAL (2)   - Standard processing                            │   │
    │  │  • LOW (3)      - Background tasks                               │   │
    │  │  • BULK (4)     - Batch processing                               │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ FLOW:                                                            │   │
    │  │                                                                  │   │
    │  │  Agent A ──► MessageEnvelope ──► MessageRouter ──► Agent B       │   │
    │  │                     │                   │                        │   │
    │  │                     │                   ├──► Priority Queue      │   │
    │  │                     │                   ├──► Load Balancer       │   │
    │  │                     │                   └──► Statistics          │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

## SUPPORTING SYSTEMS
### SUPPORTING INFRASTRUCTURE

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 1. HEALTH MONITOR (client/health_monitor.py)                            │
    │                                                                         │
    │    ┌──────────────────────────────────────────────────────────┐         │
    │    │ • Tracks agent health (HEALTHY, DEGRADED, UNHEALTHY)     │         │
    │    │ • Monitors response times                                │         │
    │    │ • Counts errors and task completions                     │         │
    │    │ • Provides health summary                                │         │
    │    └──────────────────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 2. PERFORMANCE METRICS (client/performance_metrics.py)                  │
    │                                                                         │
    │    ┌──────────────────────────────────────────────────────────┐         │
    │    │ • Records task durations                                 │         │
    │    │ • Tracks success/failure rates                           │         │
    │    │ • Monitors LLM calls and token usage                     │         │
    │    │ • Provides comparative statistics                        │         │
    │    └──────────────────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ 3. NEGOTIATION ENGINE (client/negotiation_engine.py)                    │
    │                                                                         │
    │    ┌──────────────────────────────────────────────────────────┐         │
    │    │ • Handles agent-to-agent negotiations                    │         │
    │    │ • Manages resource allocation                            │         │
    │    │ • Resolves conflicts                                     │         │
    │    │ • Tracks negotiation statistics                          │         │
    │    └──────────────────────────────────────────────────────────┘         │
    └─────────────────────────────────────────────────────────────────────────┘

## A2A vs STANDARD MULTI-AGENT COMPARISON
### STANDARD MULTI-AGENT vs A2A

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ STANDARD MULTI-AGENT (orchestrator.execute())                           │
    │                                                                         │
    │  Orchestrator                                                           │
    │       │                                                                 │
    │       ├──► Create plan with base_llm                                    │
    │       │                                                                 │
    │       ├──► Execute task 1 ──► agent_executors[role]["agent"]            │
    │       ├──► Execute task 2 ──► agent_executors[role]["agent"]            │
    │       ├──► Execute task 3 ──► agent_executors[role]["agent"]            │
    │       │                                                                 │
    │       └──► Aggregate results with base_llm                              │
    │                                                                         │
    │  • Simple execution flow                                                │
    │  • No inter-agent communication                                         │
    │  • Sequential or parallel task execution                                │
    │  • Results returned to orchestrator                                     │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ A2A MULTI-AGENT (orchestrator.execute_a2a())                            │
    │                                                                         │
    │  OrchestratorAgent                                                      │
    │       │                                                                 │
    │       ├──► create_plan()                                                │
    │       │                                                                 │
    │       ├──► ResearcherAgent.execute_task()                               │
    │       │         │                                                       │
    │       │         ├──► send_message() to AnalystAgent                     │
    │       │         └──► MessageRouter ──► AnalystAgent                     │
    │       │                                                                 │
    │       ├──► AnalystAgent.execute_task()                                  │
    │       │         │                                                       │
    │       │         ├──► receive_message() from Researcher                  │
    │       │         ├──► send_message() to PlannerAgent                     │
    │       │         └──► MessageRouter ──► PlannerAgent                     │
    │       │                                                                 │
    │       └──► Aggregate all results                                        │
    │                                                                         │
    │  • Complex execution flow with message passing                          │
    │  • Agents communicate directly via message bus                          │
    │  • Health monitoring and performance tracking                           │
    │  • Load balancing and priority routing                                  │
    │  • Negotiation and conflict resolution                                  │
    └─────────────────────────────────────────────────────────────────────────┘


## REMOTE A2A SERVER INTEGRATION
### REMOTE A2A SERVER CONNECTION

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ CLIENT SIDE                                                             │
    │                                                                         │
    │  1. Parse endpoints from .env:                                          │
    │     A2A_ENDPOINTS=http://server1:8000,http://server2:8000               │
    │                                                                         │
    │  2. Register tools from each endpoint:                                  │
    │     register_all_a2a_endpoints()                                        │
    │         │                                                               │
    │         ├──► A2AClient(base_url).discover()                             │
    │         ├──► Get tool definitions                                       │
    │         └──► make_a2a_tool() wraps each tool                            │
    │                                                                         │
    │  3. Tools added to mcp_agent._tools                                     │
    │                                                                         │
    │  4. When agent calls tool:                                              │
    │     make_a2a_tool() ──► HTTP POST ──► Remote A2A Server                 │
    └─────────────────────────────────────────────────────────────────────────┘
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ REMOTE A2A SERVER                                                       │
    │                                                                         │
    │  FastAPI Server                                                         │
    │       │                                                                 │
    │       ├──► GET /discover                                                │
    │       │      └──► Return tool capabilities                              │
    │       │                                                                 │
    │       ├──► POST /execute                                                │
    │       │      ├──► Receive: tool_name, parameters                        │
    │       │      ├──► Execute local MCP tool                                │
    │       │      └──► Return: result                                        │
    │       │                                                                 │
    │       └──► WebSocket /ws (optional)                                     │
    │              └──► Real-time communication                               │
    └─────────────────────────────────────────────────────────────────────────┘


## COMPLETE DATA FLOW EXAMPLE
### EXAMPLE: "Research X and then analyze Y"

    1. USER QUERY
       │
       └──► "Research the latest AI developments and then analyze their impact"
   
    2. CLIENT.PY
        │
        ├──► should_use_multi_agent()
        │    └──► YES (multi-step: "and then")
        │
        ├──► Check A2A enabled?
        │    └──► YES (A2A_STATE["enabled"] = True)
        │
        └──► orchestrator.execute_a2a(query)

    3. ORCHESTRATOR CREATES PLAN
        │
        ├──► OrchestratorAgent.create_plan()
        │    │
        │    └──► LLM generates:
        │         {
        │           "subtasks": [
        │             {
        │               "id": "task_1",
        │               "agent": "researcher",
        │               "description": "Research latest AI developments",
        │               "depends_on": []
        │             },
        │             {
        │               "id": "task_2",
        │               "agent": "analyst", 
        │               "description": "Analyze impact of developments",
        │               "depends_on": ["task_1"]
        │             }
        │           ]
        │         }

    4. EXECUTE TASK 1 (Researcher)
        │
        ├──► ResearcherAgent.execute_task()
        │    │
        │    ├──► Calls tools: rag_search_tool, search_entries
        │    ├──► Gathers information
        │    │
        │    ├──► send_message(to="analyst_1", content=research_results)
        │    │    │
        │    │    └──► MessageRouter.route_message()
        │    │         ├──► Priority: NORMAL
        │    │         ├──► Strategy: DIRECT
        │    │         └──► AnalystAgent.receive_message()
        │    │
        │    └──► Returns: "Research completed: [findings]"
        │
        └──► HealthMonitor.record_task_completion()
             PerformanceMetrics.record_task()

    5. EXECUTE TASK 2 (Analyst)
        │
        ├──► AnalystAgent.execute_task()
        │    │
        │    ├──► Waits for task_1 dependency
        │    ├──► Receives message from Researcher
        │    ├──► Calls tools: rag_search_tool
        │    ├──► Analyzes data
        │    │
        │    └──► Returns: "Analysis: [insights]"
        │
        └──► HealthMonitor.record_task_completion()
             PerformanceMetrics.record_task()

    6. AGGREGATE RESULTS
        │
        ├──► OrchestratorAgent combines:
        │    • task_1 result (research)
        │    • task_2 result (analysis)
        │
        └──► Returns unified response to user

    7. RESPONSE TO USER
        │
        └──► "Based on recent research, AI developments show [findings].
             Analysis indicates [insights and impact]."


## FILE STRUCTURE
### PROJECT STRUCTURE

    mcp_a2a/
    ├── client.py                      # Main entry point
    │   └── run_agent_wrapper()        # Decides A2A vs standard
    │
    ├── client/
    │   ├── multi_agent.py             # MultiAgentOrchestrator
    │   │   ├── execute()              # Standard multi-agent
    │   │   └── execute_a2a()          # A2A execution
    │   │
    │   ├── agents/                    # A2A Agent implementations
    │   │   ├── base_agent.py          # BaseAgent class
    │   │   ├── orchestrator.py        # OrchestratorAgent
    │   │   ├── researcher.py          # ResearcherAgent
    │   │   ├── analyst.py             # AnalystAgent
    │   │   ├── planner.py             # PlannerAgent
    │   │   ├── writer.py              # WriterAgent
    │   │   └── plex_ingester.py      # PlexIngesterAgent
    │   │
    │   ├── message_router.py          # Message routing system
    │   ├── health_monitor.py          # Agent health tracking
    │   ├── performance_metrics.py     # Performance monitoring
    │   ├── negotiation_engine.py      # Agent negotiation
    │   │
    │   ├── a2a_client.py              # Remote A2A client
    │   └── a2a_mcp_bridge.py          # Tool wrapping
    │
    └── servers/                       # Local MCP servers
        ├── rag/
        ├── plex/
        ├── code/
        └── weather/


## KEY CONCEPTS
### KEY A2A CONCEPTS

    1. AGENT AUTONOMY
        • Each agent is independent
        • Has its own tools and capabilities
        • Maintains internal state
        • Makes decisions based on context

    2. MESSAGE PASSING
        • Agents communicate via message_bus
        • Messages routed through MessageRouter
        • Support for different priorities and strategies
        • Asynchronous communication

    3. HEALTH MONITORING
        • Tracks agent availability
        • Monitors response times
        • Detects failures early
        • Enables recovery mechanisms

    4. PERFORMANCE TRACKING
        • Records task durations
        • Tracks success rates
        • Monitors resource usage
        • Provides optimization insights

    5. DISTRIBUTED TOOLS
        • Remote A2A servers provide tools
        • Tools discovered at startup
        • HTTP/WebSocket communication
        • Transparent to agents

    6. NEGOTIATION
        • Resource allocation
        • Conflict resolution
        • Priority management
        • Load balancing