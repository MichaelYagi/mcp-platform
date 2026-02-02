# MULTI-AGENT ARCHITECTURE
MCP Multi-Agent System Flow

    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         USER REQUEST                                        │
    │                    "Research X and then analyze Y"                          │
    └──────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                    run_agent_wrapper() - Decision Point                     │
    │                                                                             │
    │  1. Check: Should use A2A? (Agent-to-Agent with message bus)                │
    │     ├─ YES → A2A_STATE["enabled"] = True                                    │
    │     └─ NO  → Continue to multi-agent check                                  │
    │                                                                             │
    │  2. Check: Should use Multi-Agent? (Multiple specialized agents)            │
    │     ├─ YES → MULTI_AGENT_STATE["enabled"] = True                            │
    │     │         + await should_use_multi_agent(message) = True                │
    │     └─ NO  → Single agent execution                                         │
    └──────────────────────────────────┬──────────────────────────────────────────┘
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             │                         │                         │
             ▼                         ▼                         ▼
        ┌────────┐              ┌──────────┐             ┌──────────┐
        │  A2A   │              │  MULTI   │             │  SINGLE  │
        │  MODE  │              │  AGENT   │             │  AGENT   │
        └────┬───┘              └─────┬────┘             └─────┬────┘
             │                        │                        │
             │                        │                        │
    ┌────────▼────────────────────────▼────────────────────────▼────────────────┐
    │                                                                           │
    │                         MULTI-AGENT FLOW                                  │
    │                    (What happens in multi-agent mode)                     │
    │                                                                           │
    └───────────────────────────────────────────────────────────────────────────┘


    ═══════════════════════════════════════════════════════════════════════════════
                               MULTI-AGENT EXECUTION FLOW
    ═══════════════════════════════════════════════════════════════════════════════
    
    STEP 1: ORCHESTRATOR CREATES PLAN
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         Base LLM (qwen2.5:14b)                              │
    │                     Role: ORCHESTRATOR (no tools)                           │
    │                                                                             │
    │  Input: "Research X and then analyze Y"                                     │
    │                                                                             │
    │  Output: JSON Plan                                                          │
    │  {                                                                          │
    │    "subtasks": [                                                            │
    │      {                                                                      │
    │        "id": "task_1",                                                      │
    │        "role": "researcher",  ← Picks agent based on task                   │
    │        "description": "Research topic X using search tools",                │
    │        "dependencies": []                                                   │
    │      },                                                                     │
    │      {                                                                      │
    │        "id": "task_2",                                                      │
    │        "role": "analyst",     ← Different agent                             │
    │        "description": "Analyze data from task_1",                           │
    │        "dependencies": ["task_1"]  ← Wait for task_1                        │
    │      }                                                                      │
    │    ]                                                                        │
    │  }                                                                          │
    └─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    STEP 2: EXECUTE TASKS IN DEPENDENCY ORDER
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                                                                             │
    │  TASK 1: RESEARCHER AGENT                                                   │
    │  ┌───────────────────────────────────────────────────────────────────┐      │
    │  │  Base LLM (qwen2.5:14b)                                           │      │
    │  │  System Prompt: "You are a Researcher Agent..."                   │      │
    │  │                                                                   │      │
    │  │  Tools Available (filtered):                                      │      │
    │  │    • rag_search_tool                                              │      │
    │  │    • search_entries                                               │      │
    │  │    • semantic_media_search_text                                   │      │
    │  │    • get_weather_tool                                             │      │
    │  │                                                                   │      │
    │  │  Execution:                                                       │      │
    │  │    1. LLM decides which tool to use                               │      │
    │  │    2. Calls rag_search_tool(query="topic X")                      │      │
    │  │    3. Tool returns results                                        │      │
    │  │    4. LLM formats response                                        │      │
    │  │                                                                   │      │
    │  │  Output: "Research results about X..."                            │      │
    │  └───────────────────────────────────────────────────────────────────┘      │
    │                                   │                                         │
    │                                   ▼                                         │
    │  Results["task_1"] = "Research results about X..."                          │
    │                                                                             │
    └───────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼ (task_1 completed, task_2 can start)
                                        │
    ┌───────────────────────────────────┴─────────────────────────────────────────┐
    │                                                                             │
    │  TASK 2: ANALYST AGENT                                                      │
    │  ┌───────────────────────────────────────────────────────────────────┐      │
    │  │  Base LLM (qwen2.5:14b) ← SAME MODEL                              │      │
    │  │  System Prompt: "You are an Analyst Agent..."                     │      │
    │  │                                                                   │      │
    │  │  Tools Available (different tools):                               │      │
    │  │    • rag_search_tool                                              │      │
    │  │    • search_entries                                               │      │
    │  │                                                                   │      │
    │  │  Context Provided:                                                │      │
    │  │    Result from task_1: "Research results about X..."              │      │
    │  │                                                                   │      │
    │  │  Execution:                                                       │      │
    │  │    1. LLM receives task_1 results as context                      │      │ 
    │  │    2. Analyzes the research results                               │      │ 
    │  │    3. May use rag_search_tool for additional data                 │      │
    │  │    4. Produces analysis                                           │      │
    │  │                                                                   │      │
    │  │  Output: "Analysis: Based on research about X, Y shows..."        │      │
    │  └───────────────────────────────────────────────────────────────────┘      │
    │                                   │                                         │
    │                                   ▼                                         │
    │  Results["task_2"] = "Analysis: Based on research..."                       │
    │                                                                             │
    └───────────────────────────────────┬─────────────────────────────────────────┘
                                        │
                                        ▼
    STEP 3: AGGREGATE RESULTS
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                         Base LLM (qwen2.5:14b)                              │
    │                     Role: Result Synthesizer (no tools)                     │
    │                                                                             │
    │  Input:                                                                     │
    │    User request: "Research X and then analyze Y"                            │
    │    task_1 result: "Research results about X..."                             │
    │    task_2 result: "Analysis: Based on research..."                          │
    │                                                                             │
    │  Task: "Synthesize these results into a coherent response"                  │
    │                                                                             │
    │  Output: Final unified response combining both results                      │
    └──────────────────────────────────┬──────────────────────────────────────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │   RETURN TO USER     │
                            └──────────────────────┘

## KEY ARCHITECTURAL POINTS

    1. SAME LLM FOR ALL AGENTS
       ┌─────────────────────────────────────────────────────────┐
       │  Orchestrator:  qwen2.5:14b (no tools)                  │
       │  Researcher:    qwen2.5:14b (search tools)              │
       │  Analyst:       qwen2.5:14b (analysis tools)            │
       │  Coder:         qwen2.5:14b (code tools)                │
       │  Writer:        qwen2.5:14b (writing tools)             │
       │  Planner:       qwen2.5:14b (todo tools)                │
       │  Plex Ingester: qwen2.5:14b (plex tools)                │
       │  Aggregator:    qwen2.5:14b (no tools)                  │
       └─────────────────────────────────────────────────────────┘

    2. SPECIALIZATION = DIFFERENT TOOLS + DIFFERENT PROMPTS
       ┌──────────────────┬────────────────────────────────────────┐
       │     Agent        │        Specialization Via              │
       ├──────────────────┼────────────────────────────────────────┤
       │  Researcher      │  • Search tools (rag, semantic)        │
       │                  │  • System prompt about research        │
       ├──────────────────┼────────────────────────────────────────┤
       │  Analyst         │  • Data analysis tools                 │
       │                  │  • System prompt about analysis        │
       ├──────────────────┼────────────────────────────────────────┤
       │  Coder           │  • Code tools (analyze, fix, etc)      │
       │                  │  • System prompt about coding          │
       ├──────────────────┼────────────────────────────────────────┤
       │  Plex Ingester   │  • plex_ingest_batch, find, stats      │
       │                  │  • System prompt about plex workflow   │
       └──────────────────┴────────────────────────────────────────┘

    3. DEPENDENCY GRAPH EXECUTION
   
        Example: "Research X, analyze Y, then write report"
        
        task_1 (researcher)  ────┐
                                │
        task_2 (analyst)     ────┼──→  task_4 (writer)
                                │      dependencies: [1,2,3]
        task_3 (researcher)  ────┘
        
        Execution order:
         Round 1: task_1, task_2, task_3 (parallel)
         Round 2: task_4 (waits for 1,2,3)

    4. STOP SIGNAL HANDLING
       
       User types ":stop"
           ↓
       is_stop_requested() = True
           ↓
       Checks happen at:
       • Before planning
       • Before each task batch
       • After each task batch
       • Before aggregation
           ↓
       Returns partial results with "🛑 Stopped" message
                            
## COMPARISON: SINGLE vs MULTI

    SINGLE AGENT:
    ┌─────────────────────────────────────────────────────────────────┐
    │  User: "What's the weather?"                                    │
    │    ↓                                                            │
    │  LLM (qwen2.5:14b) with ALL 69 tools                            │
    │    ↓                                                            │
    │  Calls get_weather_tool()                                       │
    │    ↓                                                            │
    │  Returns: "Weather is sunny, 72°F"                              │
    └─────────────────────────────────────────────────────────────────┘
    
    MULTI-AGENT:
    ┌─────────────────────────────────────────────────────────────────┐
    │  User: "Research Python frameworks and analyze their usage"     │
    │    ↓                                                            │
    │  Orchestrator (qwen2.5:14b, no tools)                           │
    │    Creates plan: [researcher task, analyst task]                │ 
    │    ↓                                                            │
    │  Task 1: Researcher (qwen2.5:14b, search tools)                 │
    │    Searches for Python frameworks                               │
    │    ↓                                                            │
    │  Task 2: Analyst (qwen2.5:14b, analysis tools)                  │
    │    Receives task 1 results, analyzes usage patterns             │
    │    ↓                                                            │
    │  Aggregator (qwen2.5:14b, no tools)                             │
    │    Combines both results into coherent response                 │
    │    ↓                                                            │
    │  Returns: "Research found: FastAPI, Django... Analysis: ..."    │
    └─────────────────────────────────────────────────────────────────┘



                              USER QUERY
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │ should_use_multi_agent? │
                    └────────┬────────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              NO                           YES
              │                             │
              ▼                             ▼
    ┌──────────────────┐        ┌──────────────────────┐
    │  SINGLE AGENT    │        │  ORCHESTRATOR        │
    │  • All 69 tools  │        │  • No tools          │
    │  • Direct answer │        │  • Creates plan      │
    └──────────────────┘        └──────┬───────────────┘
                                       │
                                       ▼
                            ┌──────────────────┐
                            │  Parse JSON Plan │
                            │  Extract subtasks│
                            └──────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │  Execute Task Graph  │
                        │  (dependency order)  │
                        └──────┬───────────────┘
                               │
                               ▼
                ┌──────────────┴────────────────┐
                │                               │
        ┌───────▼─────────┐           ┌────────▼────────┐
        │ RESEARCHER      │           │ ANALYST         │
        │ • Search tools  │           │ • Analysis tools│
        │ • Same LLM      │           │ • Same LLM      │
        └───────┬─────────┘           └────────┬────────┘
                │                               │
                └───────────┬───────────────────┘
                            │
                            ▼
                   ┌────────────────┐
                   │  AGGREGATOR    │
                   │  • Same LLM    │
                   │  • Synthesizes │
                   └────────┬───────┘
                            │
                            ▼
                      FINAL RESULT