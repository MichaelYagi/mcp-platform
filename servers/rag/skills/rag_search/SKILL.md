---
name: rag_search
description: >
  Perform semantic search over the RAG vector database. Use this skill when the
  user wants to search previously ingested content, browse what's stored, or
  retrieve information from web research that was automatically captured.
tags:
  - rag
  - search
  - semantic
  - browse
  - sources
tools:
  - rag_search_tool
  - rag_list_sources_tool
  - rag_browse_tool
---

# RAG Search & Browse Skill

## Overview

The RAG (Retrieval-Augmented Generation) system automatically stores content when you research topics using URLs. This skill enables you to search and browse that stored content later using semantic similarity.

## When Content Gets Stored

Content is **automatically** added to RAG when:
- Researching topics using URLs (e.g., Wikipedia articles)
- Using web search with specific sources
- The system fetches and processes web pages

**Example:**
```
User: "Write a report about quantum computing using https://wikipedia.org/wiki/Quantum_computing"
→ System fetches page, chunks it, generates embeddings, stores in RAG
→ Content available for future searches
```

## Available Tools

### 1. `rag_search_tool` - Semantic Search
Search stored content using natural language queries.

**Use when user asks:**
- "Search RAG for X"
- "What do you have about Y?"
- "Find information on Z"
- "Look up content about..."
- "Do you have anything on...?"

**Parameters:**
- `query` or `text`: Search query (accepts both for flexibility)
- `top_k`: Number of results (default: 5)
- `min_score`: Minimum similarity threshold (default: 0.0)

**Example queries:**
```
"Search RAG for quantum entanglement"
"What information do you have about Trump's health?"
"Find content about machine learning algorithms"
```

### 2. `rag_list_sources_tool` - List All Sources
Show all unique sources (URLs/documents) stored in RAG with document counts.

**Use when user asks:**
- "What sources are in RAG?"
- "List all stored content"
- "What articles have been saved?"
- "Show me what URLs are stored"

**Returns:**
- Source URLs/identifiers
- Document count per source
- Word count per source
- Sample text from each source

### 3. `rag_browse_tool` - Browse Recent Documents
Preview recent documents with text snippets.

**Use when user asks:**
- "Browse the RAG"
- "Show recent documents"
- "What was stored recently?"
- "Preview RAG content"

**Parameters:**
- `limit`: Number of documents to show (default: 10, max: 50)

**Returns:**
- Document previews (first 200 chars)
- Source information
- Word counts
- Creation timestamps

## Usage Patterns

### Pattern 1: After Research → Search Later
```
User: "Write about AI ethics using https://wikipedia.org/wiki/AI_ethics"
[System stores content automatically]

Later...
User: "What did that article say about bias?"
Assistant: [uses rag_search_tool with query="bias"]
```

### Pattern 2: Browse What's Available
```
User: "What topics do I have in RAG?"
Assistant: [uses rag_list_sources_tool to show all sources]

User: "Show me some of the content"
Assistant: [uses rag_browse_tool to preview documents]
```

### Pattern 3: Targeted Search
```
User: "Find everything about neural networks"
Assistant: [uses rag_search_tool with query="neural networks", top_k=10]
```

### Pattern 4: Check Before Research
```
User: "Do I already have information about quantum computing?"
Assistant: [uses rag_search_tool first, then suggests fetching if not found]
```

## Important Notes

### Semantic Search Behavior
- Returns **most similar** content even if not exact match
- If query term doesn't exist, returns closest semantic matches
- This is by design - helps find related information

**Example:**
```
Query: "carney"
RAG contains: Only Trump articles
Result: Returns Trump content (most similar available)
```

### Deduplication
- Same URL won't be stored twice
- Prevents redundant content
- Saves storage space and search time

### Performance
- Search takes ~1.6 seconds
- Includes embedding generation + similarity computation
- Top 5 results returned by default

## Error Handling

### Empty Results
If RAG is empty or no matches found:
```json
{
  "results": [],
  "query": "search term",
  "total_results": 0,
  "message": "No matching content found"
}
```

### Tool Parameter Flexibility
The `rag_search_tool` accepts both `query` and `text` parameters:
- Preferred: `{"query": "search term"}`
- Also works: `{"text": "search term"}`
- This handles LLM inconsistencies in tool calling

## Integration with Other Skills

### Combine with Web Search
```
1. Search RAG first (fast, already stored)
2. If insufficient, fetch new content from web
3. New content automatically stored for future
```

### Combine with Code Review
```
User: "Review this code and find similar patterns in my stored docs"
→ Use code review tools + rag_search_tool
```

## Best Practices

1. **Check RAG first** before fetching new content
2. **Use specific queries** for better semantic matching
3. **Browse sources** to understand what's available
4. **Increase top_k** for comprehensive searches (e.g., top_k=10)
5. **Use status tool** to verify content was stored after research

## Common User Phrases

**Search:**
- "look up", "find", "search for", "what do you have about"
- "retrieve", "get information on", "show me content about"

**Browse:**
- "what's in the RAG", "show me what's stored", "list sources"
- "browse database", "what articles are saved"

**Status:**
- "RAG stats", "how much is stored", "database info"
- "show RAG status", "what's in the database"