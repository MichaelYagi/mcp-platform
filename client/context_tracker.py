"""
Context Tracker - Session-Scoped Semantic Context Injection

Before each LLM response, retrieves the most relevant past turns from this
session's RAG store and injects them as a SystemMessage.

Strategy:
  - RAG store  → semantic relevance  (what past context matters RIGHT NOW)
  - SQLite     → chronological truth (what the UI shows, what the LLM sees in its window)

The LLM is not replaying history — it's thinking with the relevant past.
"""

import re
import os
import logging
from typing import Dict, Optional, List, Any
from langchain_core.messages import SystemMessage

MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "20"))
# How many RAG-retrieved turns to inject (keeps prompt lean)
RAG_CONTEXT_TOP_K = int(os.getenv("RAG_CONTEXT_TOP_K", "4"))
# Minimum similarity score to include a past turn
RAG_CONTEXT_MIN_SCORE = float(os.getenv("RAG_CONTEXT_MIN_SCORE", "0.40"))

logger = logging.getLogger("mcp_client")


class ContextTracker:
    """Tracks and injects context from session RAG store"""

    def __init__(self, session_manager):
        self.session_manager = session_manager

    # ──────────────────────────────────────────────────────────────────────────
    # Primary path: semantic retrieval from session RAG
    # ──────────────────────────────────────────────────────────────────────────

    def retrieve_semantic_context(
        self,
        session_id: int,
        current_prompt: str,
        top_k: int = RAG_CONTEXT_TOP_K,
        min_score: float = RAG_CONTEXT_MIN_SCORE
    ) -> List[Dict[str, Any]]:
        """
        Query the session's RAG store for past turns relevant to current_prompt.
        Returns a list of result dicts from conversation_rag.retrieve_context().
        """
        try:
            from tools.rag.conversation_rag import retrieve_context
            results = retrieve_context(
                session_id=session_id,
                query=current_prompt,
                top_k=top_k,
                min_score=min_score
            )
            return results
        except Exception as e:
            logger.warning(f"⚠️ RAG context retrieval failed: {e}")
            return []

    def build_rag_context_message(self, results: List[Dict[str, Any]]) -> Optional[SystemMessage]:
        """
        Format retrieved RAG results into a SystemMessage for injection.
        Only called when results is non-empty.
        """
        if not results:
            return None

        lines = [
            "RELEVANT CONTEXT FROM THIS SESSION:",
            "(These are semantically relevant past exchanges — use them to maintain continuity "
            "without me repeating the full history)\n"
        ]

        for i, r in enumerate(results, 1):
            score_pct = int(r["score"] * 100)
            text = r["text"].strip()
            # Truncate very long turns for prompt efficiency
            if len(text) > 400:
                text = text[:400] + "..."
            lines.append(f"[{i}] (relevance {score_pct}%) {text}")

        lines.append("")  # trailing newline
        return SystemMessage(content="\n".join(lines))

    # ──────────────────────────────────────────────────────────────────────────
    # Fallback path: regex extraction of structured entities (file paths, etc.)
    # Kept because it's cheap, instant, and doesn't depend on embeddings
    # ──────────────────────────────────────────────────────────────────────────

    def extract_structured_context(
        self,
        session_id: int,
        current_prompt: str
    ) -> Dict[str, Any]:
        """
        Fast regex scan of recent SQLite messages for structured entities
        (project paths, media titles, locations). Used as a fallback when
        RAG retrieval yields nothing, and as a supplement for entities that
        embeddings may not reliably surface.
        """
        if not session_id or not self.session_manager:
            return {}

        messages = self.session_manager.get_session_messages(session_id)
        if not messages:
            return {}

        context = {}

        for msg in reversed(messages[-MAX_MESSAGE_HISTORY:]):
            text = msg["text"]
            role = msg["role"]

            if "project_path" not in context and role == "user":
                project_match = re.search(r'(/mnt/c/[A-Za-z0-9/_-]{10,200})', text)
                if project_match:
                    path = project_match.group(1).rstrip('`"\' ')
                    if not any(x in path for x in ['*', ':', '**']):
                        context["project_path"] = path

            if "project_path" in context:
                break

        return context

    def build_structured_context_message(self, context: Dict) -> Optional[SystemMessage]:
        """Format structured entity context into a SystemMessage."""
        if not context:
            return None

        parts = ["CONVERSATION CONTEXT:\n"]

        if "project_path" in context:
            parts.append(f"Active Project: {context['project_path']}")
            parts.append(
                f"All follow-up questions refer to this project unless user specifies a different path."
            )
            parts.append(
                f"Use project_path=\"{context['project_path']}\" for code analysis tools.\n"
            )

        if "media_title" in context:
            parts.append(f"Discussing Media: {context['media_title']}\n")

        if "location" in context:
            parts.append(f"Location: {context['location']}\n")

        if len(parts) == 1:
            return None  # Only the header — nothing to inject

        return SystemMessage(content="\n".join(parts))


# ──────────────────────────────────────────────────────────────────────────────
# Public integration function — called from client.py before each response
# ──────────────────────────────────────────────────────────────────────────────

def integrate_context_tracking(
    session_manager,
    session_id: int,
    prompt: str,
    conversation_state: dict,
    logger_
) -> bool:
    """
    Retrieve and inject relevant session context before running the agent.

    Strategy:
      1. Try semantic RAG retrieval (best relevance, requires embeddings)
      2. Fall back to structured regex extraction (fast, no ML needed)
      3. If both yield nothing, inject nothing

    Injects a SystemMessage into conversation_state["messages"].

    Returns True if any context was injected, False otherwise.
    """
    if not session_manager or not session_id:
        return False

    tracker = ContextTracker(session_manager)
    injected = False

    # ── 1. Semantic RAG context ───────────────────────────────────────────────
    try:
        rag_results = tracker.retrieve_semantic_context(session_id, prompt)

        if rag_results:
            context_msg = tracker.build_rag_context_message(rag_results)
            if context_msg:
                conversation_state["messages"].append(context_msg)
                logger_.info(
                    f"✅ RAG context injected: {len(rag_results)} relevant turns "
                    f"(top score: {rag_results[0]['score']:.2f})"
                )
                injected = True

    except Exception as e:
        logger_.warning(f"⚠️ RAG context step failed, trying fallback: {e}")

    # ── 2. Structured entity fallback ────────────────────────────────────────
    # Always run — project paths are cheap to extract and highly reliable
    try:
        structured = tracker.extract_structured_context(session_id, prompt)
        if structured:
            struct_msg = tracker.build_structured_context_message(structured)
            if struct_msg:
                conversation_state["messages"].append(struct_msg)
                logger_.info(f"📁 Structured context injected: {list(structured.keys())}")
                injected = True

    except Exception as e:
        logger_.warning(f"⚠️ Structured context extraction failed: {e}")

    return injected