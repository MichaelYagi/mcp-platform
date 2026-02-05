"""
Context Tracker - Extracts context from session history for follow-up queries
"""

import re
import os
from typing import Dict, Optional, List, Any
from langchain_core.messages import SystemMessage
MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "20"))

class ContextTracker:
    """Tracks and injects context from session database"""

    def __init__(self, session_manager):
        self.session_manager = session_manager

        self.context_patterns = {
            "project": {
                "trigger_phrases": [
                    r"."  # Match everything - we'll extract paths from ANY message
                ],
                "extract_pattern": r"(/mnt/c/[^\s\"'`]+|/[a-zA-Z0-9/_-]+/[^\s\"'`]+)",
                "key": "project_path"
            }
        }

    def extract_context_from_session(self, session_id: int, current_prompt: str) -> Dict[str, Any]:
        """Extract context from the last few tool calls in the session"""
        if not session_id or not self.session_manager:
            return {}

        messages = self.session_manager.get_session_messages(session_id)
        if not messages:
            return {}

        context = {}
        recent_messages = messages[-10:]

        # Find the most recent assistant message with substantial content
        last_response = None
        for msg in reversed(recent_messages):
            if msg["role"] == "assistant" and len(msg["text"]) > 50:
                last_response = msg["text"]
                break

        if last_response:
            # Extract entities from the last response

            # Project paths
            project_match = re.search(r'(/mnt/c/[^\s"\'`]+|/[a-zA-Z0-9/_-]+/[^\s"\'`]+)', last_response)
            if project_match:
                context["project_path"] = project_match.group(1).rstrip('`"\' ')

            # Movie/media titles
            media_match = re.search(r'(?:movie|film|show|series).*?["\']([^"\']+)["\']', last_response, re.IGNORECASE)
            if media_match:
                context["media_title"] = media_match.group(1)

            # Locations
            location_match = re.search(r'\b([A-Z][a-z]+(?:,\s*[A-Z]{2})?)\b.*?(?:weather|temperature|forecast)',
                                       last_response)
            if location_match:
                context["location"] = location_match.group(1)

            # Store the full last response as fallback
            context["last_response_preview"] = last_response[:200]

        return context

    def create_context_message(self, context: Dict) -> Optional[SystemMessage]:
        """Create system message with extracted context"""
        if not context:
            return None

        parts = ["CONVERSATION CONTEXT:\n"]

        if "project_path" in context:
            parts.append(f"Active Project: {context['project_path']}")
            parts.append(f"All follow-up questions refer to this project unless user specifies a different path.")
            parts.append(f"Use project_path=\"{context['project_path']}\" for code analysis tools.\n")

        if "media_title" in context:
            parts.append(f"Discussing Media: {context['media_title']}\n")

        if "location" in context:
            parts.append(f"Location: {context['location']}\n")

        if "last_response_preview" in context and len(parts) == 1:
            parts.append(f"Recent topic: {context['last_response_preview']}...\n")

        return SystemMessage(content="\n".join(parts))

    def should_inject_context(self, prompt: str, context: Dict) -> bool:
        """
        Simple rule: If we have context from this session, always inject it.
        Let the LLM decide if it's relevant.
        """
        return len(context) > 0

    def extract_context_from_session(self, session_id: int, current_prompt: str) -> Dict[str, Any]:
        """Extract context from recent conversation, prioritizing most recent mentions"""
        if not session_id or not self.session_manager:
            return {}

        messages = self.session_manager.get_session_messages(session_id)
        if not messages:
            return {}

        context = {}

        # Search from most recent to oldest
        for msg in reversed(messages[-MAX_MESSAGE_HISTORY:]):
            text = msg["text"]
            role = msg["role"]

            # Only extract project paths from USER messages, not assistant responses
            if "project_path" not in context and role == "user":
                # More strict regex - must start with /mnt/c/ and have reasonable length
                project_match = re.search(r'(/mnt/c/[A-Za-z0-9/_-]{10,200})', text)
                if project_match:
                    path = project_match.group(1).rstrip('`"\' ')
                    # Validate it's a real path, not garbage
                    if not any(x in path for x in ['*', ':', '**']):
                        context["project_path"] = path

            # Stop if we have what we need
            if "project_path" in context:
                break

        return context

def integrate_context_tracking(session_manager, session_id, prompt, conversation_state, logger):
    """Main integration - call before running agent"""
    if not session_manager or not session_id:
        return False

    tracker = ContextTracker(session_manager)
    context = tracker.extract_context_from_session(session_id, prompt)

    if not context:
        return False

    logger.info(f"📚 Extracted context: {list(context.keys())}")

    if not tracker.should_inject_context(prompt, context):
        return False

    context_msg = tracker.create_context_message(context)
    if context_msg:
        conversation_state["messages"].append(context_msg)
        logger.info(f"✅ Context injected: project_path={context.get('project_path')}")
        return True

    return False