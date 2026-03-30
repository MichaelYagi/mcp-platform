"""
Plex Ingester Agent - Processes ONE Plex item at a time
"""

from typing import Optional, Dict, Any
from .base_agent import BaseAgent, AgentMessage, MessageType


class PlexIngesterAgent(BaseAgent):
    """
    Ingests individual Plex items into RAG database
    Designed for parallel execution by orchestrator
    """

    def __init__(self, agent_id: str, llm, tools, logger, message_bus):
        system_prompt = """You are a Plex Ingester Agent specialized in processing media items.

Your tools:
- plex_ingest_single: Ingest ONE specific item by ID
- plex_ingest_items: Ingest multiple items in parallel (batch)
- plex_ingest_batch: All-in-one tool (find + ingest)
- plex_get_stats: Get statistics about processed items
- rag_search_tool: Check what's already ingested

When given a task:
1. If you receive specific item IDs, use plex_ingest_items or plex_ingest_single
2. If asked to "ingest N items" without IDs, use plex_ingest_batch(limit=N)
3. After ingestion, you can call plex_get_stats to show progress

CRITICAL: Never make up item IDs! Only use IDs provided to you."""

        super().__init__(
            agent_id=agent_id,
            role="plex_ingester",
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            logger=logger,
            message_bus=message_bus
        )

    async def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle ingestion requests"""
        if message.message_type == MessageType.REQUEST:
            item_id = message.metadata.get("item_id")
            item_name = message.metadata.get("item_name", "Unknown")

            if item_id:
                # Single item ingestion
                self.logger.info(f"📥 [{self.agent_id}] Ingesting: {item_name}")

                try:
                    result = await self.call_tool("plex_ingest_single", item_id=item_id)

                    return AgentMessage(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        message_type=MessageType.RESPONSE,
                        content=f"Successfully ingested {item_name}",
                        metadata={
                            "success": True,
                            "item_id": item_id,
                            "result": result
                        }
                    )
                except Exception as e:
                    return AgentMessage(
                        from_agent=self.agent_id,
                        to_agent=message.from_agent,
                        message_type=MessageType.RESPONSE,
                        content=f"Failed to ingest {item_name}: {str(e)}",
                        metadata={
                            "success": False,
                            "item_id": item_id,
                            "error": str(e)
                        }
                    )
            else:
                # General ingestion request
                result = await self.execute_task(
                    str(message.content),
                    context=message.metadata
                )

                return AgentMessage(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    message_type=MessageType.RESPONSE,
                    content=result,
                    metadata={"ingestion_completed": True}
                )
        return None

    async def ingest_item(self, item_id: str, item_name: str = None) -> Dict[str, Any]:
        """Direct method to ingest a single item"""
        self.logger.info(f"📥 [{self.agent_id}] Ingesting item: {item_id}")

        try:
            result = await self.call_tool("plex_ingest_single", item_id=item_id)
            return {
                "success": True,
                "item_id": item_id,
                "item_name": item_name,
                "result": result
            }
        except Exception as e:
            self.logger.error(f"❌ [{self.agent_id}] Ingestion failed: {e}")
            return {
                "success": False,
                "item_id": item_id,
                "item_name": item_name,
                "error": str(e)
            }