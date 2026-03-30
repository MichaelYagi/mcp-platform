# client/a2a_client.py
from typing import Any, Dict, List
import uuid
import httpx
from urllib.parse import urljoin


class A2AClient:
    def __init__(self, base_url: str, timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.rpc_url = None

    async def _rpc(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.rpc_url:
            raise RuntimeError("Must call discover() before making RPC calls")

        payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": method,
            "params": params,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.rpc_url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        if "error" in data:
            raise RuntimeError(f"A2A error: {data['error']}")
        return data.get("result", {})

    async def discover(self):
        # 1. Fetch the agent card from the well-known location
        agent_card_url = f"{self.base_url}/.well-known/agent-card.json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(agent_card_url)
            resp.raise_for_status()
            card = resp.json()

        # 2. Extract the RPC endpoint from the card
        endpoints = card.get("endpoints", {})
        rpc_url = endpoints.get("a2a")

        if not rpc_url:
            raise ValueError("Agent card does not advertise an A2A endpoint")

        # 3. Handle relative URLs by joining with base_url
        self.rpc_url = urljoin(self.base_url + "/", rpc_url)

        # 4. Now discover available tools via RPC
        result = await self._rpc("a2a.discover", {})
        return result

    async def call(self, tool: str, arguments: Dict[str, Any]) -> Any:
        return await self._rpc("a2a.call", {
            "tool": tool,
            "arguments": arguments,
        })