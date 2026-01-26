from __future__ import annotations

from contextlib import asynccontextmanager
import json
import logging
from typing import Any, Optional


def _clean_header_value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def _normalize_tool(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):
        return {
            "name": tool.get("name"),
            "description": tool.get("description"),
            "input_schema": tool.get("input_schema") or tool.get("inputSchema"),
            "output_schema": tool.get("output_schema") or tool.get("outputSchema"),
            "raw": tool,
        }

    return {
        "name": getattr(tool, "name", None),
        "description": getattr(tool, "description", None),
        "input_schema": getattr(tool, "input_schema", None)
        or getattr(tool, "inputSchema", None),
        "output_schema": getattr(tool, "output_schema", None)
        or getattr(tool, "outputSchema", None),
        "raw": tool,
    }


logger = logging.getLogger(__name__)


class GuardMCPClientSession:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def list_tools(self) -> list[dict[str, Any]]:
        tools = await self._client.list_tools()
        logger.info("Raw tools from server: %s", tools)
        return [_normalize_tool(tool) for tool in tools]

    async def list_tool_names(self) -> list[str]:
        tools = await self.list_tools()
        return [tool["name"] for tool in tools if tool.get("name")]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        result = await self._client.call_tool(name, arguments)
        if getattr(result, "is_error", False):
            content = getattr(result, "content", [])
            if content:
                text = getattr(content[0], "text", None)
                if text:
                    raise RuntimeError(text)
            raise RuntimeError(f"Tool call failed: {name}")

        content = getattr(result, "content", [])
        if not content:
            return result

        if len(content) == 1:
            text = getattr(content[0], "text", None)
            if isinstance(text, str):
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return content[0]

        parsed_items: list[Any] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    parsed_items.append(json.loads(text))
                except json.JSONDecodeError:
                    parsed_items.append(text)
            else:
                parsed_items.append(item)
        return parsed_items

    async def send_intent(
        self,
        *,
        action: str,
        resource: dict[str, Any],
        data: dict[str, Any],
        risk: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> Any:
        payload = {
            "action": action,
            "resource": resource,
            "data": data,
            "risk": risk,
        }
        if context is not None:
            payload["context"] = context
        return await self.call_tool("send_intent", payload)


class GuardMCPClient:
    """Standard MCP client for Guard enforcement."""

    def __init__(
        self,
        server_url: str = "http://localhost:3001/mcp",
        api_key: Optional[str] = None,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> None:
        self.server_url = server_url
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.user_id = user_id

    def _build_headers(self) -> dict[str, str]:
        tenant_id = _clean_header_value(self.tenant_id)
        user_id = _clean_header_value(self.user_id)
        api_key = _clean_header_value(self.api_key)

        if tenant_id:
            headers = {"X-Tenant-Id": tenant_id}
            if user_id:
                headers["X-User-Id"] = user_id
            return headers

        if api_key:
            return {"Authorization": f"Bearer {api_key}"}

        return {}

    @asynccontextmanager
    async def connect(self):
        headers = self._build_headers()
        logger.info("Connecting to MCP server_url=%s", self.server_url)
        if headers:
            logger.info("Using MCP headers: %s", {key: "***" if key == "Authorization" else value for key, value in headers.items()})
        else:
            logger.info("No MCP auth headers provided")

        from fastmcp import Client
        from fastmcp.client.transports import StreamableHttpTransport

        transport = StreamableHttpTransport(url=self.server_url, headers=headers)
        client = Client(transport)

        async with client:
            yield GuardMCPClientSession(client)

    async def __aenter__(self):
        self._context = self.connect()
        self.session = await self._context.__aenter__()
        return self.session

    async def __aexit__(self, *args):
        await self._context.__aexit__(*args)
