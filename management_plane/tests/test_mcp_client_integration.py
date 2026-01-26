import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from guard_mcp_client import GuardMCPClient


@pytest.mark.asyncio
async def test_mcp_send_intent_with_tenant_header():
    async with GuardMCPClient(
        server_url="http://localhost:3001/mcp",
        tenant_id="demo_tenant_123",
    ) as client:
        tool_names = set(await client.list_tool_names())
        print("Discovered MCP tools:", tool_names)
        assert "send_intent" in tool_names

        result = await client.send_intent(
            action="read",
            resource={
                "type": "database",
                "name": "customers",
                "location": "cloud",
            },
            data={"sensitivity": ["internal"], "pii": True, "volume": "single"},
            risk={"authn": "required"},
            context={"tool_name": "test_mcp_client"},
        )

    assert isinstance(result, dict)
    assert result.get("decision") in {"ALLOW", "DENY"}
    assert result.get("request_id")
