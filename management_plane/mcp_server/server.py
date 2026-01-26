from __future__ import annotations

import logging
import os

from .app import mcp


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

def run() -> None:
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "3001"))
    transport = os.getenv("MCP_SERVER_TRANSPORT", "http")
    path = os.getenv("MCP_SERVER_PATH", "/mcp")

    logger.info(
        "Starting MCP server with transport=%s host=%s port=%s path=%s",
        transport,
        host,
        port,
        path,
    )
    try:
        list_tools = getattr(mcp, "list_tools", None)
        registered_tools = list_tools() if callable(list_tools) else "n/a"
    except Exception as exc:
        registered_tools = f"error reading tools: {exc}"
    logger.info("Registered tools at startup: %s", registered_tools)
    mcp.run(transport=transport, host=host, port=port, path=path)


if __name__ == "__main__":
    run()
