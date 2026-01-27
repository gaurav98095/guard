from __future__ import annotations

import asyncio
import logging
import os
import sys

from .app import mcp


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


async def _verify_tools_registered() -> None:
    """
    Verify that all critical tools are registered.
    
    Raises SystemExit if critical tools are missing (hard failure).
    """
    required_tools = {"send_intent"}
    
    try:
        # get_tools() is async - this returns a dict with tool names as keys
        tools = await mcp.get_tools()
        registered_names = set(tools.keys()) if isinstance(tools, dict) else set()
        
        logger.info("Registered MCP tools: %s", registered_names if registered_names else "none")
        
        missing_tools = required_tools - registered_names
        if missing_tools:
            logger.error(
                "❌ CRITICAL: Required MCP tools not registered: %s",
                ", ".join(sorted(missing_tools))
            )
            logger.error(
                "This usually means the tools module was not imported properly during initialization."
            )
            sys.exit(1)
        
        logger.info("✅ All required MCP tools are registered: %s", ", ".join(sorted(required_tools)))
    except Exception as exc:
        logger.error("❌ Failed to verify tools: %s", exc, exc_info=True)
        sys.exit(1)


def run() -> None:
    host = os.getenv("MCP_SERVER_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SERVER_PORT", "3001"))
    transport = os.getenv("MCP_SERVER_TRANSPORT", "http")
    path = os.getenv("MCP_SERVER_PATH", "/mcp")

    logger.info(
        "🚀 Starting MCP server: transport=%s host=%s port=%s path=%s",
        transport,
        host,
        port,
        path,
    )
    
    # Verify all critical tools are registered BEFORE starting the server
    try:
        asyncio.run(_verify_tools_registered())
    except Exception as exc:
        logger.error("Failed to verify tools: %s", exc, exc_info=True)
        sys.exit(1)
    
    logger.info("✅ MCP server initialization complete, starting HTTP server...")
    # Pass transport and other config as kwargs
    mcp.run(transport=transport, host=host, port=port, path=path)  # type: ignore


if __name__ == "__main__":
    run()
