from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

import httpx
import logging
from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import BaseModel, Field, ValidationError

from app.models import Actor, LooseData, LooseIntentEvent, LooseResource, Risk

from .auth import authenticate_request
from .app import mcp


logger = logging.getLogger(__name__)


class SendIntentResponse(BaseModel):
    decision: str
    request_id: str
    rationale: str
    enforcement_latency_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


def _resolve_layer(context: dict[str, Any] | None) -> str | None:
    if isinstance(context, dict):
        layer = context.get("layer")
        if isinstance(layer, str) and layer.strip():
            return layer.strip()
    return None


def _resolve_tool_context(context: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(context, dict):
        return {}

    tool_context: dict[str, Any] = {}
    for key in ("tool_name", "tool_method", "tool_params"):
        if key in context:
            tool_context[key] = context.get(key)
    return tool_context


def _build_rationale(decision: str) -> str:
    if decision == "ALLOW":
        return "Guard approved the intent."
    if decision == "DENY":
        return "Guard denied the intent based on policy."
    return "Guard returned an unknown decision."


async def _call_enforce(
    event: LooseIntentEvent,
    headers: dict[str, str],
) -> dict[str, Any]:
    base_url = os.getenv("MANAGEMENT_PLANE_URL", "http://localhost:8000").rstrip("/")
    url = f"{base_url}/api/v2/enforce"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                url,
                json=event.model_dump(exclude_none=True),
                headers=headers,
            )
    except httpx.RequestError as exc:
        raise ToolError("Management Plane unavailable while enforcing intent") from exc

    if response.status_code != 200:
        detail = None
        try:
            detail = response.json().get("detail")
        except Exception:
            detail = response.text
        message = f"Enforcement failed ({response.status_code})"
        if detail:
            message = f"{message}: {detail}"
        raise ToolError(message)

    try:
        return response.json()
    except ValueError as exc:
        raise ToolError("Enforcement response was not valid JSON") from exc


logger.info("Registering MCP tool: send_intent")


@mcp.tool()
async def send_intent(
    action: str,
    resource: dict[str, Any],
    data: dict[str, Any],
    risk: dict[str, Any],
    ctx: Context | None = None,
    context: dict[str, Any] | None = None,
) -> SendIntentResponse:
    logger.info("send_intent called")
    auth_context = await authenticate_request(ctx)

    layer = _resolve_layer(context)
    tool_context = _resolve_tool_context(context)

    try:
        event = LooseIntentEvent(
            id=str(uuid.uuid4()),
            tenantId=auth_context.tenant_id,
            timestamp=time.time(),
            actor=Actor(id="llm-agent", type="agent"),
            action=action,
            resource=LooseResource(**resource),
            data=LooseData(**data),
            risk=Risk(**risk),
            context=context,
            layer=layer,
            tool_name=tool_context.get("tool_name"),
            tool_method=tool_context.get("tool_method"),
            tool_params=tool_context.get("tool_params"),
        )
    except ValidationError as exc:
        raise ToolError(f"Invalid intent payload: {exc}") from exc

    response_payload = await _call_enforce(event, auth_context.forward_headers)

    decision = response_payload.get("decision", "DENY")
    enforcement_latency_ms = response_payload.get("enforcement_latency_ms", 0.0)
    metadata = response_payload.get("metadata", {}) or {}
    request_id = metadata.get("request_id", "")

    return SendIntentResponse(
        decision=decision,
        request_id=request_id,
        rationale=_build_rationale(decision),
        enforcement_latency_ms=enforcement_latency_ms,
        metadata=metadata,
    )
