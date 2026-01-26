from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Optional

from fastmcp import Context
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers

from app.auth import validate_api_key


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    api_key: Optional[str]
    authorization_header: Optional[str]
    forward_headers: dict[str, str]


def _normalize_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {key.lower(): value for key, value in headers.items()}


def _get_headers() -> dict[str, str]:
    return _normalize_headers(get_http_headers())


def _extract_bearer_token(headers: Mapping[str, str]) -> str:
    auth_header = headers.get("authorization", "").strip()
    if not auth_header:
        raise ToolError("Unauthorized: missing Authorization header")

    prefix = "Bearer "
    if not auth_header.startswith(prefix):
        raise ToolError("Unauthorized: Authorization header must use Bearer scheme")

    token = auth_header[len(prefix) :].strip()
    if not token:
        raise ToolError("Unauthorized: missing API key")

    return token


def _extract_tenant_headers(headers: Mapping[str, str]) -> tuple[Optional[str], Optional[str]]:
    tenant_id = headers.get("x-tenant-id")
    if tenant_id is None:
        return None, None

    tenant_id = tenant_id.strip()
    if not tenant_id:
        raise ToolError("Unauthorized: missing X-Tenant-Id header")

    user_id = headers.get("x-user-id")
    if user_id is not None:
        user_id = user_id.strip() or None

    return tenant_id, user_id


async def authenticate_request(ctx: Optional[Context] = None) -> AuthContext:
    headers = _get_headers()

    logger.debug("MCP auth headers received: %s", {key: "***" if key == "authorization" else value for key, value in headers.items()})

    tenant_id, user_id = _extract_tenant_headers(headers)
    if tenant_id:
        logger.info("Authenticated via X-Tenant-Id header")
        forward_headers = {"X-Tenant-Id": tenant_id}
        if user_id:
            forward_headers["X-User-Id"] = user_id
        return AuthContext(
            tenant_id=tenant_id,
            api_key=None,
            authorization_header=None,
            forward_headers=forward_headers,
        )

    api_key = _extract_bearer_token(headers)
    logger.info("Authenticated via Authorization Bearer token")

    user = await validate_api_key(api_key)
    if not user:
        raise ToolError("Unauthorized: invalid API key")

    return AuthContext(
        tenant_id=user.id,
        api_key=api_key,
        authorization_header=f"Bearer {api_key}",
        forward_headers={"Authorization": f"Bearer {api_key}"},
    )
logger = logging.getLogger(__name__)
