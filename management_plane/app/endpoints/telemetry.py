"""
Telemetry ingestion endpoint.

Handles POST /api/v1/telemetry for storing comparison results and
security events.
"""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, status
from pydantic import BaseModel, Field

from ..auth import User, get_current_tenant

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/telemetry", tags=["telemetry"])


class TelemetryEvent(BaseModel):
    """
    Telemetry event data.

    Week 1: Simple logging
    Week 3: Store in database for analytics
    """

    eventId: str = Field(..., description="Unique event identifier")
    eventType: str = Field(..., description="Type of event (e.g., 'comparison', 'boundary_update')")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    data: dict[str, Any] = Field(..., description="Event-specific data")


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def ingest_telemetry(
    event: Annotated[TelemetryEvent, Body(..., description="Telemetry event to ingest")],
    current_user: User = Depends(get_current_tenant)
) -> dict[str, str]:
    """
    Ingest telemetry data.

    **Week 1 Implementation**: Logs to console only.
    Week 3 will implement database storage and analytics.

    Args:
        event: TelemetryEvent containing event data

    Returns:
        Acknowledgment with event ID

    Example:
        ```json
        POST /api/v1/telemetry
        {
          "eventId": "tel_123",
          "eventType": "comparison",
          "timestamp": "2025-11-12T12:00:00Z",
          "data": {
            "intentId": "evt_123",
            "decision": "allow",
            "matchedBoundary": "default_boundary",
            "similarities": [0.92, 0.88, 0.85, 0.90],
            "latencyMs": 8.5
          }
        }
        ```

        Response:
        ```json
        {
          "status": "accepted",
          "eventId": "tel_123"
        }
        ```
    """
    logger.info(f"Telemetry event: {event.eventType} - {event.eventId}")
    logger.debug(f"Telemetry data: {event.data}")

    # Week 1: Log only
    # Week 3: Store in database
    if event.eventType == "comparison":
        logger.info(
            f"  Intent: {event.data.get('intentId')} -> "
            f"Decision: {event.data.get('decision')} "
            f"(latency: {event.data.get('latencyMs', 0):.1f}ms)"
        )

    return {
        "status": "accepted",
        "eventId": event.eventId,
    }


# ============================================================================
# Telemetry Query Endpoints (Batch 2)
# ============================================================================

def get_data_plane_client():
    """
    Get Data Plane gRPC client for telemetry queries.
    
    Uses environment variable DATA_PLANE_URL or defaults to localhost:50051.
    """
    import os
    from app.services.dataplane_client import DataPlaneClient
    
    url = os.getenv("DATA_PLANE_URL", "localhost:50051")
    insecure = "localhost" in url or "127.0.0.1" in url
    return DataPlaneClient(url=url, insecure=insecure)


@router.get("/sessions", status_code=status.HTTP_200_OK)
async def query_sessions(
    current_user: User = Depends(get_current_tenant),
    agent_id: str | None = None,
    tenant_id: str | None = None,
    decision: int | None = None,
    layer: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = 50,
    offset: int = 0,
):
    """
    Query enforcement sessions from telemetry data.
    
    Returns paginated list of session summaries with filtering options.
    
    Args:
        agent_id: Filter by agent ID
        tenant_id: Filter by tenant ID
        decision: Filter by decision (0=BLOCK, 1=ALLOW)
        layer: Filter by layer (L0-L6)
        start_time_ms: Start time in milliseconds (Unix timestamp)
        end_time_ms: End time in milliseconds (Unix timestamp)
        limit: Maximum results per page (default 50, max 500)
        offset: Pagination offset
    
    Returns:
        TelemetrySessionsResponse with sessions, total_count, limit, offset
    
    Example:
        ```
        GET /api/v1/sessions?agent_id=agent_123&layer=L4&limit=10
        ```
        
        Response:
        ```json
        {
          "sessions": [
            {
              "session_id": "session_001",
              "agent_id": "agent_123",
              "tenant_id": "tenant_abc",
              "layer": "L4",
              "timestamp_ms": 1700000000000,
              "final_decision": 1,
              "rules_evaluated_count": 3,
              "duration_us": 1250,
              "intent_summary": "web_search"
            }
          ],
          "total_count": 1,
          "limit": 10,
          "offset": 0
        }
        ```
    """
    from ..telemetry_models import SessionSummary, TelemetrySessionsResponse
    
    try:
        # Get Data Plane client
        client = get_data_plane_client()

        # Cap limit at 500
        capped_limit = min(limit, 500)

        # Enforce tenant scoping - always use current user's tenant_id
        effective_tenant_id = current_user.id

        # Query telemetry via gRPC
        response = client.query_telemetry(
            agent_id=agent_id,
            tenant_id=effective_tenant_id,
            decision=decision,
            layer=layer,
            start_time_ms=start_time_ms,
            end_time_ms=end_time_ms,
            limit=capped_limit,
            offset=offset,
        )
        
        # Convert gRPC response to Pydantic models
        sessions = [
            SessionSummary(
                session_id=s.session_id,
                agent_id=s.agent_id,
                tenant_id=s.tenant_id,
                layer=s.layer,
                timestamp_ms=s.timestamp_ms,
                final_decision=s.final_decision,
                rules_evaluated_count=s.rules_evaluated_count,
                duration_us=s.duration_us,
                intent_summary=s.intent_summary,
            )
            for s in response.sessions
        ]
        
        return TelemetrySessionsResponse(
            sessions=sessions,
            total_count=response.total_count,
            limit=capped_limit,
            offset=offset,
        )
        
    except Exception as e:
        logger.error(f"Failed to query telemetry: {e}")
        from fastapi import HTTPException
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to query telemetry: {str(e)}"
        )


@router.get("/sessions/{session_id}", status_code=status.HTTP_200_OK)
async def get_session_detail(
    session_id: str,
    current_user: User = Depends(get_current_tenant),
):
    """
    Get full details for a specific enforcement session.
    
    Returns complete session data including all rule evaluations,
    intent details, and timing information.
    
    Args:
        session_id: Unique session identifier
    
    Returns:
        SessionDetail with full session data
    
    Example:
        ```
        GET /api/v1/sessions/session_001
        ```
        
        Response:
        ```json
        {
          "session": {
            "session_id": "session_001",
            "agent_id": "agent_123",
            "tenant_id": "tenant_abc",
            "layer": "L4",
            "timestamp_ms": 1700000000000,
            "final_decision": 1,
            "rules_evaluated": [
              {
                "rule_id": "rule_001",
                "decision": 1,
                "similarities": [0.92, 0.88, 0.85, 0.90]
              }
            ],
            "duration_us": 1250,
            "intent": {
              "id": "intent_123",
              "action": "read",
              "tool_name": "web_search"
            }
          }
        }
        ```
    """
    import json
    from ..telemetry_models import SessionDetail
    from fastapi import HTTPException
    
    try:
        # Get Data Plane client
        client = get_data_plane_client()
        
        # Get session via gRPC
        response = client.get_session(session_id)
        
        # Parse JSON response
        try:
            session_data = json.loads(response.session_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse session JSON: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to parse session data"
            )
        
        return SessionDetail(session=session_data)
        
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}")
        
        # Check if it's a "not found" error
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Session not found: {session_id}"
            )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )
