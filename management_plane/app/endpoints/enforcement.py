"""Data Plane enforcement proxy endpoint."""

from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, status

from app.auth import User, get_current_tenant
from app.encoding import encode_to_128d
from app.models import IntentEvent, ComparisonResult
from app.services.dataplane_client import DataPlaneClient, DataPlaneError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enforce", tags=["enforcement"])


@lru_cache(maxsize=1)
def get_data_plane_client():
    url = os.getenv("DATA_PLANE_URL", "localhost:50051")
    insecure = "localhost" in url or "127.0.0.1" in url
    return DataPlaneClient(url=url, insecure=insecure)


@router.post("", response_model=ComparisonResult, status_code=status.HTTP_200_OK)
async def enforce_intent(
    event: IntentEvent,
    current_user: User = Depends(get_current_tenant),
) -> ComparisonResult:
    """Encode the intent locally and proxy enforcement to the Data Plane."""

    # Override tenant_id with authenticated user's ID for proper telemetry scoping
    event.tenantId = current_user.id

    try:
        vector = encode_to_128d(event)
    except Exception as exc:
        logger.error("Intent encoding failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Intent encoding failed")

    client = get_data_plane_client()

    try:
        result: ComparisonResult = await asyncio.to_thread(
            client.enforce,
            event,
            vector.tolist(),
        )
        return result
    except Exception as exc:
        logger.error("Data Plane enforcement failed: %s", exc, exc_info=True)

        if isinstance(exc, DataPlaneError):
            raise HTTPException(
                status_code=502,
                detail=f"Data Plane error: {exc}",
            ) from exc
        raise HTTPException(status_code=500, detail="Enforcement proxy failed") from exc
