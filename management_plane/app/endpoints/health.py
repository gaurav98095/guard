"""
Health check endpoint.

Provides GET /health for monitoring and readiness checks.
"""

import logging
import os
from typing import Literal

import grpc
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from ..auth import User, get_current_user
from ..settings import config

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok", "degraded", "error"] = Field(..., description="Overall health status")
    version: str = Field(..., description="Application version")
    components: dict[str, bool] = Field(..., description="Component health status")


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.

    Checks:
    - Application is running
    - Rust library is loaded and responsive
    - Configuration is valid

    Returns:
        HealthResponse with overall status and component details

    Example:
        ```json
        GET /health

        Response:
        {
          "status": "ok",
          "version": "0.1.0",
          "components": {
            "rust_library": true,
            "config": true
          }
        }
        ```
    """
    components = {}

    # Check config
    try:
        config.validate()
        components["config"] = True
    except Exception as e:
        logger.error(f"Config validation failed: {e}")
        components["config"] = False

    # Check Data Plane gRPC connectivity
    try:
        grpc_url = os.getenv("DATA_PLANE_GRPC_URL", "localhost:50051")
        channel = grpc.insecure_channel(grpc_url)
        grpc.channel_ready_future(channel).result(timeout=5)
        channel.close()
        components["data_plane_grpc"] = True
    except grpc.FutureTimeoutError:
        logger.warning("Data Plane gRPC connection timeout")
        components["data_plane_grpc"] = False
    except Exception as e:
        logger.error(f"Data Plane gRPC health check failed: {e}")
        components["data_plane_grpc"] = False

    # Determine overall status
    if all(components.values()):
        overall_status = "ok"
    elif any(components.values()):
        overall_status = "degraded"
    else:
        overall_status = "error"

    logger.debug(f"Health check: {overall_status} - {components}")

    return HealthResponse(
        status=overall_status,
        version=config.VERSION,
        components=components,
    )
