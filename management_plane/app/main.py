"""
Management Plane FastAPI application.

Main entry point for the LLM Security Policy Enforcement Management Plane.
Provides REST API for intent comparison, boundary management, and telemetry.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .settings import config
from .endpoints import agents, auth, boundaries, encoding, enforcement, enforcement_v2, health, intents, telemetry

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.

    Handles startup and shutdown tasks:
    - Validate configuration
    - Initialize Rust library connection
    - Setup resources
    - Cleanup on shutdown
    """
    # Startup
    logger.info(f"Starting {config.APP_NAME} v{config.VERSION}")

    try:
        # Validate configuration
        config.validate()
        logger.info("Configuration validated successfully")

        # Initialize canonicalization services if enabled
        if config.CANONICALIZATION_ENABLED:
            try:
                from .endpoints.enforcement_v2 import (
                    get_canonicalizer,
                    get_intent_encoder,
                    get_policy_encoder,
                    get_canonicalization_logger,
                )

                logger.info("Initializing canonicalization services...")

                # Load canonicalizer
                start_time = time.time()
                canonicalizer = get_canonicalizer()
                if canonicalizer:
                    canon_time = (time.time() - start_time) * 1000
                    logger.info(f"BERT canonicalizer loaded in {canon_time:.1f}ms")
                else:
                    logger.warning("BERT canonicalizer not available")

                # Load intent encoder
                intent_encoder = get_intent_encoder()
                if intent_encoder:
                    logger.info("Intent encoder initialized")
                else:
                    logger.warning("Intent encoder not available")

                # Load policy encoder
                policy_encoder = get_policy_encoder()
                if policy_encoder:
                    logger.info("Policy encoder initialized")
                else:
                    logger.warning("Policy encoder not available")

                # Initialize logger
                canon_logger = get_canonicalization_logger()
                if canon_logger:
                    await canon_logger.start()
                    logger.info(f"Canonicalization logger started: {canon_logger.log_dir}")
                else:
                    logger.warning("Canonicalization logger not available")

            except Exception as e:
                logger.warning(f"Canonicalization services initialization warning: {e}")

        # Seed a default test boundary for E2E tests if none exist
        try:
            from .endpoints.boundaries import _boundaries_store  # in-memory demo store
            from .endpoints.intents import _get_test_boundary

            if not _boundaries_store:
                test_boundary = _get_test_boundary()
                _boundaries_store[test_boundary.id] = test_boundary
                logger.info(
                    "Seeded default test boundary '%s' for E2E tests",
                    test_boundary.id,
                )
        except Exception as se:
            logger.warning("Boundary seeding skipped: %s", se)

    except Exception as e:
        logger.error(f"Startup validation failed: {e}", exc_info=True)
        raise

    logger.info(f"Management Plane ready on {config.HOST}:{config.PORT}")

    yield

    # Shutdown
    logger.info("Shutting down Management Plane")

    # Cleanup canonicalization logger
    if config.CANONICALIZATION_ENABLED:
        try:
            from .endpoints.enforcement_v2 import get_canonicalization_logger

            canon_logger = get_canonicalization_logger()
            if canon_logger:
                await canon_logger.stop()
                logger.info("Canonicalization logger stopped")
        except Exception as e:
            logger.warning(f"Error stopping canonicalization logger: {e}")


# Create FastAPI application
app = FastAPI(
    title=config.APP_NAME,
    description=config.DESCRIPTION,
    version=config.VERSION,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router)
app.include_router(auth.router, prefix=config.API_V1_PREFIX)  # Token validation for gRPC proxy
app.include_router(intents.router, prefix=config.API_V1_PREFIX)
app.include_router(boundaries.router, prefix=config.API_V1_PREFIX)
app.include_router(telemetry.router, prefix=config.API_V1_PREFIX)
app.include_router(encoding.router, prefix=config.API_V1_PREFIX)  # v1.3: Encoding endpoints
app.include_router(agents.router, prefix=config.API_V1_PREFIX)  # v1.0: Agent policies
app.include_router(enforcement.router, prefix=config.API_V1_PREFIX)
app.include_router(enforcement_v2.router, prefix=config.API_V2_PREFIX)  # NEW v2: Canonicalization + Enforcement


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for uncaught errors.

    Args:
        request: The request that caused the error
        exc: The exception that was raised

    Returns:
        JSON response with error details
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": "internal_error",
        },
    )


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """
    Root endpoint.

    Returns:
        Basic API information
    """
    return {
        "name": config.APP_NAME,
        "version": config.VERSION,
        "status": "running",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=True,  # Enable auto-reload for development
    )
