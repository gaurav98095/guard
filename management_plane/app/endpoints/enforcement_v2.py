"""
API v2 Enforcement Endpoints with Canonicalization.

New v2 endpoints that add canonicalization layer to the enforcement pipeline:
- POST /api/v2/enforce - Enforce with automatic canonicalization
- POST /api/v2/canonicalize - Debug endpoint to show canonicalization trace
- POST /api/v2/policies/install - Install policies with canonicalization

Features:
- BERT-based canonicalization of variable vocabulary
- Full trace visibility in responses
- Backward compatible (v1 endpoints unchanged)
- Async canonicalization logging

Example Request (POST /api/v2/enforce):
{
  "action": "query",  # Non-canonical term
  "actor": {"id": "user-123", "type": "user"},
  "resource": {"type": "postgres_db", "name": "users"},  # Non-canonical
  "data": {"sensitivity": ["confidential"], "pii": false, "volume": "single"},
  "risk": {"authn": "required"}
}

Example Response:
{
  "decision": "ALLOW",
  "enforcement_latency_ms": 15.2,
  "metadata": {
    "canonicalization_trace": [
      {
        "field": "action",
        "raw_input": "query",
        "prediction": {"canonical": "read", "confidence": 0.95, "source": "bert_high"}
      },
      {
        "field": "resource_type",
        "raw_input": "postgres_db",
        "prediction": {"canonical": "database", "confidence": 0.92, "source": "bert_high"}
      },
      ...
    ]
  }
}
"""

import asyncio
import logging
import os
import time
import uuid
from functools import lru_cache
from typing import Optional, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.auth import User, get_current_tenant
from app.models import ComparisonResult, IntentEvent, LooseDesignBoundary, LooseIntentEvent
from app.services import (
    BertCanonicalizer,
    CanonicalizedPredictionLogger,
    DataPlaneClient,
    DataPlaneError,
    IntentEncoder,
    PolicyEncoder,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["enforcement-v2"])


# ============================================================================
# Lazy-loaded Service Instances
# ============================================================================


@lru_cache(maxsize=1)
def get_canonicalizer() -> Optional[BertCanonicalizer]:
    """
    Get singleton BERT canonicalizer.

    Lazy-loads on first access, caches model in memory.
    """
    from app.settings import config
    from pathlib import Path

    model_path = Path(config.BERT_MODEL_PATH)
    tokenizer_path = Path(config.BERT_TOKENIZER_PATH)

    if not model_path.is_absolute():
        model_path = config.PROJECT_ROOT / model_path

    if not tokenizer_path.is_absolute():
        tokenizer_path = config.PROJECT_ROOT / tokenizer_path

    label_maps_path = model_path.parent / "label_maps.json"

    try:
        canonicalizer = BertCanonicalizer(
            model_dir=model_path.parent,
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            label_maps_path=label_maps_path,
            confidence_high=float(config.BERT_CONFIDENCE_HIGH),
            confidence_medium=float(config.BERT_CONFIDENCE_MEDIUM),
        )
        logger.info("BERT canonicalizer loaded successfully")
        return canonicalizer
    except FileNotFoundError as e:
        logger.error(f"BERT canonicalizer files missing: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load BERT canonicalizer: {e}")
        return None


@lru_cache(maxsize=1)
def get_intent_encoder() -> Optional[IntentEncoder]:
    """
    Get singleton intent encoder.

    Lazy-loads model and projection matrices on first access.
    """
    try:
        encoder = IntentEncoder()
        logger.info("Intent encoder initialized")
        return encoder
    except Exception as e:
        logger.error(f"Failed to initialize intent encoder: {e}")
        return None


@lru_cache(maxsize=1)
def get_policy_encoder() -> Optional[PolicyEncoder]:
    """
    Get singleton policy encoder.

    Lazy-loads model and projection matrices on first access.
    """
    try:
        encoder = PolicyEncoder()
        logger.info("Policy encoder initialized")
        return encoder
    except Exception as e:
        logger.error(f"Failed to initialize policy encoder: {e}")
        return None


@lru_cache(maxsize=1)
def get_canonicalization_logger() -> CanonicalizedPredictionLogger:
    """
    Get singleton canonicalization logger.

    Manages async JSONL logging with file rotation.
    """
    from app.settings import config

    log_dir = config.CANONICALIZATION_LOG_DIR
    retention_days = config.CANONICALIZATION_LOG_RETENTION_DAYS

    logger_instance = CanonicalizedPredictionLogger(
        log_dir=log_dir,
        retention_days=retention_days,
    )
    return logger_instance


@lru_cache(maxsize=1)
def get_data_plane_client():
    """Get singleton Data Plane gRPC client."""
    url = os.getenv("DATA_PLANE_URL", "localhost:50051")
    insecure = "localhost" in url or "127.0.0.1" in url
    return DataPlaneClient(url=url, insecure=insecure)


# ============================================================================
# Request/Response Models
# ============================================================================


class CanonicalizedField(BaseModel):
    """Canonicalization trace for a single field."""

    field: str = Field(..., description="Field name (action, resource_type, sensitivity)")
    raw_input: str = Field(..., description="Original input value")
    prediction: dict = Field(..., description="Prediction with canonical value and confidence")


class EnforcementResponse(BaseModel):
    """Response from v2 enforcement endpoint."""

    decision: str = Field(..., description="ALLOW or DENY")
    enforcement_latency_ms: float = Field(..., description="Time to enforce in milliseconds")
    metadata: dict = Field(default_factory=dict, description="Additional metadata including canonicalization trace")


class CanonicalizeResponse(BaseModel):
    """Response from v2 canonicalize debug endpoint."""

    canonical_intent: IntentEvent = Field(..., description="Canonicalized IntentEvent")
    canonicalization_trace: list[CanonicalizedField] = Field(..., description="Trace of all canonicalizations")


class InstallPoliciesResponse(BaseModel):
    """Response when a policy is installed via v2 endpoint."""

    status: str = Field(..., description="Installation status (installed)")
    boundary_id: str = Field(..., description="Boundary ID that was installed")
    request_id: str = Field(..., description="Request trace identifier")
    canonicalization_trace: list[CanonicalizedField] = Field(
        ..., description="Canonicalization trace applied to the boundary"
    )
    installation_stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Statistics returned by the Data Plane after installation",
    )


# ============================================================================
# Helper Functions
# ============================================================================


async def _log_prediction_async(
    logger_instance: CanonicalizedPredictionLogger,
    request_id: str,
    field: str,
    raw_input: str,
    canonical: str,
    confidence: float,
    source: str,
    enforcement_outcome: Optional[str] = None,
) -> None:
    """
    Log a prediction asynchronously (non-blocking).

    Args:
        logger_instance: Logger instance
        request_id: Request ID
        field: Field name
        raw_input: Raw input value
        canonical: Canonical value
        confidence: Confidence score
        source: Source of prediction
        enforcement_outcome: Optional enforcement result
    """
    try:
        await logger_instance.log_prediction(
            request_id=request_id,
            field=field,
            raw_input=raw_input,
            canonical=canonical,
            confidence=confidence,
            source=source,
            enforcement_outcome=enforcement_outcome,
        )
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")


# ============================================================================
# V2 Endpoints
# ============================================================================


@router.post("/enforce", response_model=EnforcementResponse, status_code=status.HTTP_200_OK)
async def enforce_v2(
    event: LooseIntentEvent,
    current_user: User = Depends(get_current_tenant),
) -> EnforcementResponse:
    """
    Enforce intent with automatic canonicalization.

    Flow:
    1. Canonicalize IntentEvent to canonical terms
    2. Encode canonical intent to 128d vector
    3. Proxy enforcement to Data Plane
    4. Log canonicalization predictions asynchronously
    5. Return decision with canonicalization trace

    Args:
        event: IntentEvent (may contain non-canonical vocabulary)
        current_user: Authenticated user

    Returns:
        EnforcementResponse with decision and canonicalization trace

    Raises:
        HTTPException: On encoding, enforcement, or service errors
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    # Set tenant_id
    event.tenantId = current_user.id

    if not event.layer:
        event.layer = "L4"

    logger.info(f"V2 enforce request: {request_id}, action={event.action}, resource={event.resource.type}")

    try:
        # Get services
        canonicalizer = get_canonicalizer()
        intent_encoder = get_intent_encoder()
        canon_logger = get_canonicalization_logger()

        if not canonicalizer or not intent_encoder or not canon_logger:
            logger.error("Required services not initialized")
            raise HTTPException(status_code=500, detail="Service initialization failed")

        # Canonicalize intent
        try:
            canonicalized = canonicalizer.canonicalize(event)
            canonical_event = canonicalized.canonical_event
            trace_dict = canonicalized.to_trace_dict()
        except Exception as e:
            logger.error(f"Canonicalization failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Canonicalization failed")

        # Log canonicalization asynchronously (non-blocking)
        for field in canonicalized.trace:
            asyncio.create_task(
                _log_prediction_async(
                    canon_logger,
                    request_id,
                    field.field_name,
                    field.raw_value,
                    field.canonical_value,
                    field.confidence,
                    field.source,
                )
            )

        # Encode canonical intent
        try:
            vector = intent_encoder.encode(canonical_event)
        except Exception as e:
            logger.error(f"Intent encoding failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Intent encoding failed")

        # Enforce via Data Plane
        client = get_data_plane_client()

        try:
            result: ComparisonResult = await asyncio.to_thread(
                client.enforce,
                canonical_event,
                vector.tolist(),
            )

            # Log enforcement outcome
            enforcement_outcome = "ALLOW" if result.decision == 1 else "DENY"
            for field in canonicalized.trace:
                asyncio.create_task(
                    _log_prediction_async(
                        canon_logger,
                        request_id,
                        field.field_name,
                        field.raw_value,
                        field.canonical_value,
                        field.confidence,
                        field.source,
                        enforcement_outcome,
                    )
                )

        except Exception as e:
            logger.error(f"Data Plane enforcement failed: {e}", exc_info=True)

            if isinstance(e, DataPlaneError):
                raise HTTPException(
                    status_code=502,
                    detail=f"Data Plane error: {e}",
                ) from e
            raise HTTPException(status_code=500, detail="Enforcement failed") from e

        # Build response
        elapsed_ms = (time.time() - start_time) * 1000

        decision = "ALLOW" if result.decision == 1 else "DENY"

        return EnforcementResponse(
            decision=decision,
            enforcement_latency_ms=elapsed_ms,
            metadata={
                "request_id": request_id,
                "canonicalization_trace": trace_dict["canonicalization_trace"],
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in V2 enforce: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e


@router.post("/canonicalize", response_model=CanonicalizeResponse, status_code=status.HTTP_200_OK)
async def canonicalize_debug(
    event: LooseIntentEvent,
    current_user: User = Depends(get_current_tenant),
) -> CanonicalizeResponse:
    """
    Debug endpoint to show canonicalization without enforcement.

    Useful for testing/validating vocabulary mappings and debugging
    canonicalization issues.

    Args:
        event: IntentEvent to canonicalize
        current_user: Authenticated user

    Returns:
        CanonicalizeResponse with canonical intent and full trace

    Raises:
        HTTPException: On canonicalization errors
    """
    request_id = str(uuid.uuid4())

    event.tenantId = current_user.id

    if not event.layer:
        event.layer = "L4"

    logger.info(f"Canonicalize debug request: {request_id}")

    try:
        canonicalizer = get_canonicalizer()
        if not canonicalizer:
            raise HTTPException(status_code=500, detail="Canonicalizer not initialized")

        canonicalized = canonicalizer.canonicalize(event)

        # Convert trace to response format
        trace_items = [
            CanonicalizedField(
                field=field.field_name,
                raw_input=field.raw_value,
                prediction={
                    "canonical": field.canonical_value,
                    "confidence": field.confidence,
                    "source": field.source,
                },
            )
            for field in canonicalized.trace
        ]

        return CanonicalizeResponse(
            canonical_intent=canonicalized.canonical_event,
            canonicalization_trace=trace_items,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Canonicalization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Canonicalization failed") from e


@router.post(
    "/policies/install",
    response_model=InstallPoliciesResponse,
    status_code=status.HTTP_201_CREATED,
)
async def install_policies_v2(
    boundary: LooseDesignBoundary,
    current_user: User = Depends(get_current_tenant),
) -> InstallPoliciesResponse:
    """
    Install policy with automatic canonicalization.

    Flow:
    1. Canonicalize DesignBoundary to canonical terms
    2. Encode canonical policy to RuleVector
    3. Install via Data Plane gRPC
    4. Log canonicalization trace
    5. Return installation status

    Args:
        boundary: DesignBoundary policy
        current_user: Authenticated user

    Returns:
        Dict with installation status and canonicalization trace

    Raises:
        HTTPException: On canonicalization, encoding, or installation errors
    """
    request_id = str(uuid.uuid4())
    boundary.scope.tenantId = current_user.id
    logger.info(f"V2 policy install request: {request_id}, boundary_id={boundary.id}")

    try:
        canonicalizer = get_canonicalizer()
        policy_encoder = get_policy_encoder()
        canon_logger = get_canonicalization_logger()

        if not canonicalizer or not policy_encoder or not canon_logger:
            raise HTTPException(status_code=500, detail="Service initialization failed")

        try:
            canonicalized = canonicalizer.canonicalize_boundary(boundary)
            canonical_boundary = canonicalized.canonical_boundary
            trace_dict = canonicalized.to_trace_dict()
        except Exception as e:
            logger.error(f"Boundary canonicalization failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Canonicalization failed")

        for field in canonicalized.trace:
            asyncio.create_task(
                _log_prediction_async(
                    canon_logger,
                    request_id,
                    field.field_name,
                    field.raw_value,
                    field.canonical_value,
                    field.confidence,
                    field.source,
                )
            )

        try:
            rule_vector = policy_encoder.encode(canonical_boundary)
        except Exception as e:
            logger.error(f"Policy encoding failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Policy encoding failed")

        client = get_data_plane_client()

        try:
            installation_stats = await asyncio.to_thread(
                client.install_policies,
                [canonical_boundary],
                [rule_vector],
            )
        except DataPlaneError as e:
            logger.error(f"Data Plane installation error: {e}", exc_info=True)
            raise HTTPException(status_code=502, detail=f"Data Plane error: {e}") from e
        except Exception as e:
            logger.error(f"Policy installation failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Policy installation failed") from e

        logger.info(f"Policy installed: {boundary.id}")

        return InstallPoliciesResponse(
            status="installed",
            boundary_id=boundary.id,
            request_id=request_id,
            canonicalization_trace=trace_dict["canonicalization_trace"],
            installation_stats=installation_stats,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unhandled error in V2 policy install: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error") from e
