"""
Data contract type definitions for the Management Plane.

This module defines Pydantic models that match the canonical schemas from plan.md sections 2.1-2.4.
All types must remain synchronized across Python, TypeScript, and Rust components.

Key constraints:
- No Dict[str, Any] fields (Google GenAI compatibility)
- All fields explicitly typed
- Deterministic serialization
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, Optional


# ============================================================================
# IntentEvent Types (Section 2.2)
# ============================================================================

class Actor(BaseModel):
    """
    Represents the entity initiating an action.

    v1.2: Added "llm" and "agent" actor types for AI/autonomous systems.

    Examples:
        {"id": "user-123", "type": "user"}
        {"id": "llm-gpt4", "type": "llm"}
        {"id": "agent-123", "type": "agent"}
    """
    id: str
    type: Literal["user", "service", "llm", "agent"]


class Resource(BaseModel):
    """
    Describes the resource being accessed (v1.1 simplified).

    MVP vocabulary: database, file, api only.

    Example:
        {"type": "database", "name": "users_db", "location": "cloud"}
    """
    type: Literal["database", "file", "api"]
    name: Optional[str] = None
    location: Optional[Literal["local", "cloud"]] = None


class Data(BaseModel):
    """
    Describes the data characteristics of the operation (v1.1 simplified).

    MVP: Simplified to sensitivity (internal/public), pii flag, and volume (single/bulk).

    Example:
        {"sensitivity": ["internal"], "pii": false, "volume": "single"}
    """
    sensitivity: list[Literal["internal", "public"]]
    pii: Optional[bool] = None
    volume: Optional[Literal["single", "bulk"]] = None


class Risk(BaseModel):
    """
    Describes the risk context of the operation (v1.1 simplified).

    MVP: Only authentication requirement (required/not_required).

    Example:
        {"authn": "required"}
    """
    authn: Literal["required", "not_required"]


class LooseResource(BaseModel):
    """
    Resource model for v2 ingress before canonicalization.

    Allows non-canonical resource types while preserving other constraints.
    """

    type: str
    name: Optional[str] = None
    location: Optional[Literal["local", "cloud"]] = None


class LooseData(BaseModel):
    """
    Data model for v2 ingress before canonicalization.

    Allows non-canonical sensitivity values.
    """

    sensitivity: list[str]
    pii: Optional[bool] = None
    volume: Optional[Literal["single", "bulk"]] = None


class RateLimitContext(BaseModel):
    """
    Rate limit tracking context (v1.3).

    Tracks the number of calls within a time window for rate limit enforcement.

    Example:
        {"agent_id": "agent-123", "window_start": 1699564800.0, "call_count": 5}
    """
    agent_id: str
    window_start: float  # Unix timestamp
    call_count: int = 0


class IntentEvent(BaseModel):
    """
    Structured record of an LLM/tool call intent (v1.3 with layer-based enforcement).

    This is the canonical IntentEvent schema (plan.md section 2.2 + Appendix v1.3).
    Captured by SDKs and sent to the Management Plane for encoding and comparison.

    v1.2: Added "llm" and "agent" actor types for AI/autonomous systems.
    v1.3: Added layer-based enforcement fields (layer, tool_name, tool_method,
          tool_params, rate_limit_context) for Data Plane rule enforcement.

    Example:
        {
            "id": "550e8400-e29b-41d4-a716-446655440000",
            "schemaVersion": "v1.3",
            "tenantId": "tenant-123",
            "timestamp": 1699564800.0,
            "actor": {"id": "agent-123", "type": "agent"},
            "action": "read",
            "resource": {"type": "database", "name": "users_db", "location": "cloud"},
            "data": {"sensitivity": ["internal"], "pii": false, "volume": "single"},
            "risk": {"authn": "required"},
            "layer": "L4",
            "tool_name": "web_search",
            "tool_method": "query",
            "tool_params": {"query": "example search"},
            "rate_limit_context": {"agent_id": "agent-123", "window_start": 1699564800.0, "call_count": 5}
        }
    """
    # Existing v1.2 fields
    id: str  # UUID
    schemaVersion: Literal["v1.1", "v1.2", "v1.3"] = "v1.3"  # v1.3: Added support for v1.3
    tenantId: str
    timestamp: float  # Unix timestamp
    actor: Actor
    action: Literal["read", "write", "delete", "export", "execute", "update"]
    resource: Resource
    data: Data
    risk: Risk
    context: Optional[dict] = None  # Future extensibility

    # NEW v1.3 fields for layer-based enforcement
    layer: Optional[str] = None  # "L0", "L1", ..., "L6"
    tool_name: Optional[str] = None
    tool_method: Optional[str] = None
    tool_params: Optional[dict] = None
    rate_limit_context: Optional[RateLimitContext] = None


class LooseIntentEvent(BaseModel):
    """
    IntentEvent for v2 ingress before canonicalization.

    Uses free-form strings for fields that will be canonicalized.
    """

    id: str
    schemaVersion: Literal["v1.1", "v1.2", "v1.3"] = "v1.3"
    tenantId: str
    timestamp: float
    actor: Actor
    action: str
    resource: LooseResource
    data: LooseData
    risk: Risk
    context: Optional[dict] = None
    layer: Optional[str] = None
    tool_name: Optional[str] = None
    tool_method: Optional[str] = None
    tool_params: Optional[dict] = None
    rate_limit_context: Optional[RateLimitContext] = None


# ============================================================================
# DesignBoundary Types (Section 2.3 + v1.1 Constraints)
# ============================================================================

class BoundaryScope(BaseModel):
    """
    Defines the scope for a design boundary (tenant + optional domain filters).

    Example:
        {"tenantId": "tenant-123", "domains": ["database", "file"]}
    """
    tenantId: str
    domains: Optional[list[str]] = None  # For candidate filtering


class SliceThresholds(BaseModel):
    """
    Per-slice similarity thresholds (0.0 - 1.0).

    Each slot (action, resource, data, risk) has an independent threshold.

    Example:
        {"action": 0.85, "resource": 0.80, "data": 0.75, "risk": 0.70}
    """
    action: float = Field(ge=0.0, le=1.0)
    resource: float = Field(ge=0.0, le=1.0)
    data: float = Field(ge=0.0, le=1.0)
    risk: float = Field(ge=0.0, le=1.0)


class SliceWeights(BaseModel):
    """
    Per-slice weights for weighted-avg aggregation mode.

    Example:
        {"action": 1.0, "resource": 1.0, "data": 1.5, "risk": 0.5}
    """
    action: float = Field(default=1.0, ge=0.0)
    resource: float = Field(default=1.0, ge=0.0)
    data: float = Field(default=1.0, ge=0.0)
    risk: float = Field(default=1.0, ge=0.0)


class BoundaryRules(BaseModel):
    """
    Comparison rules for a design boundary.

    - effect: Policy effect - "allow" permits matching operations, "deny" blocks them
    - thresholds: Per-slice minimum similarity thresholds
    - weights: Optional per-slice weights (for weighted-avg mode)
    - decision: Aggregation method (min or weighted-avg)
    - globalThreshold: Overall threshold for weighted-avg mode

    Example (allow policy - min mode):
        {
            "effect": "allow",
            "thresholds": {"action": 0.85, "resource": 0.80, "data": 0.75, "risk": 0.70},
            "decision": "min"
        }

    Example (deny policy - blocks matching operations):
        {
            "effect": "deny",
            "thresholds": {"action": 0.85, "resource": 0.80, "data": 0.75, "risk": 0.70},
            "decision": "min"
        }

    Example (weighted-avg mode):
        {
            "effect": "allow",
            "thresholds": {"action": 0.85, "resource": 0.80, "data": 0.75, "risk": 0.70},
            "weights": {"action": 1.0, "resource": 1.0, "data": 1.5, "risk": 0.5},
            "decision": "weighted-avg",
            "globalThreshold": 0.78
        }
    """
    effect: Literal["allow", "deny"] = "allow"  # Default to allow for backward compatibility
    thresholds: SliceThresholds
    weights: Optional[SliceWeights] = None
    decision: Literal["min", "weighted-avg"]
    globalThreshold: Optional[float] = Field(None, ge=0.0, le=1.0)


# ============================================================================
# v1.1 Boundary Constraints (Appendix - Simplified MVP)
# ============================================================================

class ActionConstraint(BaseModel):
    """
    Defines allowed actions and actor types for v1.2 boundaries.

    v1.2: Added "llm" and "agent" to actor_types for AI/autonomous systems.

    Examples:
        {"actions": ["read", "write"], "actor_types": ["user"]}
        {"actions": ["read"], "actor_types": ["llm", "agent"]}
    """
    actions: list[Literal["read", "write", "delete", "export", "execute", "update"]]
    actor_types: list[Literal["user", "service", "llm", "agent"]]


class LooseActionConstraint(BaseModel):
    """
    Loose action constraints for v2 ingress.

    Allows non-canonical actions while preserving actor type constraints.
    """

    actions: list[str]
    actor_types: list[Literal["user", "service", "llm", "agent"]]


class ResourceConstraint(BaseModel):
    """
    Defines allowed resource types, names, and locations for v1.1 boundaries.

    MVP: Only database, file, api types; exact name matching; local or cloud only.

    Example:
        {"types": ["database"], "names": ["prod_users"], "locations": ["cloud"]}
    """
    types: list[Literal["database", "file", "api"]]
    names: Optional[list[str]] = None  # Exact match only for MVP
    locations: Optional[list[Literal["local", "cloud"]]] = None


class LooseResourceConstraint(BaseModel):
    """
    Loose resource constraints for v2 ingress.

    Allows non-canonical resource types.
    """

    types: list[str]
    names: Optional[list[str]] = None
    locations: Optional[list[Literal["local", "cloud"]]] = None


class DataConstraint(BaseModel):
    """
    Defines allowed data sensitivity levels and characteristics for v1.1 boundaries.

    MVP: Simplified to internal/public sensitivity, pii flag, single/bulk volume.

    Example:
        {"sensitivity": ["internal"], "pii": false, "volume": "single"}
    """
    sensitivity: list[Literal["internal", "public"]]
    pii: Optional[bool] = None
    volume: Optional[Literal["single", "bulk"]] = None


class LooseDataConstraint(BaseModel):
    """
    Loose data constraints for v2 ingress.

    Allows non-canonical sensitivity values.
    """

    sensitivity: list[str]
    pii: Optional[bool] = None
    volume: Optional[Literal["single", "bulk"]] = None


class RiskConstraint(BaseModel):
    """
    Defines required authentication level for v1.1 boundaries.

    MVP: Simplified to required/not_required only.

    Example:
        {"authn": "required"}
    """
    authn: Literal["required", "not_required"]


class BoundaryConstraints(BaseModel):
    """
    Complete constraint specification for v1.1 boundaries.

    Replaces the need for exemplars in MVP - boundaries encode allowed operation patterns
    using the same vocabulary as IntentEvents.

    Example:
        {
            "action": {"actions": ["read"], "actor_types": ["user"]},
            "resource": {"types": ["database"], "locations": ["cloud"]},
            "data": {"sensitivity": ["internal"], "pii": false},
            "risk": {"authn": "required"}
        }
    """
    action: ActionConstraint
    resource: ResourceConstraint
    data: DataConstraint
    risk: RiskConstraint


class LooseBoundaryConstraints(BaseModel):
    """
    Loose boundary constraints for v2 ingress.
    """

    action: LooseActionConstraint
    resource: LooseResourceConstraint
    data: LooseDataConstraint
    risk: RiskConstraint


class DesignBoundary(BaseModel):
    """
    Policy rule with constraints-based encoding (v1.2 with LangGraph support).

    Boundaries encode allowed operation patterns using same vocabulary as intents.
    This is the canonical DesignBoundary schema (plan.md Appendix v1.2).

    v1.2: Added support for "llm" and "agent" actor types in ActionConstraint.

    Example:
        {
            "id": "boundary-002",
            "name": "Safe Read Access",
            "status": "active",
            "type": "mandatory",
            "boundarySchemaVersion": "v1.2",
            "scope": {"tenantId": "tenant-123"},
            "rules": {
                "thresholds": {"action": 0.85, "resource": 0.80, "data": 0.75, "risk": 0.70},
                "decision": "min"
            },
            "constraints": {
                "action": {"actions": ["read"], "actor_types": ["user", "llm"]},
                "resource": {"types": ["database", "file"], "locations": ["cloud"]},
                "data": {"sensitivity": ["internal"], "pii": false, "volume": "single"},
                "risk": {"authn": "required"}
            },
            "createdAt": 1699564800.0,
            "updatedAt": 1699564800.0
        }
    """
    id: str
    name: str
    status: Literal["active", "disabled"]
    type: Literal["mandatory", "optional"]
    boundarySchemaVersion: Literal["v1.1", "v1.2"] = "v1.2"  # v1.2: Support both versions, default to v1.2
    scope: BoundaryScope
    layer: Optional[str] = None
    rules: BoundaryRules
    constraints: BoundaryConstraints
    notes: Optional[str] = None
    createdAt: float  # Unix timestamp
    updatedAt: float  # Unix timestamp


class LooseDesignBoundary(BaseModel):
    """
    DesignBoundary for v2 ingress before canonicalization.
    """

    id: str
    name: str
    status: Literal["active", "disabled"]
    type: Literal["mandatory", "optional"]
    boundarySchemaVersion: Literal["v1.1", "v1.2"] = "v1.2"
    scope: BoundaryScope
    layer: Optional[str] = None
    rules: BoundaryRules
    constraints: LooseBoundaryConstraints
    notes: Optional[str] = None
    createdAt: float
    updatedAt: float


# ============================================================================
# FFI Boundary Types (Section 2.4)
# ============================================================================

class BoundaryEvidence(BaseModel):
    """
    Evidence about a boundary's evaluation for debugging and audit purposes.

    Provides visibility into which boundaries were evaluated and how they contributed
    to the final decision.

    Fields:
    - boundary_id: Unique identifier for the boundary
    - boundary_name: Human-readable boundary name
    - effect: Policy effect (allow or deny)
    - decision: Individual boundary decision (0 = block, 1 = allow)
    - similarities: Per-slot similarity scores [action, resource, data, risk]

    Example:
        {
            "boundary_id": "allow-read-ops",
            "boundary_name": "Allow Read Operations",
            "effect": "allow",
            "decision": 1,
            "similarities": [0.92, 0.88, 0.85, 0.90]
        }
    """
    boundary_id: str
    boundary_name: str
    effect: Literal["allow", "deny"]
    decision: Literal[0, 1]
    similarities: list[float] = Field(min_length=4, max_length=4)


class ComparisonResult(BaseModel):
    """
    Result from Rust semantic sandbox comparison with boundary evidence.

    This matches the C struct ComparisonResult in the Rust CDylib (plan.md section 2.4),
    extended with evidence for debugging.

    Fields:
    - decision: 0 = block, 1 = allow
    - slice_similarities: Per-slot similarity scores [action, resource, data, risk]
    - boundaries_evaluated: Number of boundaries evaluated (for diagnostics)
    - timestamp: Unix timestamp of comparison
    - evidence: List of boundary evaluations (for debugging/audit)

    Example:
        {
            "decision": 1,
            "slice_similarities": [0.92, 0.88, 0.85, 0.90],
            "boundaries_evaluated": 3,
            "timestamp": 1699564800.0,
            "evidence": [...]
        }
    """
    decision: int = Field(ge=0, le=1)  # 0 = block, 1 = allow
    slice_similarities: list[float] = Field(min_length=4, max_length=4)
    boundaries_evaluated: int = Field(default=0, ge=0)
    timestamp: float = Field(default=0.0)
    evidence: list[BoundaryEvidence] = Field(default_factory=list)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "decision": 1,
                "slice_similarities": [0.92, 0.88, 0.85, 0.90],
                "boundaries_evaluated": 3,
                "timestamp": 1699564800.0,
                "evidence": []
            }
        }
    )


# ============================================================================
# Validation Vocabularies (v1.2 - LangGraph Support)
# ============================================================================

# From plan.md Appendix - Slot Contract V1.2
VALID_ACTIONS = {"read", "write", "delete", "export", "execute", "update"}
VALID_ACTOR_TYPES = {"user", "service", "llm", "agent"}  # v1.2: Added llm, agent
VALID_RESOURCE_TYPES = {"database", "file", "api"}  # Simplified from v1.0
VALID_RESOURCE_LOCATIONS = {"local", "cloud"}  # Simplified from v1.0
VALID_DATA_SENSITIVITY = {"internal", "public"}  # Replaces categories in v1.0
VALID_DATA_VOLUMES = {"single", "bulk"}  # Simplified from v1.0
VALID_AUTHN_REQUIREMENTS = {"required", "not_required"}  # Simplified from v1.0
