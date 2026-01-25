"""
Agent management endpoints.

Handles agent registration, policy management, and template operations.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ..auth import User, get_current_tenant
from ..settings import config
from ..database import get_db, SupabaseDB
from ..nl_policy_parser import NLPolicyParser, PolicyRules
from ..policy_templates import POLICY_TEMPLATES, get_template_by_id, get_templates_by_category
from ..rule_installer import RuleInstaller

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])

# Dependency caches
_policy_parser: Optional[NLPolicyParser] = None
_rule_installer: Optional[RuleInstaller] = None


def get_policy_parser() -> NLPolicyParser:
    """Lazy-load NLPolicyParser for policy endpoints."""
    global _policy_parser
    if _policy_parser is None:
        api_key = config.get_google_api_key()
        _policy_parser = NLPolicyParser(api_key=api_key)
    return _policy_parser


def get_rule_installer() -> RuleInstaller:
    """Lazy-load RuleInstaller for policy endpoints."""
    global _rule_installer
    if _rule_installer is None:
        _rule_installer = RuleInstaller()
    return _rule_installer

# ============================================================================
# Request/Response Models
# ============================================================================

class RegisterAgentRequest(BaseModel):
    agent_id: str
    sdk_version: Optional[str] = None
    metadata: Optional[dict] = None


class RegisteredAgent(BaseModel):
    id: str
    agent_id: str
    first_seen: datetime
    last_seen: datetime
    sdk_version: Optional[str]


class ListAgentsResponse(BaseModel):
    total: int
    agents: list[RegisteredAgent]


class CreatePolicyRequest(BaseModel):
    agent_id: str
    template_id: str
    template_text: Optional[str] = None
    customization: Optional[str] = None


class AgentPolicy(BaseModel):
    id: str
    agent_id: str
    template_id: str
    template_text: str
    customization: Optional[str]
    policy_rules: PolicyRules
    embedding_metadata: Optional[dict] = None
    created_at: datetime
    updated_at: datetime


# ============================================================================
# Agent Registration Endpoints
# ============================================================================

@router.post("/register", response_model=RegisteredAgent, status_code=status.HTTP_200_OK)
async def register_agent(
    request: RegisterAgentRequest,
    current_user: User = Depends(get_current_tenant),
    db: SupabaseDB = Depends(get_db)
) -> RegisteredAgent:
    """
    Register or update an agent.

    Auto-registers agents when wrapped with enforcement_agent().
    Re-registration updates last_seen and sdk_version.

    Args:
        request: Agent registration details
        current_user: Authenticated user
        db: Database connection

    Returns:
        Registered agent details
    """
    # Check if agent already exists
    existing = db.select(
        "registered_agents",
        eq={"tenant_id": current_user.id, "agent_id": request.agent_id}
    )

    if existing:
        # Update last_seen and sdk_version
        updated = db.update(
            "registered_agents",
            {
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "sdk_version": request.sdk_version
            },
            eq={"tenant_id": current_user.id, "agent_id": request.agent_id}
        )
        if not updated:
            logger.error("Supabase update returned no rows for agent %s", request.agent_id)
            raise HTTPException(status_code=500, detail="Failed to update agent registration")
        agent = updated[0]
        logger.info(f"Agent re-registered: {request.agent_id} (tenant={current_user.id})")
    else:
        # Insert new agent
        inserted = db.insert(
            "registered_agents",
            {
                "tenant_id": current_user.id,
                "agent_id": request.agent_id,
                "sdk_version": request.sdk_version,
                "metadata": request.metadata or {}
            }
        )
        if not inserted:
            logger.error("Supabase insert returned no rows for agent %s", request.agent_id)
            raise HTTPException(status_code=500, detail="Failed to register agent")
        agent = inserted[0]
        logger.info(f"Agent registered: {request.agent_id} (tenant={current_user.id})")

    return RegisteredAgent(
        id=agent["id"],
        agent_id=agent["agent_id"],
        first_seen=_parse_timestamp(agent["first_seen"]),
        last_seen=_parse_timestamp(agent["last_seen"]),
        sdk_version=agent.get("sdk_version")
    )


@router.get("/list", response_model=ListAgentsResponse, status_code=status.HTTP_200_OK)
async def list_agents(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(get_current_tenant),
    db: SupabaseDB = Depends(get_db)
) -> ListAgentsResponse:
    """
    List all registered agents for the current tenant.

    Args:
        limit: Maximum number of agents to return
        offset: Number of agents to skip
        current_user: Authenticated user
        db: Database connection

    Returns:
        List of registered agents with total count
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    # Get total count
    total = db.count("registered_agents", eq={"tenant_id": current_user.id})

    # Get agents
    agents_data = db.select(
        "registered_agents",
        eq={"tenant_id": current_user.id},
        order="last_seen",
        desc=True,
        limit=limit,
        offset=offset
    )

    agents = [
        RegisteredAgent(
            id=a["id"],
            agent_id=a["agent_id"],
            first_seen=_parse_timestamp(a["first_seen"]),
            last_seen=_parse_timestamp(a["last_seen"]),
            sdk_version=a.get("sdk_version")
        )
        for a in agents_data
    ]

    logger.info(f"Listed {len(agents)} agents for tenant={current_user.id}")
    return ListAgentsResponse(total=total, agents=agents)


@router.post("/policies", response_model=AgentPolicy, status_code=status.HTTP_200_OK)
async def create_agent_policy(
    request: CreatePolicyRequest,
    current_user: User = Depends(get_current_tenant),
    db: SupabaseDB = Depends(get_db),
    parser: NLPolicyParser = Depends(get_policy_parser),
    installer: RuleInstaller = Depends(get_rule_installer)
) -> AgentPolicy:
    """Create or update the natural-language policy for a registered agent."""
    agent_rows = db.select(
        "registered_agents",
        eq={"tenant_id": current_user.id, "agent_id": request.agent_id},
        limit=1,
    )

    if not agent_rows:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "agent_not_registered",
                "message": f"Agent '{request.agent_id}' not found. Wrap it with enforcement_agent() first.",
                "agent_id": request.agent_id,
            },
        )

    template = get_template_by_id(request.template_id)
    if not template:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "template_not_found",
                "message": f"Template '{request.template_id}' not found.",
            },
        )

    template_text = request.template_text or template.template_text

    policy_rules = await parser.parse_policy(
        template_id=request.template_id,
        template_text=template_text,
        customization=request.customization,
    )

    # Build deterministic rule dict for persistence and installation
    rule_dict = installer.policy_to_rule(
        agent_id=request.agent_id,
        template_id=request.template_id,
        policy=policy_rules,
    )

    payload = {
        "tenant_id": current_user.id,
        "agent_id": request.agent_id,
        "template_id": request.template_id,
        "template_text": template_text,
        "customization": request.customization,
        "policy_rules": policy_rules.model_dump(),
    }

    # Persist anchors to Chroma (best effort)
    chroma_payload = await installer.persist_rule_payload(
        tenant_id=current_user.id,
        rule_dict=rule_dict,
    )
    anchors_for_install = chroma_payload.get("anchors") if chroma_payload else None
    if chroma_payload is not None:
        payload["embedding_metadata"] = {
            "rule_id": rule_dict["rule_id"],
            "chroma_synced_at": datetime.now(timezone.utc).isoformat(),
        }

    existing_policy = db.select(
        "agent_policies",
        eq={"tenant_id": current_user.id, "agent_id": request.agent_id},
        limit=1,
    )

    if existing_policy:
        result = db.update(
            "agent_policies",
            {**payload, "updated_at": datetime.now(timezone.utc).isoformat()},
            eq={"tenant_id": current_user.id, "agent_id": request.agent_id},
        )
    else:
        result = db.insert("agent_policies", payload)

    if not result:
        raise HTTPException(status_code=500, detail="Failed to persist agent policy")

    # Push policy to Data Plane as rules
    install_success = await installer.install_policy(
        agent_id=request.agent_id,
        template_id=request.template_id,
        policy=policy_rules,
        tenant_id=current_user.id,
        rule=rule_dict,
        anchors_payload=anchors_for_install,
    )

    if not install_success:
        logger.warning(
            f"Policy persisted but Data Plane installation failed for agent '{request.agent_id}'. "
            "Agent will need to fetch policy on next initialization."
        )

    return _deserialize_policy(result[0])


@router.get("/policies/{agent_id}", response_model=AgentPolicy, status_code=status.HTTP_200_OK)
async def get_agent_policy(
    agent_id: str,
    current_user: User = Depends(get_current_tenant),
    db: SupabaseDB = Depends(get_db)
) -> AgentPolicy:
    """Fetch the stored policy for an agent."""
    records = db.select(
        "agent_policies",
        eq={"tenant_id": current_user.id, "agent_id": agent_id},
        limit=1,
    )
    if not records:
        raise HTTPException(status_code=404, detail="Policy not found")
    return _deserialize_policy(records[0])


@router.delete("/policies/{agent_id}", status_code=status.HTTP_200_OK)
async def delete_agent_policy(
    agent_id: str,
    current_user: User = Depends(get_current_tenant),
    db: SupabaseDB = Depends(get_db)
) -> dict[str, bool]:
    """Delete an agent's stored policy."""
    db.delete(
        "agent_policies",
        eq={"tenant_id": current_user.id, "agent_id": agent_id},
    )
    return {"success": True}


@router.get("/templates", status_code=status.HTTP_200_OK)
async def list_policy_templates(category: Optional[str] = None) -> dict[str, list[dict]]:
    """List policy templates, optionally filtering by category."""
    templates = get_templates_by_category(category) if category else POLICY_TEMPLATES
    return {"templates": [t.model_dump() for t in templates]}


@router.get("/templates/{template_id}", status_code=status.HTTP_200_OK)
async def get_policy_template(template_id: str) -> dict:
    """Retrieve a single policy template."""
    template = get_template_by_id(template_id)
    if not template:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "template_not_found",
                "message": f"Template '{template_id}' not found.",
                "available_templates": [t.id for t in POLICY_TEMPLATES],
            },
        )
    return template.model_dump()


def _parse_timestamp(value: datetime | str) -> datetime:
    """Convert Supabase timestamp strings (with optional Z suffix) to datetime."""
    if isinstance(value, datetime):
        return value
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value)


def _deserialize_policy(record: dict) -> AgentPolicy:
    """Convert Supabase record to AgentPolicy model."""
    return AgentPolicy(
        id=record["id"],
        agent_id=record["agent_id"],
        template_id=record["template_id"],
        template_text=record["template_text"],
        customization=record.get("customization"),
        policy_rules=PolicyRules.model_validate(record["policy_rules"]),
        embedding_metadata=record.get("embedding_metadata"),
        created_at=_parse_timestamp(record.get("created_at")),
        updated_at=_parse_timestamp(record.get("updated_at", record.get("created_at"))),
    )
