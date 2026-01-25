"""
Rule Installer - Converts NL policies to Data Plane rules.

Handles:
1. PolicyRules → RuleInstance conversion
2. gRPC communication with Data Plane
3. Rule installation for agent policies
"""

import logging
import time
from typing import Optional
from .nl_policy_parser import PolicyRules
from .settings import config
from .chroma_client import upsert_rule_payload, fetch_rule_payload
from .rule_encoding import build_tool_whitelist_anchors

logger = logging.getLogger(__name__)

# Import gRPC if available
try:
    import grpc
    from app.generated.rule_installation_pb2 import (
        AnchorVector,
        InstallRulesRequest,
        RemoveAgentRulesRequest,
        RuleAnchorsPayload,
        RuleInstance,
        ParamValue,
        StringList,
    )
    from app.generated.rule_installation_pb2_grpc import DataPlaneStub
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    logger.warning("gRPC not available - rule installation disabled")


class RuleInstaller:
    """
    Converts PolicyRules to Data Plane rules and installs them.

    Architecture:
    - NL Policy (PolicyRules) → ToolWhitelist rule for L4 layer
    - One rule per agent policy
    - Uses gRPC to push to Data Plane
    """

    def __init__(self, data_plane_url: Optional[str] = None):
        """
        Initialize RuleInstaller.

        Args:
            data_plane_url: Data Plane gRPC URL (default: from config or localhost:50051)
        """
        if not GRPC_AVAILABLE:
            raise RuntimeError("gRPC not available - install grpcio and grpcio-tools")

        self.data_plane_url = data_plane_url or config.data_plane_url or "localhost:50051"
        logger.info(f"RuleInstaller initialized: data_plane={self.data_plane_url}")

    def policy_to_rule(
        self,
        agent_id: str,
        template_id: str,
        policy: PolicyRules
    ) -> dict:
        """
        Convert PolicyRules to rule dictionary.

        Args:
            agent_id: Agent identifier
            template_id: Template ID (used for rule_id)
            policy: Structured policy rules

        Returns:
            Dictionary representing a ToolWhitelist rule
        """
        # Generate rule ID from agent + template
        rule_id = f"{agent_id}:{template_id}"

        # Build params for ToolWhitelist rule
        # Maps PolicyRules constraints to rule parameters
        params = {
            # Action slice: allowed actions
            "allowed_actions": policy.constraints.action.actions,

            # Resource slice: resource types
            "allowed_resource_types": policy.constraints.resource.types,

            # Data slice: sensitivity levels
            "allowed_sensitivity": policy.constraints.data.sensitivity,

            # Risk slice: authentication requirement
            "require_authn": policy.constraints.risk.authn == "required",

            # Thresholds (per-slot)
            "action_threshold": policy.thresholds.action,
            "resource_threshold": policy.thresholds.resource,
            "data_threshold": policy.thresholds.data,
            "risk_threshold": policy.thresholds.risk,

            # Decision mode
            "decision_mode": policy.decision,
        }

        # Add optional resource constraints
        if policy.constraints.resource.names:
            params["allowed_resource_names"] = policy.constraints.resource.names

        if policy.constraints.resource.locations:
            params["allowed_locations"] = policy.constraints.resource.locations

        # Add optional data constraints
        if policy.constraints.data.pii is not None:
            params["allow_pii"] = policy.constraints.data.pii

        if policy.constraints.data.volume:
            params["allowed_volume"] = policy.constraints.data.volume

        return {
            "rule_id": rule_id,
            "family_id": "tool_whitelist",  # Must match Data Plane snake_case naming
            "layer": "L4",  # NL policies apply to tool gateway layer
            "agent_id": agent_id,
            "priority": 100,  # Default priority
            "enabled": True,
            "created_at_ms": int(time.time() * 1000),
            "params": params,
        }

    def _param_to_proto(self, value) -> ParamValue:
        """Convert Python value to ParamValue proto."""
        if isinstance(value, list):
            # String list
            return ParamValue(string_list=StringList(values=value))
        elif isinstance(value, str):
            return ParamValue(string_value=value)
        elif isinstance(value, bool):
            return ParamValue(bool_value=value)
        elif isinstance(value, int):
            return ParamValue(int_value=value)
        elif isinstance(value, float):
            return ParamValue(float_value=value)
        else:
            raise ValueError(f"Unsupported param type: {type(value)}")

    def _dict_to_proto_rule(self, rule_dict: dict) -> RuleInstance:
        """Convert rule dictionary to proto RuleInstance."""
        # Convert params
        proto_params = {}
        for key, value in rule_dict.get("params", {}).items():
            proto_params[key] = self._param_to_proto(value)

        return RuleInstance(
            rule_id=rule_dict["rule_id"],
            family_id=rule_dict["family_id"],
            layer=rule_dict["layer"],
            agent_id=rule_dict["agent_id"],
            priority=rule_dict.get("priority", 100),
            enabled=rule_dict.get("enabled", True),
            created_at_ms=rule_dict.get("created_at_ms", int(time.time() * 1000)),
            params=proto_params,
        )

    def _anchors_dict_to_proto(self, anchors: dict) -> RuleAnchorsPayload:
        """Convert stored anchor dict to RuleAnchorsPayload proto."""

        def _to_vectors(key: str) -> list[AnchorVector]:
            rows = anchors.get(key) or []
            vectors: list[AnchorVector] = []
            for row in rows:
                vectors.append(AnchorVector(values=[float(v) for v in row]))
            return vectors

        return RuleAnchorsPayload(
            action_anchors=_to_vectors("action_anchors"),
            action_count=int(anchors.get("action_count", 0)),
            resource_anchors=_to_vectors("resource_anchors"),
            resource_count=int(anchors.get("resource_count", 0)),
            data_anchors=_to_vectors("data_anchors"),
            data_count=int(anchors.get("data_count", 0)),
            risk_anchors=_to_vectors("risk_anchors"),
            risk_count=int(anchors.get("risk_count", 0)),
        )

    async def persist_rule_payload(self, tenant_id: str, rule_dict: dict) -> Optional[dict]:
        """Encode anchors for rule_dict and store them in Chroma."""
        try:
            anchors = await build_tool_whitelist_anchors(rule_dict)
            payload = {
                "rule": rule_dict,
                "anchors": anchors,
            }
            metadata = {
                "agent_id": rule_dict.get("agent_id"),
                "family_id": rule_dict.get("family_id"),
                "layer": rule_dict.get("layer"),
            }
            upsert_rule_payload(tenant_id, rule_dict["rule_id"], payload, metadata)
            return payload
        except Exception as exc:
            logger.error(
                "Failed to persist rule payload for %s: %s",
                rule_dict.get("rule_id"),
                exc,
            )
            return None

    def get_stored_rule_payload(self, tenant_id: str, rule_id: str) -> Optional[dict]:
        """Fetch stored rule payload (anchors) from Chroma."""
        try:
            return fetch_rule_payload(tenant_id, rule_id)
        except Exception as exc:
            logger.warning("Failed to fetch rule payload for %s: %s", rule_id, exc)
            return None

    async def install_policy(
        self,
        agent_id: str,
        template_id: str,
        policy: PolicyRules,
        tenant_id: str,
        rule: Optional[dict] = None,
        anchors_payload: Optional[dict] = None,
    ) -> bool:
        """
        Install agent policy as Data Plane rule.

        Args:
            agent_id: Agent identifier
            template_id: Template ID
            policy: Structured policy rules

        Returns:
            True if installation succeeded, False otherwise
        """
        logger.info("=" * 60)
        logger.info(f"INSTALLING POLICY FOR AGENT: {agent_id}")
        logger.info(f"  Template ID: {template_id}")
        logger.info(f"  Tenant ID: {tenant_id}")
        logger.info(f"  Data Plane URL: {self.data_plane_url}")
        logger.info("=" * 60)

        try:
            # Convert policy to rule if needed
            rule_dict = rule or self.policy_to_rule(agent_id, template_id, policy)
            rule_proto = self._dict_to_proto_rule(rule_dict)

            logger.info(f"Rule converted to proto: rule_id={rule_dict['rule_id']}, layer={rule_dict['layer']}")

            # Attach anchors if provided or retrievable from Chroma
            anchors_data = anchors_payload
            if anchors_data is None:
                logger.info(f"No anchors provided, fetching from Chroma for rule_id={rule_dict['rule_id']}")
                stored_payload = self.get_stored_rule_payload(tenant_id, rule_dict["rule_id"])
                if stored_payload:
                    anchors_data = stored_payload.get("anchors")
                    logger.info(f"Retrieved anchors from Chroma: {len(anchors_data) if anchors_data else 0} slots")
                else:
                    logger.warning(f"No stored payload found in Chroma for rule_id={rule_dict['rule_id']}")
            else:
                logger.info(f"Using provided anchors_payload")

            if anchors_data:
                try:
                    rule_proto.anchors.CopyFrom(self._anchors_dict_to_proto(anchors_data))
                    logger.info(f"Successfully attached anchors to rule proto")
                except Exception as exc:
                    logger.warning(
                        "Failed to attach anchors for %s: %s",
                        rule_dict["rule_id"],
                        exc,
                    )
            else:
                logger.warning(f"No anchors available for rule {rule_dict['rule_id']} - Data Plane will have no embeddings!")

            # Determine if remote (requires TLS)
            is_remote = "localhost" not in self.data_plane_url and \
                       "127.0.0.1" not in self.data_plane_url

            logger.info(f"Creating gRPC channel: url={self.data_plane_url}, is_remote={is_remote}")

            # Create gRPC channel
            if is_remote:
                # Use TLS for remote connections
                credentials = grpc.ssl_channel_credentials()
                channel = grpc.aio.secure_channel(self.data_plane_url, credentials)
                logger.info("Using secure gRPC channel with TLS")
            else:
                # Insecure for local connections
                channel = grpc.aio.insecure_channel(self.data_plane_url)
                logger.info("Using insecure gRPC channel (local)")

            try:
                # Create stub and call
                stub = DataPlaneStub(channel)
                # Remove previously installed rules for this agent to avoid stale policies
                try:
                    logger.info(f"Calling RemoveAgentRules for agent_id={agent_id}")
                    removal_response = await stub.RemoveAgentRules(
                        RemoveAgentRulesRequest(agent_id=agent_id),
                        timeout=5.0
                    )
                    if removal_response.success:
                        logger.info(
                            "Removed %d existing rules for agent '%s' before reinstall",
                            removal_response.rules_removed,
                            agent_id,
                        )
                    else:
                        logger.warning(
                            "Failed to remove existing rules for agent '%s': %s",
                            agent_id,
                            removal_response.message,
                        )
                except Exception as removal_error:
                    logger.warning(
                        "Error removing existing rules for agent '%s': %s",
                        agent_id,
                        removal_error,
                    )

                request = InstallRulesRequest(
                    agent_id=agent_id,
                    rules=[rule_proto],
                    config_id=template_id,
                    owner="management-plane"
                )

                logger.info(f"Calling InstallRules: agent_id={agent_id}, num_rules={len(request.rules)}, config_id={template_id}")
                response = await stub.InstallRules(request, timeout=5.0)

                if response.success:
                    logger.info(
                        f"✅ Policy installed for agent '{agent_id}': "
                        f"{response.rules_installed} rules in {response.rules_by_layer}"
                    )
                    logger.info("=" * 60)
                    return True
                else:
                    logger.error(f"❌ Policy installation failed: {response.message}")
                    logger.info("=" * 60)
                    return False

            finally:
                await channel.close()
                logger.info("gRPC channel closed")

        except Exception as e:
            logger.error(f"❌ Exception during policy installation for agent '{agent_id}': {e}", exc_info=True)
            logger.info("=" * 60)
            return False
