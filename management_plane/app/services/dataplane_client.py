"""
Data Plane gRPC client for rule enforcement.

Internal client for Management Plane to communicate with Rust Data Plane.
"""

import logging
import grpc
import json
from typing import Optional, List, Any
from app.generated.rule_installation_pb2 import (
    EnforceRequest,
    EnforceResponse,
    InstallRulesRequest,
    QueryTelemetryRequest,
    GetSessionRequest,
    RefreshRulesRequest,
)
from app.generated.rule_installation_pb2_grpc import DataPlaneStub
from app.models import BoundaryEvidence, ComparisonResult, DesignBoundary
from app.services.policy_converter import PolicyConverter
from app.services.policy_encoder import RuleVector

logger = logging.getLogger(__name__)

class DataPlaneError(Exception):
    """Error communicating with the Data Plane gRPC server."""
    def __init__(self, message: str, status_code: Optional[grpc.StatusCode] = None):
        super().__init__(message)
        self.status_code = status_code

class DataPlaneClient:
    """
    gRPC client for the Rust Data Plane enforcement engine.
    """
    def __init__(
        self,
        url: str = "localhost:50051",
        timeout: float = 5.0,
        insecure: bool = True,
        token: Optional[str] = None,
    ):
        self.url = url
        self.timeout = timeout
        self.insecure = insecure
        self.token = token

        if insecure:
            self.channel = grpc.insecure_channel(url)
        else:
            credentials = grpc.ssl_channel_credentials()
            self.channel = grpc.secure_channel(url, credentials)

        self.stub = DataPlaneStub(self.channel)

    def enforce(
        self,
        intent: Any,  # Can be IntentEvent or dict
        intent_vector: Optional[List[float]] = None,
    ) -> ComparisonResult:
        """Enforce rules against an IntentEvent."""
        # Validate required fields for v1.3
        layer = getattr(intent, 'layer', None) or (intent.get('layer') if isinstance(intent, dict) else None)
        if not layer:
            raise ValueError("IntentEvent must include 'layer' field for enforcement")

        # Serialize IntentEvent to JSON
        try:
            if hasattr(intent, 'model_dump_json'):
                intent_json = intent.model_dump_json()
            elif isinstance(intent, dict):
                intent_json = json.dumps(intent)
            else:
                intent_json = str(intent)
        except Exception as e:
            raise ValueError(f"Failed to serialize IntentEvent: {e}")

        request = EnforceRequest(
            intent_event_json=intent_json,
            intent_vector=intent_vector or [],
        )

        metadata = []
        if self.token:
            metadata.append(("authorization", f"Bearer {self.token}"))

        try:
            response: EnforceResponse = self.stub.Enforce(
                request,
                timeout=self.timeout,
                metadata=metadata if metadata else None,
            )
            return self._convert_response(response)
        except grpc.RpcError as e:
            status_code = e.code()
            details = e.details()
            # Fail-closed: treat all errors as BLOCK (Data Plane engine does this internally too)
            raise DataPlaneError(
                f"Data Plane error [{status_code}]: {details}",
                status_code
            )

    def install_policies(
        self,
        boundaries: list[DesignBoundary],
        rule_vectors: list[RuleVector],
    ) -> dict:
        """Install policies on the Data Plane via gRPC."""

        if not boundaries or len(boundaries) != len(rule_vectors):
            raise ValueError("Boundaries and rule vectors must be non-empty and aligned")

        agent_id = boundaries[0].scope.tenantId
        rules = [
            PolicyConverter.boundary_to_rule_instance(boundary, vector, agent_id)
            for boundary, vector in zip(boundaries, rule_vectors)
        ]

        request = InstallRulesRequest(
            agent_id=agent_id,
            rules=rules,
            config_id="design_boundary_v2",
            owner="management_plane",
        )

        metadata = []
        if self.token:
            metadata.append(("authorization", f"Bearer {self.token}"))

        try:
            response = self.stub.InstallRules(
                request,
                timeout=self.timeout,
                metadata=metadata if metadata else None,
            )

            return {
                "success": response.success,
                "message": response.message,
                "rules_installed": response.rules_installed,
                "rules_by_layer": dict(response.rules_by_layer),
                "bridge_version": response.bridge_version,
            }

        except grpc.RpcError as e:
            raise DataPlaneError(f"InstallRules failed: {e.details()}", e.code())

    def _convert_response(self, response: EnforceResponse) -> ComparisonResult:
        """Convert gRPC EnforceResponse to ComparisonResult."""
        evidence = [
            BoundaryEvidence(
                boundary_id=ev.rule_id,
                boundary_name=ev.rule_name,
                effect="deny" if ev.decision == 0 else "allow",
                decision=ev.decision,
                similarities=list(ev.similarities),
            )
            for ev in response.evidence
        ]

        return ComparisonResult(
            decision=response.decision,
            slice_similarities=list(response.slice_similarities),
            boundaries_evaluated=response.rules_evaluated,
            timestamp=0.0,
            evidence=evidence,
        )

    def query_telemetry(self, **kwargs):
        request = QueryTelemetryRequest(
            limit=min(kwargs.get('limit', 50), 500),
            offset=kwargs.get('offset', 0),
        )
        if kwargs.get('agent_id'): request.agent_id = kwargs['agent_id']
        if kwargs.get('tenant_id'): request.tenant_id = kwargs['tenant_id']
        if kwargs.get('decision') is not None: request.decision = kwargs['decision']
        if kwargs.get('layer'): request.layer = kwargs['layer']
        if kwargs.get('start_time_ms') is not None: request.start_time_ms = kwargs['start_time_ms']
        if kwargs.get('end_time_ms') is not None: request.end_time_ms = kwargs['end_time_ms']
        
        try:
            return self.stub.QueryTelemetry(request, timeout=self.timeout)
        except grpc.RpcError as e:
            raise DataPlaneError(f"QueryTelemetry failed: {e.details()}", e.code())

    def get_session(self, session_id: str):
        request = GetSessionRequest(session_id=session_id)
        try:
            return self.stub.GetSession(request, timeout=self.timeout)
        except grpc.RpcError as e:
            raise DataPlaneError(f"GetSession failed: {e.details()}", e.code())

    def refresh_rules(self):
        request = RefreshRulesRequest()
        try:
            return self.stub.RefreshRules(request, timeout=self.timeout)
        except grpc.RpcError as e:
            raise DataPlaneError(f"RefreshRules failed: {e.details()}", e.code())

    def close(self):
        if hasattr(self, 'channel'):
            self.channel.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
