"""Convert canonical boundaries to protobuf RuleInstances."""

import json
from typing import Iterable

from app.generated.rule_installation_pb2 import (
    AnchorVector,
    ParamValue,
    RuleAnchorsPayload,
    RuleInstance,
)
from app.models import DesignBoundary
from app.services.policy_encoder import RuleVector


class PolicyConverter:
    """Convert canonical DesignBoundary objects into gRPC RuleInstance formats."""

    PRIORITY_MAP = {
        "mandatory": 100,
        "optional": 50,
    }

    RULE_TYPE = "design_boundary"

    @staticmethod
    def boundary_to_rule_instance(
        boundary: DesignBoundary,
        rule_vector: RuleVector,
        tenant_id: str,
    ) -> RuleInstance:
        """Build a RuleInstance protobuf message from a canonical boundary."""

        payload = PolicyConverter.rule_vector_to_anchor_payload(rule_vector)

        rule_instance = RuleInstance(
            rule_id=boundary.id,
            agent_id=tenant_id,
            priority=PolicyConverter.PRIORITY_MAP.get(boundary.type, 50),
            enabled=boundary.status == "active",
            created_at_ms=int(boundary.createdAt * 1000),
            anchors=payload,
        )

        if boundary.layer:
            rule_instance.layer = boundary.layer

        for key, value in PolicyConverter._build_params(boundary).items():
            rule_instance.params[key].CopyFrom(value)

        return rule_instance

    @staticmethod
    def _build_params(boundary: DesignBoundary) -> dict[str, ParamValue]:
        params: dict[str, ParamValue] = {}

        params["rule_type"] = PolicyConverter._string_param(PolicyConverter.RULE_TYPE)
        params["boundary_id"] = PolicyConverter._string_param(boundary.id)
        params["boundary_name"] = PolicyConverter._string_param(boundary.name)
        params["boundary_type"] = PolicyConverter._string_param(boundary.type)
        params["boundary_status"] = PolicyConverter._string_param(boundary.status)
        params["rule_effect"] = PolicyConverter._string_param(boundary.rules.effect)
        params["rule_decision"] = PolicyConverter._string_param(boundary.rules.decision)
        params["thresholds"] = PolicyConverter._json_param(
            boundary.rules.thresholds.model_dump()
        )

        if boundary.rules.weights is not None:
            params["weights"] = PolicyConverter._json_param(
                boundary.rules.weights.model_dump()
            )

        if boundary.rules.globalThreshold is not None:
            params["global_threshold"] = PolicyConverter._float_param(
                boundary.rules.globalThreshold
            )

        if boundary.constraints is not None:
            params["constraints"] = PolicyConverter._json_param(
                boundary.constraints.model_dump()
            )

        if boundary.scope is not None:
            params["scope"] = PolicyConverter._json_param(boundary.scope.model_dump())
            if boundary.scope.domains:
                params["scope_domains"] = PolicyConverter._string_list_param(
                    boundary.scope.domains
                )

        if boundary.layer:
            params["layer"] = PolicyConverter._string_param(boundary.layer)

        if boundary.notes:
            params["notes"] = PolicyConverter._string_param(boundary.notes)

        return params

    @staticmethod
    def rule_vector_to_anchor_payload(rule_vector: RuleVector) -> RuleAnchorsPayload:
        payload = RuleAnchorsPayload()
        for slot in ["action", "resource", "data", "risk"]:
            anchor_field = f"{slot}_anchors"
            count_field = f"{slot}_count"
            anchors = PolicyConverter._anchor_vectors(rule_vector.layers[slot])
            getattr(payload, anchor_field).extend(anchors)
            setattr(payload, count_field, rule_vector.anchor_counts[slot])
        return payload

    @staticmethod
    def _anchor_vectors(matrix: Iterable[Iterable[float]]) -> list[AnchorVector]:
        vectors: list[AnchorVector] = []
        for row in matrix:
            vector = AnchorVector(values=list(row))
            vectors.append(vector)
        return vectors

    @staticmethod
    def _string_param(value: str) -> ParamValue:
        param = ParamValue()
        param.string_value = value
        return param

    @staticmethod
    def _float_param(value: float) -> ParamValue:
        param = ParamValue()
        param.float_value = value
        return param

    @staticmethod
    def _json_param(value: object) -> ParamValue:
        param = ParamValue()
        param.string_value = json.dumps(value, sort_keys=True, separators=(',', ':'))
        return param

    @staticmethod
    def _string_list_param(values: Iterable[str]) -> ParamValue:
        param = ParamValue()
        param.string_list.values.extend(values)
        return param
