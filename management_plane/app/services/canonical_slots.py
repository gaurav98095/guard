"""Canonical slot serialization helpers shared by encoders.

These helpers convert canonicalized slot fields into deterministic strings that
feed the semantic encoder. They intentionally avoid any template wording or
vocabulary lookups so that intents and policies produce identical text for the
same canonical values.
"""

from __future__ import annotations

from typing import Iterable, Tuple, Any


Field = Tuple[str, Any]


def _format_value(value: Any) -> str:
    """Render values in a deterministic, template-free way."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _join_fields(fields: Iterable[Field]) -> str:
    """Serialize key/value pairs with stable ordering and separators."""

    parts: list[str] = []
    for key, value in fields:
        if value is None:
            continue
        parts.append(f"{key}={_format_value(value)}")
    return " | ".join(parts)


def serialize_action_slot(action: str, actor_type: str) -> str:
    """Serialize action slot text."""

    return _join_fields([
        ("action", action),
        ("actor_type", actor_type),
    ])


def serialize_resource_slot(
    resource_type: str,
    *,
    resource_name: str | None = None,
    resource_location: str | None = None,
) -> str:
    """Serialize resource slot text."""

    return _join_fields([
        ("resource_type", resource_type),
        ("resource_name", resource_name),
        ("resource_location", resource_location),
    ])


def serialize_data_slot(
    sensitivity: str,
    *,
    pii: bool | None = None,
    volume: str | None = None,
) -> str:
    """Serialize data slot text."""

    return _join_fields([
        ("sensitivity", sensitivity),
        ("pii", pii),
        ("volume", volume),
    ])


def serialize_risk_slot(authn: str) -> str:
    """Serialize risk slot text."""

    return _join_fields([("authn", authn)])
