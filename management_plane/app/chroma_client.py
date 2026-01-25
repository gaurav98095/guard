"""ChromaDB helper utilities for storing rule anchor payloads."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any, Optional
from urllib.parse import urlparse

from app.settings import config

logger = logging.getLogger(__name__)

try:  # Import lazily so unit tests without chromadb still run.
    import chromadb  # type: ignore
except ImportError as exc:  # pragma: no cover - surface missing dep clearly
    chromadb = None
    logger.warning("chromadb package not installed: %s", exc)


def _build_http_client() -> Any:
    parsed = urlparse(config.CHROMA_URL)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8000
    ssl = parsed.scheme == "https"
    return chromadb.HttpClient(host=host, port=port, ssl=ssl)


def _build_persistent_client() -> Any:
    return chromadb.PersistentClient(path=config.CHROMA_URL)


@lru_cache(maxsize=1)
def get_chroma_client() -> Any:
    """Return a singleton Chroma client based on CHROMA_URL."""
    if chromadb is None:
        raise RuntimeError("chromadb dependency missing; install chromadb>=0.5.0")

    parsed = urlparse(config.CHROMA_URL)
    if parsed.scheme in ("http", "https"):
        logger.info(
            "Connecting to Chroma over HTTP (%s)",
            config.CHROMA_URL,
        )
        return _build_http_client()

    logger.info("Connecting to Chroma persistent client at %s", config.CHROMA_URL)
    return _build_persistent_client()


def _collection_name(tenant_id: str) -> str:
    return f"{config.CHROMA_COLLECTION_PREFIX}{tenant_id}".lower()


def get_rules_collection(tenant_id: str):
    """Fetch or create the rules collection for a tenant."""
    client = get_chroma_client()
    name = _collection_name(tenant_id)
    return client.get_or_create_collection(name=name, metadata={"tenant_id": tenant_id})


def upsert_rule_payload(
    tenant_id: str,
    rule_id: str,
    payload: dict[str, Any],
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Persist the rule anchor payload (JSON) to Chroma."""
    collection = get_rules_collection(tenant_id)
    document = json.dumps(payload)
    meta = {"rule_id": rule_id, **(metadata or {})}
    collection.upsert(ids=[rule_id], documents=[document], metadatas=[meta])
    logger.info("Upserted rule '%s' into Chroma collection %s", rule_id, collection.name)


def fetch_rule_payload(tenant_id: str, rule_id: str) -> Optional[dict[str, Any]]:
    """Fetch stored anchor payload for rule_id if present."""
    collection = get_rules_collection(tenant_id)
    result = collection.get(ids=[rule_id])
    documents = result.get("documents") if result else None
    if not documents:
        return None
    try:
        return json.loads(documents[0])
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Failed to decode Chroma payload for %s: %s", rule_id, exc)
        return None
