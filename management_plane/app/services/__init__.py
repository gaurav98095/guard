"""
Management Plane Services

Provides core services for semantic security enforcement:
- canonicalizer: BERT-based term canonicalization
- canonicalization_logger: Async JSONL logging for predictions
- semantic_encoder: Base semantic encoding class
- intent_encoder: 128-dimensional intent vectors
- policy_encoder: 4×16×32 rule vector encoding
"""

from app.services.canonicalizer import (
    BertCanonicalizer,
    CanonicalizedBoundary,
    CanonicalizedEvent,
    CanonicalizedField,
)
from app.services.canonicalization_logger import (
    CanonicalizedPredictionLog,
    CanonicalizedPredictionLogger,
)
from app.services.intent_encoder import IntentEncoder
from app.services.dataplane_client import DataPlaneClient, DataPlaneError
from app.services.policy_encoder import PolicyEncoder, RuleVector
from app.services.policy_converter import PolicyConverter
from app.services.semantic_encoder import SemanticEncoder

__all__ = [
    "BertCanonicalizer",
    "CanonicalizedField",
    "CanonicalizedEvent",
    "CanonicalizedBoundary",
    "CanonicalizedPredictionLogger",
    "CanonicalizedPredictionLog",
    "SemanticEncoder",
    "IntentEncoder",
    "DataPlaneClient",
    "DataPlaneError",
    "PolicyEncoder",
    "RuleVector",
    "PolicyConverter",
]
