"""
Policy Encoder for RuleVector anchor encoding.

Subclass of SemanticEncoder that:
1. Extracts anchors from canonical DesignBoundary for 4 layers
2. Encodes each anchor to 32-dimensional vector
3. Aggregates to 4×16×32 RuleVector structure:
   - 4 layers: action, resource, data, risk
   - 16 anchors per layer (padded with zeros if fewer)
   - 32 dimensions per anchor

The base class handles:
- Model loading (sentence-transformers)
- Embedding generation (384d)
- Projection matrix creation (sparse random projection)
- Caching

This class adds:
- Anchor extraction from DesignBoundary constraints
- Anchor encoding with max/padding logic
- RuleVector aggregation

Example:
    encoder = PolicyEncoder()
    canonical_boundary = DesignBoundary(...)
    rule_vector = encoder.encode(canonical_boundary)  # Returns RuleVector(4, 16, 32)
"""

import logging
from typing import Tuple

import numpy as np

from app.models import DesignBoundary
from app.services.semantic_encoder import SemanticEncoder
from app.services.canonical_slots import (
    serialize_action_slot,
    serialize_resource_slot,
    serialize_data_slot,
    serialize_risk_slot,
)

logger = logging.getLogger(__name__)


class RuleVector:
    """
    4×16×32 tensor representation of policy boundaries.

    Structure:
    - 4 layers: action, resource, data, risk
    - Each layer has 16 anchor slots (padded with zeros if fewer anchors)
    - Each anchor is 32-dimensional

    Flattened shape: (4, 16, 32) → can be reshaped to (2048,) for comparison
    """

    def __init__(self):
        """Initialize empty RuleVector."""
        self.layers = {
            "action": np.zeros((16, 32), dtype=np.float32),
            "resource": np.zeros((16, 32), dtype=np.float32),
            "data": np.zeros((16, 32), dtype=np.float32),
            "risk": np.zeros((16, 32), dtype=np.float32),
        }
        self.anchor_counts = {
            "action": 0,
            "resource": 0,
            "data": 0,
            "risk": 0,
        }

    def set_layer(self, layer_name: str, anchor_vectors: np.ndarray, count: int) -> None:
        """
        Set anchor vectors for a layer.

        Args:
            layer_name: Name of layer (action, resource, data, risk)
            anchor_vectors: Array of shape (16, 32) with encoded anchors
            count: Actual number of anchors (before padding)
        """
        self.layers[layer_name] = anchor_vectors
        self.anchor_counts[layer_name] = count

    def to_numpy(self) -> np.ndarray:
        """
        Convert to flattened numpy array.

        Returns:
            Flattened array of shape (2048,) = 4 × 16 × 32
        """
        stacked = np.stack([self.layers[name] for name in ["action", "resource", "data", "risk"]])
        return stacked.flatten()

    def to_dict(self) -> dict:
        """
        Convert to dictionary representation.

        Returns:
            Dict with layer arrays and anchor counts
        """
        return {
            "layers": {name: vec.tolist() for name, vec in self.layers.items()},
            "anchor_counts": self.anchor_counts,
        }


class PolicyEncoder(SemanticEncoder):
    """
    Semantic encoder for DesignBoundary to RuleVector.

    Encodes canonical DesignBoundary by:
    1. Extracting anchors for each constraint layer
    2. Encoding each anchor to 32-dim vector
    3. Aggregating with padding to 16×32 per layer
    4. Stacking to 4×16×32 RuleVector
    """

    MAX_ANCHORS_PER_LAYER = 16

    def __init__(self, embedding_model: str = SemanticEncoder.MODEL_NAME):
        """
        Initialize policy encoder.

        Args:
            embedding_model: Name of sentence-transformers model
        """
        super().__init__(embedding_model=embedding_model)

    def _extract_action_anchors(self, boundary: DesignBoundary) -> list[str]:
        """
        Extract anchor strings for action layer.

        One anchor per (action, actor_type) combination.

        Args:
            boundary: Canonical DesignBoundary

        Returns:
            List of anchor strings
        """
        anchors = []
        for action in sorted(boundary.constraints.action.actions):
            for actor_type in sorted(boundary.constraints.action.actor_types):
                anchors.append(serialize_action_slot(action, actor_type))
        return anchors

    def _extract_resource_anchors(self, boundary: DesignBoundary) -> list[str]:
        """
        Extract anchor strings for resource layer.

        One anchor per (type, location) and per resource_name.

        Args:
            boundary: Canonical DesignBoundary

        Returns:
            List of anchor strings
        """
        anchors = []

        types = sorted(boundary.constraints.resource.types)
        locations = (
            sorted(boundary.constraints.resource.locations)
            if boundary.constraints.resource.locations
            else [None]
        )
        names = (
            sorted(boundary.constraints.resource.names)
            if boundary.constraints.resource.names
            else [None]
        )

        for rtype in types:
            for location in locations:
                for name in names:
                    anchors.append(
                        serialize_resource_slot(
                            rtype,
                            resource_name=name,
                            resource_location=location,
                        )
                    )

        return anchors

    def _extract_data_anchors(self, boundary: DesignBoundary) -> list[str]:
        """
        Extract anchor strings for data layer.

        One anchor per (sensitivity, pii, volume) combination.

        Args:
            boundary: Canonical DesignBoundary

        Returns:
            List of anchor strings
        """
        anchors = []

        sensitivities = sorted(boundary.constraints.data.sensitivity)
        pii_values = [boundary.constraints.data.pii] if boundary.constraints.data.pii is not None else [True, False]
        volumes = [boundary.constraints.data.volume] if boundary.constraints.data.volume else ["single", "bulk"]

        for sensitivity in sensitivities:
            for pii in pii_values:
                for volume in volumes:
                    anchors.append(
                        serialize_data_slot(
                            sensitivity,
                            pii=pii,
                            volume=volume,
                        )
                    )

        return anchors

    def _extract_risk_anchors(self, boundary: DesignBoundary) -> list[str]:
        """
        Extract anchor strings for risk layer.

        One anchor per authn requirement.

        Args:
            boundary: Canonical DesignBoundary

        Returns:
            List of anchor strings
        """
        authn = boundary.constraints.risk.authn
        return [serialize_risk_slot(authn)]

    def _encode_anchors(self, anchor_texts: list[str], layer_name: str) -> Tuple[np.ndarray, int]:
        """
        Encode list of anchors to padded array.

        Args:
            anchor_texts: List of anchor strings to encode
            layer_name: Name of layer (for logging and seed lookup)

        Returns:
            Tuple of (anchor_array, count) where:
            - anchor_array: (16, 32) array with encoded anchors (padded with zeros)
            - count: Actual number of anchors before padding
        """
        # Truncate if exceeds max
        if len(anchor_texts) > self.MAX_ANCHORS_PER_LAYER:
            logger.warning(
                f"Layer {layer_name} has {len(anchor_texts)} anchors, "
                f"truncating to {self.MAX_ANCHORS_PER_LAYER}"
            )
            anchor_texts = anchor_texts[: self.MAX_ANCHORS_PER_LAYER]

        # Encode each anchor
        anchor_vecs = []
        for text in anchor_texts:
            vec = self.encode_slot(text, layer_name)
            anchor_vecs.append(vec)

        # Pad to 16×32
        anchor_array = np.zeros((self.MAX_ANCHORS_PER_LAYER, 32), dtype=np.float32)
        for i, vec in enumerate(anchor_vecs):
            anchor_array[i] = vec

        return anchor_array, len(anchor_texts)

    def encode(self, boundary: DesignBoundary) -> RuleVector:
        """
        Encode canonical DesignBoundary to RuleVector.

        Steps:
        1. Extract anchors for each of 4 layers
        2. Encode each anchor to 32-dim
        3. Aggregate with padding to 16×32 per layer
        4. Stack to 4×16×32 RuleVector

        Args:
            boundary: Canonical DesignBoundary

        Returns:
            RuleVector with 4 layers of 16×32 anchor vectors
        """
        rule_vector = RuleVector()

        # Action layer
        action_anchors = self._extract_action_anchors(boundary)
        action_array, action_count = self._encode_anchors(action_anchors, "action")
        rule_vector.set_layer("action", action_array, action_count)

        # Resource layer
        resource_anchors = self._extract_resource_anchors(boundary)
        resource_array, resource_count = self._encode_anchors(resource_anchors, "resource")
        rule_vector.set_layer("resource", resource_array, resource_count)

        # Data layer
        data_anchors = self._extract_data_anchors(boundary)
        data_array, data_count = self._encode_anchors(data_anchors, "data")
        rule_vector.set_layer("data", data_array, data_count)

        # Risk layer
        risk_anchors = self._extract_risk_anchors(boundary)
        risk_array, risk_count = self._encode_anchors(risk_anchors, "risk")
        rule_vector.set_layer("risk", risk_array, risk_count)

        logger.debug(
            f"Encoded boundary {boundary.id}: "
            f"action={action_count}, resource={resource_count}, "
            f"data={data_count}, risk={risk_count}"
        )

        return rule_vector
