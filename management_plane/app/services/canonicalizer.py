"""
BERT-based Canonicalization Service for Semantic Security.

Converts variable vocabulary terms (actions, resource types, sensitivity levels) 
to canonical forms using TinyBERT ONNX model with multi-head classification.

Architecture:
- Load TinyBERT ONNX model from models/canonicalizer_tinybert_v1.0/
- Implement multi-head classification for 3 semantic dimensions:
  * Action (6 classes: read, write, update, delete, execute, export)
  * Resource Type (5 classes: database, storage, api, queue, cache)
  * Sensitivity (3 classes: public, internal, secret)
- Confidence thresholds: high (≥0.9), medium (0.7-0.9), low (<0.7)
- Fallback: Unknown terms pass through unchanged with confidence=0.0

Performance Target: <10ms per prediction (vectorized batch processing)
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as rt
from transformers import AutoTokenizer

from app.models import DesignBoundary, IntentEvent, LooseDesignBoundary, LooseIntentEvent

logger = logging.getLogger(__name__)


class CanonicalizedField:
    """Result of canonicalizing a single field."""

    def __init__(
        self,
        field_name: str,
        raw_value: str,
        canonical_value: str,
        confidence: float,
        source: str = "passthrough",
    ):
        """
        Initialize canonicalized field.

        Args:
            field_name: Name of the field (e.g., 'action', 'resource_type')
            raw_value: Original value from input
            canonical_value: Canonicalized value or raw_value if passthrough
            confidence: Confidence score [0.0, 1.0]
            source: "bert", "passthrough", or "default"
        """
        self.field_name = field_name
        self.raw_value = raw_value
        self.canonical_value = canonical_value
        self.confidence = confidence
        self.source = source

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/JSON."""
        return {
            "field": self.field_name,
            "raw_input": self.raw_value,
            "prediction": {
                "canonical": self.canonical_value,
                "confidence": float(self.confidence),
                "source": self.source,
            },
        }


class CanonicalizedEvent:
    """Result of canonicalizing an entire IntentEvent."""

    def __init__(self, event: IntentEvent, canonical_event: IntentEvent, trace: list[CanonicalizedField]):
        """
        Initialize canonicalized event.

        Args:
            event: Original IntentEvent
            canonical_event: Canonicalized IntentEvent with mapped terms
            trace: List of CanonicalizedField for each normalized term
        """
        self.event = event
        self.canonical_event = canonical_event
        self.trace = trace

    def to_trace_dict(self) -> dict:
        """Return trace as dictionary for response metadata."""
        return {
            "canonicalization_trace": [field.to_dict() for field in self.trace],
        }


class CanonicalizedBoundary:
    """Result of canonicalizing a DesignBoundary."""

    def __init__(self, boundary: DesignBoundary, canonical_boundary: DesignBoundary, trace: list[CanonicalizedField]):
        """
        Initialize canonicalized boundary.

        Args:
            boundary: Original DesignBoundary
            canonical_boundary: Canonicalized DesignBoundary with mapped terms
            trace: List of CanonicalizedField for each normalized term
        """
        self.boundary = boundary
        self.canonical_boundary = canonical_boundary
        self.trace = trace

    def to_trace_dict(self) -> dict:
        """Return trace as dictionary for response metadata."""
        return {
            "canonicalization_trace": [field.to_dict() for field in self.trace],
        }


class BertCanonicalizer:
    """
    BERT-based canonicalizer for semantic security terms.

    Multi-head classifier that maps free-form terms to canonical vocabulary
    using TinyBERT ONNX model with per-head confidence thresholds.
    """

    def __init__(
        self,
        model_dir: Path | None = None,
        model_path: Path | None = None,
        tokenizer_path: Path | None = None,
        label_maps_path: Path | None = None,
        confidence_high: float = 0.9,
        confidence_medium: float = 0.7,
    ):
        """
        Initialize BERT canonicalizer.

        Args:
            model_dir: Path to canonicalizer_tinybert_v1.0 directory
            confidence_high: Threshold for high confidence predictions (default 0.9)
            confidence_medium: Threshold for medium confidence (default 0.7)

        Raises:
            FileNotFoundError: If model files not found
            RuntimeError: If ONNX model fails to load
        """
        if model_dir is None and model_path is None:
            raise ValueError("model_dir or model_path must be provided")

        if model_dir is None and model_path is not None:
            model_dir = model_path.parent

        if model_dir is None:
            raise ValueError("model_dir could not be inferred from model_path")

        self.model_dir = Path(model_dir)
        self.confidence_high = confidence_high
        self.confidence_medium = confidence_medium

        resolved_model_path = model_path or self.model_dir / "model.onnx"
        resolved_tokenizer_path = tokenizer_path or self.model_dir / "tokenizer"
        resolved_label_maps_path = label_maps_path or self.model_dir / "label_maps.json"

        if not resolved_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {resolved_model_path}")

        if not resolved_tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {resolved_tokenizer_path}")

        if not resolved_label_maps_path.exists():
            raise FileNotFoundError(f"Label maps not found at {resolved_label_maps_path}")

        # Load label maps
        with open(resolved_label_maps_path) as f:
            label_maps = json.load(f)

        self.action_labels = {v: k for k, v in label_maps["action"].items()}
        self.resource_labels = {v: k for k, v in label_maps["resource_type"].items()}
        self.sensitivity_labels = {v: k for k, v in label_maps["sensitivity"].items()}

        # Load tokenizer
        logger.info(f"Loading tokenizer from {resolved_tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(str(resolved_tokenizer_path))

        # Load ONNX model
        logger.info(f"Loading ONNX model from {resolved_model_path}")
        try:
            self.session = rt.InferenceSession(
                str(resolved_model_path),
                providers=["CPUExecutionProvider"],
            )
            logger.info("ONNX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise RuntimeError(f"ONNX model loading failed: {e}")

        # Get input/output names
        input_name = self.session.get_inputs()[0].name
        output_names = [output.name for output in self.session.get_outputs()]

        self.input_name = input_name
        self.output_names = output_names

        logger.info(f"Model loaded: inputs={input_name}, outputs={output_names}")

    def _classify_text(self, text: str) -> dict[str, dict]:
        """
        Classify text using BERT model for all three heads.

        Args:
            text: Input text to classify

        Returns:
            Dictionary with predictions for each head:
            {
                "action": {"prediction": str, "confidence": float, "raw_scores": np.ndarray},
                "resource_type": {"prediction": str, "confidence": float, "raw_scores": np.ndarray},
                "sensitivity": {"prediction": str, "confidence": float, "raw_scores": np.ndarray},
            }
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128,
        )

        # Inference
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: input_ids, "attention_mask": attention_mask},
        )

        # outputs should be [action_scores, resource_scores, sensitivity_scores]
        # each of shape (1, num_classes)

        action_scores = outputs[0][0]  # (6,)
        resource_scores = outputs[1][0]  # (5,)
        sensitivity_scores = outputs[2][0]  # (3,)

        # Apply softmax to convert logits to probabilities
        def softmax(x):
            exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
            return exp_x / exp_x.sum()

        action_probs = softmax(action_scores)
        resource_probs = softmax(resource_scores)
        sensitivity_probs = softmax(sensitivity_scores)

        # Get argmax predictions
        action_idx = np.argmax(action_probs)
        resource_idx = np.argmax(resource_probs)
        sensitivity_idx = np.argmax(sensitivity_probs)

        return {
            "action": {
                "prediction": self.action_labels[int(action_idx)],
                "confidence": float(action_probs[action_idx]),
                "raw_scores": action_probs,
            },
            "resource_type": {
                "prediction": self.resource_labels[int(resource_idx)],
                "confidence": float(resource_probs[resource_idx]),
                "raw_scores": resource_probs,
            },
            "sensitivity": {
                "prediction": self.sensitivity_labels[int(sensitivity_idx)],
                "confidence": float(sensitivity_probs[sensitivity_idx]),
                "raw_scores": sensitivity_probs,
            },
        }

    def canonicalize_field(
        self,
        field_name: str,
        raw_value: str,
    ) -> CanonicalizedField:
        """
        Canonicalize a single field value.

        Args:
            field_name: Name of field ("action", "resource_type", or "sensitivity")
            raw_value: Raw value to canonicalize

        Returns:
            CanonicalizedField with canonical term and confidence
        """
        try:
            # Handle None/empty values
            if not raw_value:
                return CanonicalizedField(
                    field_name=field_name,
                    raw_value=raw_value or "",
                    canonical_value=raw_value or "",
                    confidence=0.0,
                    source="passthrough",
                )

            # Classify using BERT
            start_time = time.time()
            predictions = self._classify_text(raw_value)
            inference_time = (time.time() - start_time) * 1000

            if field_name not in predictions:
                logger.warning(f"Field {field_name} not in BERT output")
                return CanonicalizedField(
                    field_name=field_name,
                    raw_value=raw_value,
                    canonical_value=raw_value,
                    confidence=0.0,
                    source="passthrough",
                )

            pred = predictions[field_name]
            confidence = pred["confidence"]
            canonical = pred["prediction"]

            # Determine source based on confidence
            if confidence >= self.confidence_high:
                source = "bert_high"
            elif confidence >= self.confidence_medium:
                source = "bert_medium"
            else:
                # Passthrough for low confidence
                source = "passthrough"
                canonical = raw_value

            logger.debug(
                f"Canonicalized {field_name}='{raw_value}' -> '{canonical}' "
                f"(conf={confidence:.3f}, source={source}, latency={inference_time:.1f}ms)"
            )

            return CanonicalizedField(
                field_name=field_name,
                raw_value=raw_value,
                canonical_value=canonical,
                confidence=confidence,
                source=source,
            )

        except Exception as e:
            logger.error(f"Error canonicalizing {field_name}='{raw_value}': {e}")
            # Fail-safe: passthrough
            return CanonicalizedField(
                field_name=field_name,
                raw_value=raw_value,
                canonical_value=raw_value,
                confidence=0.0,
                source="error",
            )

    def canonicalize(self, event: IntentEvent | LooseIntentEvent) -> CanonicalizedEvent:
        """
        Canonicalize all terms in an IntentEvent.

        Args:
            event: IntentEvent with variable vocabulary

        Returns:
            CanonicalizedEvent with canonical terms and full trace
        """
        trace = []

        # Canonicalize action
        action_field = self.canonicalize_field("action", event.action)
        trace.append(action_field)

        # Canonicalize resource_type
        resource_field = self.canonicalize_field("resource_type", event.resource.type)
        trace.append(resource_field)

        # Canonicalize sensitivity
        sensitivity = event.data.sensitivity[0] if event.data.sensitivity else "public"
        sensitivity_field = self.canonicalize_field("sensitivity", sensitivity)
        trace.append(sensitivity_field)

        # Create canonical event by copying and updating fields
        canonical_event = event.model_copy(deep=True)
        canonical_event.action = action_field.canonical_value
        canonical_event.resource.type = resource_field.canonical_value  # type: ignore
        if canonical_event.data.sensitivity:
            canonical_event.data.sensitivity[0] = sensitivity_field.canonical_value

        return CanonicalizedEvent(event, canonical_event, trace)

    def canonicalize_boundary(self, boundary: DesignBoundary | LooseDesignBoundary) -> CanonicalizedBoundary:
        """
        Canonicalize all terms in a DesignBoundary.

        Args:
            boundary: DesignBoundary policy with variable vocabulary

        Returns:
            CanonicalizedBoundary with canonical terms and full trace
        """
        trace = []

        # Canonicalize action constraints
        canonical_actions = set()
        for action in boundary.constraints.action.actions:
            action_field = self.canonicalize_field("action", action)
            trace.append(action_field)
            canonical_actions.add(action_field.canonical_value)

        # Canonicalize resource_type constraints
        canonical_resource_types = set()
        for resource_type in boundary.constraints.resource.types:
            resource_field = self.canonicalize_field("resource_type", resource_type)
            trace.append(resource_field)
            canonical_resource_types.add(resource_field.canonical_value)

        # Canonicalize sensitivity constraints
        canonical_sensitivities = set()
        for sensitivity in boundary.constraints.data.sensitivity:
            sensitivity_field = self.canonicalize_field("sensitivity", sensitivity)
            trace.append(sensitivity_field)
            canonical_sensitivities.add(sensitivity_field.canonical_value)

        # Create canonical boundary by copying and updating
        canonical_boundary = boundary.model_copy(deep=True)
        canonical_boundary.constraints.action.actions = sorted(canonical_actions)
        canonical_boundary.constraints.resource.types = sorted(canonical_resource_types)
        canonical_boundary.constraints.data.sensitivity = sorted(canonical_sensitivities)

        return CanonicalizedBoundary(boundary, canonical_boundary, trace)
