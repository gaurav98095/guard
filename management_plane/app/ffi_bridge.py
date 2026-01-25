"""
FFI bridge to Rust semantic sandbox library.

This module provides a Python wrapper for the Rust CDylib that performs
vector comparison. It handles:
- Loading the shared library
- Defining ctypes structures matching Rust #[repr(C)]
- Providing a clean Python API for comparison operations
"""

import ctypes
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# FFI structure definitions matching Rust #[repr(C)] in semantic-sandbox/src/lib.rs
class VectorEnvelope(ctypes.Structure):
    """
    FFI structure for passing vector comparison data to Rust.

    Fields match exactly the Rust VectorEnvelope struct.
    """

    _fields_ = [
        ("intent", ctypes.c_float * 128),

        # Anchor arrays (4 slots × 16 anchors × 32 dims)
        ("action_anchors", (ctypes.c_float * 32) * 16),
        ("action_anchor_count", ctypes.c_size_t),
        ("resource_anchors", (ctypes.c_float * 32) * 16),
        ("resource_anchor_count", ctypes.c_size_t),
        ("data_anchors", (ctypes.c_float * 32) * 16),
        ("data_anchor_count", ctypes.c_size_t),
        ("risk_anchors", (ctypes.c_float * 32) * 16),
        ("risk_anchor_count", ctypes.c_size_t),

        ("thresholds", ctypes.c_float * 4),
        ("weights", ctypes.c_float * 4),
        ("decision_mode", ctypes.c_uint8),
        ("global_threshold", ctypes.c_float),
    ]


class ComparisonResultFFI(ctypes.Structure):
    """
    FFI structure for receiving comparison results from Rust.

    Fields match exactly the Rust ComparisonResult struct.
    """

    _fields_ = [
        ("decision", ctypes.c_uint8),
        ("slice_similarities", ctypes.c_float * 4),
    ]


class SemanticSandbox:
    """
    Python wrapper for Rust semantic sandbox library.

    Provides safe, idiomatic Python API for vector comparison operations.
    """

    def __init__(self, lib_path: Optional[Path] = None):
        """
        Initialize the FFI bridge and load the Rust library.

        Args:
            lib_path: Path to libsemantic_sandbox shared library.
                     If None, uses config.SEMANTIC_SANDBOX_LIB.

        Raises:
            FileNotFoundError: If library file doesn't exist.
            OSError: If library fails to load.
        """
        if lib_path is None:
            raise RuntimeError(
                "Semantic sandbox has been removed from the v2 stack. "
                "Use the data plane gRPC enforcement path instead."
            )

        self.lib_path = lib_path

        if not self.lib_path.exists():
            raise FileNotFoundError(
                f"Rust library not found at {self.lib_path}."
            )

        logger.info(f"Loading Rust library from {self.lib_path}")
        self.lib = ctypes.CDLL(str(self.lib_path))

        # Configure function signatures
        self._configure_functions()

        # Validate library is working
        self._validate_library()

        logger.info("Rust library loaded successfully")

    def _configure_functions(self) -> None:
        """Configure ctypes function signatures for all FFI functions."""
        # Health check function
        self.lib.health_check.restype = ctypes.c_uint8
        self.lib.health_check.argtypes = []

        # Version function
        self.lib.get_version.restype = ctypes.c_uint32
        self.lib.get_version.argtypes = []

        # Main comparison function
        self.lib.compare_vectors.restype = ComparisonResultFFI
        self.lib.compare_vectors.argtypes = [ctypes.POINTER(VectorEnvelope)]

    def _validate_library(self) -> None:
        """
        Validate the library is working correctly.

        Raises:
            RuntimeError: If library health check fails.
        """
        health = self.lib.health_check()
        if health != 1:
            raise RuntimeError(f"Library health check failed: {health}")

        version = self.lib.get_version()
        logger.debug(f"Library version: {version}")

    def compare(
        self,
        intent_vector: np.ndarray,
        action_anchors: np.ndarray,
        action_anchor_count: int,
        resource_anchors: np.ndarray,
        resource_anchor_count: int,
        data_anchors: np.ndarray,
        data_anchor_count: int,
        risk_anchors: np.ndarray,
        risk_anchor_count: int,
        thresholds: list[float],
        weights: list[float],
        decision_mode: int,
        global_threshold: float,
    ) -> tuple[int, list[float]]:
        """
        Compare an intent vector against boundary anchor sets.

        Args:
            intent_vector: 128-dimensional intent vector (4 slots × 32 dims)
            action_anchors: (max_anchors, 32) array of action slot anchors
            action_anchor_count: Number of valid action anchors
            resource_anchors: (max_anchors, 32) array of resource slot anchors
            resource_anchor_count: Number of valid resource anchors
            data_anchors: (max_anchors, 32) array of data slot anchors
            data_anchor_count: Number of valid data anchors
            risk_anchors: (max_anchors, 32) array of risk slot anchors
            risk_anchor_count: Number of valid risk anchors
            thresholds: Per-slice thresholds [action, resource, data, risk]
            weights: Per-slice weights [action, resource, data, risk]
            decision_mode: 0=min (all must pass), 1=weighted (average)
            global_threshold: Global threshold for weighted mode

        Returns:
            Tuple of (decision, slice_similarities) where:
                - decision: 0=block, 1=allow
                - slice_similarities: [action_sim, resource_sim, data_sim, risk_sim]

        Raises:
            ValueError: If input dimensions are incorrect.
        """
        # Validate inputs
        if intent_vector.shape != (128,):
            raise ValueError(f"Intent vector must be 128-dim, got {intent_vector.shape}")
        if action_anchors.shape[1] != 32:
            raise ValueError(f"Action anchors must be (N, 32), got {action_anchors.shape}")
        if resource_anchors.shape[1] != 32:
            raise ValueError(f"Resource anchors must be (N, 32), got {resource_anchors.shape}")
        if data_anchors.shape[1] != 32:
            raise ValueError(f"Data anchors must be (N, 32), got {data_anchors.shape}")
        if risk_anchors.shape[1] != 32:
            raise ValueError(f"Risk anchors must be (N, 32), got {risk_anchors.shape}")
        if len(thresholds) != 4:
            raise ValueError(f"Thresholds must be length 4, got {len(thresholds)}")
        if len(weights) != 4:
            raise ValueError(f"Weights must be length 4, got {len(weights)}")

        # Create envelope
        envelope = VectorEnvelope()

        # Copy intent vector (ensure float32)
        intent_f32 = intent_vector.astype(np.float32)
        for i in range(128):
            envelope.intent[i] = float(intent_f32[i])

        # Copy anchor arrays
        action_f32 = action_anchors.astype(np.float32)
        resource_f32 = resource_anchors.astype(np.float32)
        data_f32 = data_anchors.astype(np.float32)
        risk_f32 = risk_anchors.astype(np.float32)

        for i in range(min(action_anchor_count, 16)):
            for j in range(32):
                envelope.action_anchors[i][j] = float(action_f32[i, j])
        envelope.action_anchor_count = action_anchor_count

        for i in range(min(resource_anchor_count, 16)):
            for j in range(32):
                envelope.resource_anchors[i][j] = float(resource_f32[i, j])
        envelope.resource_anchor_count = resource_anchor_count

        for i in range(min(data_anchor_count, 16)):
            for j in range(32):
                envelope.data_anchors[i][j] = float(data_f32[i, j])
        envelope.data_anchor_count = data_anchor_count

        for i in range(min(risk_anchor_count, 16)):
            for j in range(32):
                envelope.risk_anchors[i][j] = float(risk_f32[i, j])
        envelope.risk_anchor_count = risk_anchor_count

        # Set thresholds and weights
        envelope.thresholds[:] = thresholds
        envelope.weights[:] = weights
        envelope.decision_mode = decision_mode
        envelope.global_threshold = global_threshold

        # Call Rust function
        result = self.lib.compare_vectors(ctypes.byref(envelope))

        # Convert result to Python types
        decision = int(result.decision)
        similarities = [float(result.slice_similarities[i]) for i in range(4)]

        logger.debug(
            f"Comparison: decision={decision}, similarities={similarities}"
        )

        return decision, similarities

    def health_check(self) -> bool:
        """
        Check if library is healthy.

        Returns:
            True if library is healthy, False otherwise.
        """
        try:
            return self.lib.health_check() == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


# Global singleton instance
_sandbox: Optional[SemanticSandbox] = None


def get_sandbox() -> SemanticSandbox:
    """
    Get or create the global SemanticSandbox instance.

    Returns:
        Singleton SemanticSandbox instance.
    """
    raise RuntimeError(
        "Semantic sandbox has been removed from the v2 stack. "
        "Use the data plane gRPC enforcement path instead."
    )
