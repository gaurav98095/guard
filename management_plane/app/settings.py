"""
Configuration management for Management Plane.

Loads configuration from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv


class Config:
    """Application configuration."""

    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent

    MGMT_ENV_PATH: Path = PROJECT_ROOT / "management_plane" / ".env"
    if MGMT_ENV_PATH.exists():
        load_dotenv(MGMT_ENV_PATH)

    DEMO_ENV_PATH: Path = PROJECT_ROOT / "examples" / "langgraph_demo" / ".env"
    if DEMO_ENV_PATH.exists():
        load_dotenv(DEMO_ENV_PATH)

    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    API_V2_PREFIX: str = "/api/v2"
    HOST: str = os.getenv("MGMT_PLANE_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("MGMT_PLANE_PORT", "8000"))

    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://localhost:8080",  # UI container (dev)
        "https://platform.tupl.xyz",  # Legacy production domain (keep for transition)
        "https://guard.fencio.dev",  # Guard Console
        "https://developer.fencio.dev",  # Developer Platform
    ]

    # Logging Configuration
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = os.getenv(
        "LOG_LEVEL", "INFO"
    )  # type: ignore

    # Application Metadata
    APP_NAME: str = "Management Plane"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "LLM Security Policy Enforcement - Management Plane"

    # Encoding Configuration (Week 2)
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_CACHE_SIZE: int = int(os.getenv("EMBEDDING_CACHE_SIZE", "10000"))

    # Database Configuration (Week 3)
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./mgmt_plane.db")

    # LLM/Api Configuration (Week 4)
    GOOGLE_API_KEY: str | None = os.getenv("GOOGLE_API_KEY")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

    # Data Plane Configuration
    data_plane_url: str = os.getenv("DATA_PLANE_URL", "localhost:50051")

    # Chroma Configuration
    CHROMA_URL: str = os.getenv("CHROMA_URL", "http://localhost:8002")
    CHROMA_COLLECTION_PREFIX: str = os.getenv("CHROMA_COLLECTION_PREFIX", "rules_")

    # Canonicalization Configuration (v2)
    CANONICALIZATION_ENABLED: bool = os.getenv("CANONICALIZATION_ENABLED", "true").lower() == "true"
    CANONICALIZATION_CONFIG_PATH: str | None = os.getenv("CANONICALIZATION_CONFIG_PATH")
    CANONICALIZATION_LOG_DIR: str = os.getenv(
        "CANONICALIZATION_LOG_DIR",
        str(PROJECT_ROOT / "data" / "logs"),
    )
    CANONICALIZATION_LOG_RETENTION_DAYS: int = int(os.getenv("CANONICALIZATION_LOG_RETENTION_DAYS", "90"))
    BERT_MODEL_PATH: str = os.getenv(
        "BERT_MODEL_PATH",
        "management_plane/models/canonicalizer_tinybert_v1.0/model_optimized.onnx",
    )
    BERT_TOKENIZER_PATH: str = os.getenv(
        "BERT_TOKENIZER_PATH",
        "management_plane/models/canonicalizer_tinybert_v1.0/tokenizer",
    )
    BERT_CONFIDENCE_HIGH: float = float(os.getenv("BERT_CONFIDENCE_HIGH", "0.9"))
    BERT_CONFIDENCE_MEDIUM: float = float(os.getenv("BERT_CONFIDENCE_MEDIUM", "0.7"))

    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration at startup.

        Raises:
            FileNotFoundError: If Rust library is not found.
        """
        if cls.CANONICALIZATION_ENABLED:
            model_path = Path(cls.BERT_MODEL_PATH)
            tokenizer_path = Path(cls.BERT_TOKENIZER_PATH)

            if not model_path.is_absolute():
                model_path = cls.PROJECT_ROOT / model_path

            if not tokenizer_path.is_absolute():
                tokenizer_path = cls.PROJECT_ROOT / tokenizer_path

            if not model_path.exists():
                raise FileNotFoundError(
                    f"BERT model not found at {model_path}. "
                    "Set BERT_MODEL_PATH to the model .onnx path."
                )

            if not tokenizer_path.exists():
                raise FileNotFoundError(
                    f"BERT tokenizer not found at {tokenizer_path}. "
                    "Set BERT_TOKENIZER_PATH to the tokenizer directory."
                )

            label_maps_path = model_path.parent / "label_maps.json"
            if not label_maps_path.exists():
                raise FileNotFoundError(
                    f"BERT label maps not found at {label_maps_path}. "
                    "Expected label_maps.json alongside the ONNX model."
                )

    @classmethod
    def get_google_api_key(cls) -> str:
        """
        Fail fast when the Google API key is missing.

        Raises:
            ValueError: If GOOGLE_API_KEY is not configured.
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in management_plane/.env, "
                "examples/langgraph_demo/.env, or the environment."
            )
        return cls.GOOGLE_API_KEY


# Global config instance
config = Config()
