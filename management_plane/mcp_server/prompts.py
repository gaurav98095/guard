from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .app import mcp


_PROMPT_DIR = Path(__file__).parent / "prompts"


def _load_prompt(prompt_name: str) -> str:
    prompt_path = _PROMPT_DIR / prompt_name
    with prompt_path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = yaml.safe_load(handle) or {}

    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError(f"Prompt {prompt_name} is missing 'prompt' content")

    return prompt


@mcp.prompt()
def governed_agent_instructions() -> str:
    return _load_prompt("governed_agent_instructions.yaml")
