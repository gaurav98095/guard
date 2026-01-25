"""
LLM-based anchor generation for Data Plane rules.

Uses Google GenAI (Gemini 2.5 Flash Lite) to produce structured anchor
descriptions for every slot, then caches the result by content hash.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import yaml
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from app.settings import Config
from app.vocab import VOCABULARY

logger = logging.getLogger(__name__)


class ActionSlotFields(BaseModel):
    """Vocabulary-compliant field values for an action anchor."""

    action: str
    actor_type: str
    tool_call: str | None = None


class ResourceSlotFields(BaseModel):
    """Vocabulary-compliant field values for a resource anchor."""

    resource_type: str
    resource_location: str | None = None
    resource_name: str | None = None
    tool_name: str | None = None
    tool_method: str | None = None


class DataSlotFields(BaseModel):
    """Vocabulary-compliant field values for a data anchor."""

    sensitivity: str
    pii: bool
    volume: str
    params_length: str | None = None


class RiskSlotFields(BaseModel):
    """Vocabulary-compliant field values for a risk anchor."""

    authn: str
    rate_limit: str | None = None


class VocabGroundedAnchors(BaseModel):
    """Structured schema for vocabulary-grounded anchor field values."""

    action: list[ActionSlotFields]
    resource: list[ResourceSlotFields]
    data: list[DataSlotFields]
    risk: list[RiskSlotFields]


class AnchorSlots(BaseModel):
    """Structured schema for slot-level anchors returned by Gemini."""

    action: list[str]
    resource: list[str]
    data: list[str]
    risk: list[str]

class LLMAnchorGenerator:
    """
    Unified LLM anchor generator backed by Google GenAI.

    Encodes rules for every family by asking Gemini 2.5 Flash Lite to
    return natural language descriptions for the action/resource/data/risk
    slots.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._cache: dict[str, AnchorSlots] = {}
        self.vocab = VOCABULARY

    def _compute_cache_key(self, rule: dict[str, Any], family_id: str) -> str:
        """Deterministic hash for rule+family to support caching."""
        payload = json.dumps({"rule": rule, "family": family_id}, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def generate_rule_anchors(
        self,
        rule: dict[str, Any],
        family_id: str
    ) -> AnchorSlots:
        """Generate anchors asynchronously (runs blocking GenAI call in a thread)."""
        cache_key = self._compute_cache_key(rule, family_id)

        if cache_key in self._cache:
            logger.debug("LLM cache hit for %s:%s", family_id, rule.get("rule_id"))
            return self._cache[cache_key]

        anchors = await asyncio.to_thread(
            self._generate_sync,
            rule,
            family_id,
            cache_key,
        )

        self._cache[cache_key] = anchors
        return anchors

    def _generate_sync(
        self,
        rule: dict[str, Any],
        family_id: str,
        cache_key: str,
    ) -> AnchorSlots:
        """Blocking path that calls the Gemini API."""
        logger.info("Calling Gemini for %s rule anchors (cache key %s)", family_id, cache_key)
        extraction_rules = self.vocab.get_extraction_rules(family_id)
        examples = self.vocab.get_examples(family_id)

        prompt = (
            "You are a semantic security expert crafting vocabulary-aligned anchors.\n\n"
            "Rule family: {family}\n"
            "{rule_data}\n\n"
            "Canonical vocabulary: actions={actions}, resource_types={resources}, "
            "sensitivity={sensitivities}, volumes={volumes}, authn={authn}, params_length={params}\n\n"
            "Extraction rules:\n{extraction}\n"
            "Example expected output:\n{examples}\n\n"
            "Generate 2-4 field combinations per slot (action, resource, data, risk) "
            "using ONLY the canonical vocabulary values shown above. "
            "Return JSON matching the VocabGroundedAnchors schema, without assembling "
            "the anchor strings (the code will handle that)."
        ).format(
            family=family_id,
            rule_data=json.dumps(rule, indent=2),
            actions=self.vocab.get_valid_actions(),
            resources=self.vocab.get_valid_resource_types(),
            sensitivities=self.vocab.get_sensitivity_levels(),
            volumes=self.vocab.get_volumes(),
            authn=self.vocab.get_authn_levels(),
            params=self.vocab.get_params_length_buckets(),
            extraction=yaml.dump(extraction_rules, sort_keys=False),
            examples=yaml.dump(examples, sort_keys=False),
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VocabGroundedAnchors,
                    temperature=0.3,
                ),
            )

            llm_output = VocabGroundedAnchors.model_validate_json(response.text)
            action_anchors = [
                self.vocab.assemble_anchor("action", fields.model_dump(exclude_none=True))
                for fields in llm_output.action
            ]
            resource_anchors = [
                self.vocab.assemble_anchor("resource", fields.model_dump(exclude_none=True))
                for fields in llm_output.resource
            ]
            data_anchors = [
                self.vocab.assemble_anchor("data", fields.model_dump(exclude_none=True))
                for fields in llm_output.data
            ]
            risk_anchors = [
                self.vocab.assemble_anchor("risk", fields.model_dump(exclude_none=True))
                for fields in llm_output.risk
            ]

            anchors = AnchorSlots(
                action=action_anchors,
                resource=resource_anchors,
                data=data_anchors,
                risk=risk_anchors,
            )
            logger.debug(
                "Gemini produced %d/%d/%d/%d anchors for %s after assembling vocabulary templates",
                len(action_anchors),
                len(resource_anchors),
                len(data_anchors),
                len(risk_anchors),
                family_id,
            )
            return anchors

        except Exception as exc:
            logger.error("Failed to generate anchors for %s: %s", family_id, exc, exc_info=True)
            raise ValueError(
                f"LLM anchor generation failed for {family_id}: {exc}"
            ) from exc


_generator: LLMAnchorGenerator | None = None


def get_llm_generator() -> LLMAnchorGenerator:
    """Singleton accessor for the LLM anchor generator."""
    global _generator
    if _generator is None:
        api_key = Config.get_google_api_key()
        _generator = LLMAnchorGenerator(api_key=api_key, model=Config.GEMINI_MODEL)
    return _generator
