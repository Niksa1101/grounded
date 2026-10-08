"""The generation settings that change an answer (Tech.md §9.1, §11, PRD D47).

``GenerationParams`` is read from ``Settings`` once, in the pipeline. The pipeline passes
``temperature`` and ``max_output_tokens`` to every call, and the answer cache key hashes all of
them, so for those two the call and the key can never read different values. ``thinking_level`` is
the exception: the Gemini adapter reads ``GEMINI_THINKING_LEVEL`` itself when ``runtime.py`` builds
it, from the same ``Settings``, so the two agree within a process but are two readings (it moves to
the provider in Phase 4, PRD §12). Without these in the key, changing
``GEMINI_THINKING_LEVEL`` or ``LLM_TEMPERATURE`` would serve up to 30 days of answers made with the
old values (Phase 3 review #3).

The model ID is not here: it is a part of the key of its own (``generator_model``). Neither is the
timeout, which decides whether an answer arrives, not what it says. ``thinking_level`` is hashed
whatever the provider: only Gemini uses it today, and an unused value costs at most one round of
misses when it changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from grounded.settings import Settings


@dataclass(frozen=True, slots=True)
class GenerationParams:
    provider: str
    temperature: float
    max_output_tokens: int
    thinking_level: str

    @classmethod
    def from_settings(cls, settings: Settings, provider: str) -> GenerationParams:
        return cls(
            provider=provider,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_output_tokens,
            thinking_level=settings.gemini_thinking_level,
        )

    @property
    def config_hash(self) -> str:
        """sha256 hex of the key-sorted JSON, like ``RetrievalConfig.config_hash``."""
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def adapter_params(settings: Settings, provider: str) -> dict[str, str]:
    """The settings an adapter reads by itself, not as call arguments, that change what it says.

    The eval LLM cache key (Tech §11, 4.03) hashes these next to the call arguments, so a changed
    ``GEMINI_THINKING_LEVEL`` or ``GROQ_REASONING_EFFORT`` misses instead of replaying answers made
    with the old value. Only the knob the provider actually uses is listed: a Groq judge verdict
    costs scarce daily quota, and it must not be thrown away because a Gemini setting moved.
    ``runtime.py`` builds the adapters from the same ``Settings``, so the key and the call agree;
    the day each adapter exposes its own parameters (PRD §12), this function is what it replaces.
    """
    if provider == "gemini":
        return {"thinking_level": settings.gemini_thinking_level}
    if provider == "groq":
        return {"reasoning_effort": settings.groq_reasoning_effort}
    return {}  # the fake and stub providers have no such knob
