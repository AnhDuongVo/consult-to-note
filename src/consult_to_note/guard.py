"""NeMo Guardrails integration: PHI masking on the way in, a safety self-check on the way out.

Uses `LLMRails.check_async` (NeMo Guardrails >= 0.22), which runs only the input or output rails on a
piece of text, so the rails act as a filter around our own pipeline instead of wrapping a chat bot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .config import HOSTED_NIM_URL, get_settings
from .schemas import Transcript, Utterance

CONFIG_DIR = Path(__file__).parent / "guardrails_config"
_LINE_RE = re.compile(r"^\[U(?P<id>\d+)\]\s+(?P<spk>doctor|patient|other):\s?(?P<text>.*)$")


@dataclass
class OutputCheck:
    allowed: bool
    content: str
    rail: str | None = None


class MaskingError(RuntimeError):
    """Raised when masked output cannot be mapped back to every utterance (fail closed)."""


@lru_cache(maxsize=1)
def _rails(config_dir: str = str(CONFIG_DIR)):
    try:
        from nemoguardrails import LLMRails, RailsConfig
    except ImportError as err:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the guardrails extra: pip install -e '.[guardrails]'") from err
    config = RailsConfig.from_path(config_dir)
    _apply_settings(config)
    return LLMRails(config)


def _apply_settings(config) -> None:
    """Point the rails' model at the same endpoint and fast model as the rest of the pipeline."""
    settings = get_settings()
    for model in config.models:
        if model.type != "main":
            continue
        model.model = settings.model_fast
        params = dict(model.parameters or {})
        if not settings.base_url.startswith(HOSTED_NIM_URL):
            params["base_url"] = settings.base_url
        model.parameters = params


async def mask_transcript(transcript: Transcript) -> Transcript:
    """Mask personal identifiers in every utterance, keeping utterance ids and speakers intact."""
    from nemoguardrails.rails.llm.options import RailType

    rails = _rails()
    result = await rails.check_async([{"role": "user", "content": transcript.to_prompt()}], rail_types=[RailType.INPUT])
    masked: dict[int, str] = {}
    for line in result.content.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            masked[int(m.group("id"))] = m.group("text")
    missing = [u.id for u in transcript.utterances if u.id not in masked]
    if missing:
        # Fail closed: never send an utterance onwards that may not have been masked.
        raise MaskingError(f"Masked output lost utterances {missing[:5]}; refusing to continue unmasked.")
    utterances = [Utterance(**{**u.model_dump(), "text": masked[u.id]}) for u in transcript.utterances]
    return Transcript(utterances=utterances, source=transcript.source, language=transcript.language)


async def check_note_output(note_markdown: str) -> OutputCheck:
    from nemoguardrails.rails.llm.options import RailStatus, RailType

    rails = _rails()
    result = await rails.check_async([{"role": "assistant", "content": note_markdown}], rail_types=[RailType.OUTPUT])
    return OutputCheck(allowed=result.status != RailStatus.BLOCKED, content=result.content, rail=result.rail)
