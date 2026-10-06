"""LLM access layer.

The pipeline only depends on the small `LLM` protocol below, so the same agent runs on:

* `NIMClient`: the OpenAI SDK pointed at a NIM endpoint (hosted build.nvidia.com or self-hosted).
  Uses NIM structured generation (`guided_json`) so outputs always parse.
* `LangChainAdapter`: any LangChain chat model, e.g. the one NeMo Agent Toolkit hands us in
  `builder.get_llm(...)`, so the toolkit's profiler sees every token.
* `FakeLLM` in tests.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from .config import Settings

Role = Literal["reasoning", "fast"]
T = TypeVar("T", bound=BaseModel)


@dataclass
class CallRecord:
    """One LLM call, kept for profiling (latency and tokens per pipeline step)."""

    step: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class LLMResult:
    text: str
    model: str
    latency_s: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None


class LLM(Protocol):
    calls: list[CallRecord]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        role: Role = "fast",
        step: str = "llm",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
    ) -> LLMResult: ...


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model response that may contain reasoning traces or code fences."""
    cleaned = _THINK_RE.sub("", text)
    if "</think>" in cleaned:  # reasoning trace without an opening tag
        cleaned = cleaned.rsplit("</think>", 1)[1]
    cleaned = cleaned.strip()
    fence = _FENCE_RE.search(cleaned)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(cleaned)
    for cand in candidates:
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    # Fall back to the first balanced JSON object or array in the text.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        while start != -1:
            depth, in_str, escape = 0, False, False
            for i in range(start, len(cleaned)):
                ch = cleaned[i]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[start : i + 1])
                        except json.JSONDecodeError:
                            break
            start = cleaned.find(opener, start + 1)
    raise ValueError(f"No JSON found in model output: {text[:200]!r}")


async def complete_structured(
    llm: LLM,
    messages: list[dict[str, str]],
    model_cls: type[T],
    *,
    role: Role = "fast",
    step: str = "llm",
    max_tokens: int | None = None,
    retries: int = 1,
) -> T:
    """Ask for JSON matching `model_cls`, validate it, and retry once with the validation error."""
    schema = model_cls.model_json_schema()
    msgs = list(messages)
    last_err: Exception | None = None
    for _ in range(retries + 1):
        result = await llm.chat(msgs, role=role, step=step, json_schema=schema, max_tokens=max_tokens)
        if result.finish_reason == "length":
            # Output was cut off: give the retry more room instead of asking it to repair broken JSON.
            last_err = RuntimeError("output truncated at max_tokens")
            max_tokens = (max_tokens or 4096) * 2
            continue
        try:
            return model_cls.model_validate(extract_json(result.text))
        except (ValueError, ValidationError) as err:
            last_err = err
            # Repeat the schema in the prompt, in case the endpoint ignored guided decoding.
            msgs = msgs + [
                {"role": "assistant", "content": result.text[:4000]},
                {
                    "role": "user",
                    "content": f"Your output was not valid. Error: {str(err)[:800]}\n"
                    "Return only valid JSON matching this JSON schema:\n" + json.dumps(schema),
                },
            ]
    raise RuntimeError(f"{step}: model did not return valid {model_cls.__name__}: {last_err}")


class NIMClient:
    """OpenAI-compatible client for NVIDIA NIM (hosted or self-hosted)."""

    def __init__(self, settings: Settings, http_client: Any | None = None):
        from openai import AsyncOpenAI

        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.base_url, api_key=settings.require_api_key(), http_client=http_client
        )
        self.calls: list[CallRecord] = []

    def model_for(self, role: Role) -> str:
        return self.settings.model_reasoning if role == "reasoning" else self.settings.model_fast

    async def chat(self, messages, *, role="fast", step="llm", json_schema=None, max_tokens=None) -> LLMResult:
        model = self.model_for(role)
        extra_body: dict[str, Any] = {
            # Nemotron 3 / 3.5 switch reasoning traces on or off through the chat template.
            "chat_template_kwargs": {"enable_thinking": bool(self.settings.thinking and role == "reasoning")}
        }
        if json_schema is not None:
            # NIM structured generation: decoding is constrained to the JSON schema.
            extra_body["guided_json"] = json_schema
        start = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=self.settings.temperature,
            top_p=0.95,
            max_tokens=max_tokens or self.settings.max_tokens,
            extra_body=extra_body,
        )
        latency = time.perf_counter() - start
        usage = resp.usage
        choice = resp.choices[0]
        text = choice.message.content or ""
        rec = CallRecord(
            step=step,
            model=model,
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )
        self.calls.append(rec)
        return LLMResult(text, model, latency, rec.prompt_tokens, rec.completion_tokens, choice.finish_reason)


class LangChainAdapter:
    """Wrap LangChain chat models (one per role) behind the `LLM` protocol."""

    def __init__(self, reasoning_model: Any, fast_model: Any | None = None):
        self.models = {"reasoning": reasoning_model, "fast": fast_model or reasoning_model}
        self.calls: list[CallRecord] = []

    async def chat(self, messages, *, role="fast", step="llm", json_schema=None, max_tokens=None) -> LLMResult:
        model = self.models[role]
        msgs = list(messages)
        if json_schema is not None:
            # No guided decoding through the generic interface, so state the schema explicitly.
            msgs = msgs + [
                {
                    "role": "user",
                    "content": "Respond with JSON only, matching this JSON schema:\n" + json.dumps(json_schema),
                }
            ]
        start = time.perf_counter()
        resp = await model.ainvoke(msgs)
        latency = time.perf_counter() - start
        usage = getattr(resp, "usage_metadata", None) or {}
        name = getattr(model, "model", None) or getattr(model, "model_name", None) or type(model).__name__
        rec = CallRecord(step, str(name), latency, usage.get("input_tokens"), usage.get("output_tokens"))
        self.calls.append(rec)
        content = resp.content if isinstance(resp.content, str) else json.dumps(resp.content)
        return LLMResult(content, str(name), latency, rec.prompt_tokens, rec.completion_tokens)


class Seq(list):
    """Marks a FakeLLM response as a sequence: one item per call, the last item repeats."""


@dataclass
class FakeLLM:
    """Deterministic stand-in for tests and offline demos.

    `responses` maps a step name to a JSON-serialisable object, a string, a `Seq` of them, or a
    callable that receives the messages and returns one of those.
    """

    responses: dict[str, Any]
    calls: list[CallRecord] = field(default_factory=list)

    async def chat(self, messages, *, role="fast", step="llm", json_schema=None, max_tokens=None) -> LLMResult:
        if step not in self.responses:
            raise KeyError(f"FakeLLM has no response for step {step!r}")
        value = self.responses[step]
        if isinstance(value, Seq):
            value = value.pop(0) if len(value) > 1 else value[0]
        if callable(value):
            value = value(messages)
        text = value if isinstance(value, str) else json.dumps(value)
        self.calls.append(CallRecord(step, f"fake-{role}", 0.0, 0, 0))
        return LLMResult(text, f"fake-{role}", 0.0, 0, 0)
