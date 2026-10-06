"""Thin streaming chat client for any OpenAI-compatible endpoint (hosted NIM, local NIM, vLLM, Dynamo).

Unlike the batch pipeline, the real-time path picks the model per call (the router decides) and
measures time-to-first-token (TTFT), which is what a clinician perceives as responsiveness.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any


@dataclass
class ChatResult:
    text: str
    model: str
    ttft_s: float | None
    total_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    chunks: int
    tokens_estimated: bool = False  # True when the server sent no usage and chunks were counted instead
    reasoning_chunks: int = 0


class StreamingChat:
    def __init__(self, base_url: str, api_key: str | None, http_client: Any | None = None):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or "not-needed", http_client=http_client)

    async def complete(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        json_schema: dict | None = None,
        thinking: bool = False,
        include_usage: bool = True,
    ) -> ChatResult:
        extra_body: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": thinking}}
        if json_schema is not None:
            extra_body["guided_json"] = json_schema
        kwargs: dict[str, Any] = {}
        if include_usage:
            kwargs["stream_options"] = {"include_usage": True}
        start = time.perf_counter()
        ttft = None
        parts: list[str] = []
        usage = None
        chunks = 0
        stream = await self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )
        reasoning_chunks = 0
        async for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if not chunk.choices:
                continue
            d = chunk.choices[0].delta
            delta = d.content or ""
            reasoning = getattr(d, "reasoning_content", None) or getattr(d, "reasoning", None) or ""
            if (delta or reasoning) and ttft is None:
                ttft = time.perf_counter() - start  # first token of any kind, including reasoning
            if reasoning:
                reasoning_chunks += 1
            if delta:
                parts.append(delta)
                chunks += 1
        total = time.perf_counter() - start
        reported = getattr(usage, "completion_tokens", None)
        return ChatResult(
            text="".join(parts),
            model=model,
            ttft_s=ttft,
            total_s=total,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=reported if reported is not None else ((chunks + reasoning_chunks) or None),
            chunks=chunks,
            tokens_estimated=reported is None,
            reasoning_chunks=reasoning_chunks,
        )
