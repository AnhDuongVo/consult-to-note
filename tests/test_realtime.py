import asyncio
import json

import httpx

from consult_to_note.realtime.bench.benchmark import (
    BenchConfig,
    RequestResult,
    Target,
    build_prompts,
    run_benchmark,
    summarize,
)
from consult_to_note.realtime.chat import ChatResult
from consult_to_note.realtime.live_scribe import LiveScribe, NotePatch, PatchOp, apply_patch
from consult_to_note.realtime.router import Candidate, LatencyRouter
from consult_to_note.realtime.streaming_asr import simulate_from_transcript
from consult_to_note.schemas import NoteSentence, SOAPNote, Transcript


def test_router_prefers_quality_within_budget_and_adapts():
    r = LatencyRouter(
        [Candidate("big", prior_latency_s=0.5), Candidate("small", prior_latency_s=0.2)], budget_s=1.0, explore_every=0
    )
    assert r.choose().model == "big"
    for _ in range(5):
        r.observe("big", 2.0)  # big model slows down under load
    d = r.choose()
    assert d.model == "small" and "fits budget" in d.reason
    for _ in range(6):
        r.observe("small", 1.5)  # both over budget now; small is still the faster one
    d = r.choose()
    assert d.model == "small" and "over budget" in d.reason

    r.observe_failure("small")  # a failing model must look slower, not faster
    assert r.expected("small") > 1.5


def test_apply_patch_validates_ops():
    note = SOAPNote(subjective=[NoteSentence(text="Knee pain.", evidence=[1])], objective=[], assessment=[], plan=[])
    patch = NotePatch(
        ops=[
            PatchOp(op="add", section="plan", text="Ice and rest.", evidence=[2]),
            PatchOp(op="add", section="plan", text="Invented.", evidence=[99]),  # unknown utterance: dropped
            PatchOp(op="replace", section="subjective", index=0, text="Right knee pain.", evidence=[1]),
            PatchOp(op="remove", section="objective", index=3),  # bad index: dropped
        ]
    )
    new, applied = apply_patch(note, patch, {1, 2})
    assert applied == 2
    assert new.subjective[0].text == "Right knee pain." and new.plan[0].text == "Ice and rest."


class FakeChat:
    """Returns one 'add' op per update citing the newest utterance; simulates 50 ms model latency."""

    def __init__(self):
        self.calls = 0

    async def complete(self, model, messages, **kw):
        self.calls += 1
        await asyncio.sleep(0.05)
        content = messages[0]["content"]
        newest = max(int(x) for x in __import__("re").findall(r"\[U(\d+)\]", content))
        text = content.split(f"[U{newest}]")[1].split(":", 1)[1].strip().splitlines()[0]
        patch = {"ops": [{"op": "add", "section": "subjective", "text": text, "evidence": [newest]}]}
        return ChatResult(json.dumps(patch), model, 0.01, 0.05, 100, 20, 5)


async def test_live_scribe_batches_and_meets_budget():
    t = Transcript.from_text("\n".join(f"[patient] symptom number {i} started today" for i in range(6)))
    scribe = LiveScribe(FakeChat(), LatencyRouter([Candidate("m", prior_latency_s=0.05)], budget_s=1.0))
    report = await scribe.run(simulate_from_transcript(t, speed=200.0))
    s = report.summary()
    assert s["utterances"] == 6
    assert len(report.note.subjective) >= 1
    assert s["within_budget_share"] == 1.0
    assert sum(m.batch_size for m in report.updates) == 6  # every utterance reached the note exactly once


async def test_benchmark_against_mock_stream(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["stream"] is True
        chunks = []
        for w in ["{", '"ops"', ": []}"]:
            chunks.append(
                {
                    "id": "x",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {"content": w}, "finish_reason": None}],
                }
            )
        chunks.append(
            {
                "id": "x",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": body["model"],
                "choices": [],
                "usage": {"prompt_tokens": 50, "completion_tokens": 3, "total_tokens": 53},
            }
        )
        sse = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, text=sse, headers={"content-type": "text/event-stream"})

    cfg = BenchConfig(
        targets=[Target("mock", "http://mock/v1", "m", gpu_hourly_usd=2.0)],
        concurrency=[1, 4],
        requests_per_level=8,
        warmup=1,
    )
    rows = await run_benchmark(cfg, tmp_path, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert [r["concurrency"] for r in rows] == [1, 4]
    assert rows[0]["errors"] == 0 and rows[0]["ttft_p50_s"] is not None
    assert rows[1]["cost_per_1k_requests_usd"] is not None
    assert list(tmp_path.glob("*/summary.md"))


def test_prompts_and_summary_math():
    prompts = build_prompts("note_update", 3, None, seed=1)
    assert len(prompts) == 3 and "New utterances" in prompts[0][0]["content"]
    results = [RequestResult("t", 2, True, 0.1, 1.1, 11, 100), RequestResult("t", 2, False, None, 0.5, None, None)]
    row = summarize(Target("t", "u", "m", gpu_hourly_usd=3.6), 2, results, wall_s=1.0)
    assert row["errors"] == 1 and row["itl_mean_ms"] == 100.0
    assert row["cost_per_1k_requests_usd"] == 1.0  # 3.6 $/h at 1 req/s -> $0.001 per request
