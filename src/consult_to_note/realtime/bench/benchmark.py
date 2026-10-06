"""Load benchmark for OpenAI-compatible endpoints: hosted NIM vs self-hosted NIM vs Dynamo (or vLLM).

Measures what matters for a live scribe and for cost:
* TTFT (time to first token) p50/p95, end-to-end latency p50/p95, inter-token latency
* throughput (requests/s, output tokens/s) at each concurrency level
* error rate, and cost per 1,000 requests for self-hosted targets with a known GPU price

Workloads use real clinical text (ACI-Bench dialogues) so prompt lengths are realistic:
* note_update: short "patch the live note" prompts (the real-time path)
* full_note:   whole consultation in, full SOAP note out (the batch path)
"""

from __future__ import annotations

import asyncio
import csv
import json
import os
import random
import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

from ...schemas import Transcript
from ..chat import StreamingChat
from ..live_scribe import PATCH_PROMPT


@dataclass
class Target:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    gpu_hourly_usd: float | None = None  # price of ONE GPU per hour, for self-hosted targets
    gpus: int = 1
    thinking: bool = False

    def api_key(self) -> str | None:
        return os.getenv(self.api_key_env) if self.api_key_env else None


@dataclass
class BenchConfig:
    targets: list[Target]
    concurrency: list[int] = field(default_factory=lambda: [1, 4, 8])
    requests_per_level: int = 24
    workload: str = "note_update"
    max_tokens: int = 256
    data_csv: str | None = None
    warmup: int = 2
    seed: int = 7

    @classmethod
    def from_yaml(cls, path: str | Path) -> BenchConfig:
        raw = yaml.safe_load(Path(path).read_text())
        targets = [Target(**t) for t in raw.pop("targets")]
        return cls(targets=targets, **raw)


@dataclass
class RequestResult:
    target: str
    concurrency: int
    ok: bool
    ttft_s: float | None
    total_s: float
    output_tokens: int | None
    prompt_tokens: int | None
    error: str = ""
    tokens_estimated: bool = False


def _dialogues(data_csv: str | None) -> list[str]:
    if data_csv and Path(data_csv).exists():
        with open(data_csv, newline="", encoding="utf-8") as fh:
            return [row["dialogue"] for row in csv.DictReader(fh)]
    from importlib import resources

    sample = resources.files("consult_to_note") / "samples" / "diabetes_followup.txt"
    return [Path(str(sample)).read_text(encoding="utf-8")]


def build_prompts(workload: str, n: int, data_csv: str | None, seed: int) -> list[list[dict[str, str]]]:
    rng = random.Random(seed)
    dialogues = _dialogues(data_csv)
    prompts = []
    for _ in range(n):
        t = Transcript.from_text(rng.choice(dialogues))
        if workload == "full_note":
            content = (
                "Write a concise SOAP note (subjective, objective, assessment, plan) for this consultation. "
                "Cite utterance ids.\n\n" + t.to_prompt()
            )
        else:
            k = rng.randint(3, max(3, len(t.utterances) - 3))
            history, new = t.utterances[: k - 2], t.utterances[k - 2 : k + 1]
            note = {
                "subjective": [{"index": i, "text": u.text[:120]} for i, u in enumerate(history[-4:])],
                "objective": [],
                "assessment": [],
                "plan": [],
            }
            content = PATCH_PROMPT.format(
                note=json.dumps(note), utterances="\n".join(f"[U{u.id}] {u.speaker}: {u.text}" for u in new)
            )
        prompts.append([{"role": "user", "content": content}])
    return prompts


async def _run_level(target: Target, chat: StreamingChat, prompts, concurrency: int, max_tokens: int):
    sem = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def one(msgs):
        async with sem:
            start = time.perf_counter()
            try:
                r = await chat.complete(target.model, msgs, max_tokens=max_tokens, thinking=target.thinking)
                results.append(
                    RequestResult(
                        target.name,
                        concurrency,
                        True,
                        r.ttft_s,
                        r.total_s,
                        r.completion_tokens,
                        r.prompt_tokens,
                        tokens_estimated=r.tokens_estimated,
                    )
                )
            except Exception as err:
                results.append(
                    RequestResult(
                        target.name,
                        concurrency,
                        False,
                        None,
                        time.perf_counter() - start,
                        None,
                        None,
                        f"{type(err).__name__}: {err}"[:300],
                    )
                )

    wall_start = time.perf_counter()
    await asyncio.gather(*(one(m) for m in prompts))
    return results, time.perf_counter() - wall_start


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    k = min(len(vals) - 1, max(0, round(p / 100 * (len(vals) - 1))))
    return round(vals[k], 4)


def summarize(target: Target, concurrency: int, results: list[RequestResult], wall_s: float) -> dict:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    e2e = [r.total_s for r in ok]
    out_tok = sum(r.output_tokens or 0 for r in ok)
    # Inter-token latency only from server-reported token counts (chunk counts are not tokens).
    itl = [
        (r.total_s - r.ttft_s) / (r.output_tokens - 1)
        for r in ok
        if r.ttft_s is not None and r.output_tokens and r.output_tokens > 1 and not r.tokens_estimated
    ]
    req_per_s = len(ok) / wall_s if wall_s > 0 else 0.0
    row = {
        "target": target.name,
        "model": target.model,
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(results) - len(ok),
        "ttft_p50_s": _pct(ttft, 50),
        "ttft_p95_s": _pct(ttft, 95),
        "e2e_p50_s": _pct(e2e, 50),
        "e2e_p95_s": _pct(e2e, 95),
        "itl_mean_ms": round(statistics.mean(itl) * 1000, 2) if itl else None,
        "req_per_s": round(req_per_s, 3),
        "output_tok_per_s": round(out_tok / wall_s, 1) if wall_s > 0 else 0.0,
        "cost_per_1k_requests_usd": None,
        "tokens_estimated": any(r.tokens_estimated for r in ok),
    }
    if target.gpu_hourly_usd and req_per_s > 0:
        row["cost_per_1k_requests_usd"] = round(target.gpu_hourly_usd * target.gpus / (req_per_s * 3600) * 1000, 4)
    return row


async def run_benchmark(cfg: BenchConfig, out_dir: str | Path = "runs/bench", http_client=None) -> list[dict]:
    out = Path(out_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    all_rows, all_results = [], []
    for target in cfg.targets:
        chat = StreamingChat(target.base_url, target.api_key(), http_client=http_client)
        warm = build_prompts(cfg.workload, cfg.warmup, cfg.data_csv, cfg.seed + 1)
        if warm:
            await _run_level(target, chat, warm, 1, cfg.max_tokens)  # warm caches and connections
        for c in cfg.concurrency:
            # At least 4 requests per concurrent slot, so ramp-up and ramp-down do not dominate.
            n = max(cfg.requests_per_level, 4 * c)
            prompts = build_prompts(cfg.workload, n, cfg.data_csv, cfg.seed + c)
            results, wall = await _run_level(target, chat, prompts, c, cfg.max_tokens)
            row = summarize(target, c, results, wall)
            print(json.dumps(row))
            all_rows.append(row)
            all_results += results
    with (out / "requests.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(all_results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in all_results)
    (out / "summary.json").write_text(json.dumps({"config": {**asdict(cfg)}, "rows": all_rows}, indent=2))
    (out / "summary.md").write_text(to_markdown(all_rows))
    print(f"results in {out}")
    return all_rows


def to_markdown(rows: list[dict]) -> str:
    cols = [
        "target",
        "concurrency",
        "ttft_p50_s",
        "ttft_p95_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "itl_mean_ms",
        "req_per_s",
        "output_tok_per_s",
        "errors",
        "cost_per_1k_requests_usd",
    ]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines) + "\n"
