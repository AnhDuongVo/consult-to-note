"""Live scribe: keeps a SOAP note up to date while the consultation is still running.

Design for low latency:
* Incremental patches, not rewrites. The model sees the current note plus only the NEW utterances and
  returns a few add/replace/remove operations, so output stays short (output tokens dominate latency).
* Batching under load. While one update is in flight, new utterances queue up and go into the next
  update together, so the scribe never falls further and further behind.
* Latency-aware routing (`LatencyRouter`): the best model that fits the budget, otherwise the fastest.
* Cheap live grounding: new sentences get the lexical check from the grounding module (no extra LLM call).
* After the last word, the full batch pipeline (extract, draft, ground, revise, FHIR) produces the
  final grounded note. The live note is a preview; the final note is what the clinician signs.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from ..grounding import lexical_score
from ..llm import extract_json
from ..schemas import NoteSentence, SOAPNote, Transcript, Utterance
from .chat import StreamingChat
from .router import LatencyRouter
from .streaming_asr import SpeechEvent

Section = Literal["subjective", "objective", "assessment", "plan"]

PATCH_PROMPT = """You keep a SOAP note up to date DURING a doctor-patient consultation.

Current note (each sentence has an index per section):
{note}

New utterances since the last update:
{utterances}

Return JSON {{"ops": [...]}} with the minimal changes:
- {{"op": "add", "section": "...", "text": "...", "evidence": [ids]}} for a new fact
- {{"op": "replace", "section": "...", "index": i, "text": "...", "evidence": [ids]}} if new speech corrects a sentence
- {{"op": "remove", "section": "...", "index": i}} if a sentence turned out to be wrong
Only document what was said. Cite utterance ids. Return {{"ops": []}} if nothing clinically relevant is new.
"""


class PatchOp(BaseModel):
    op: Literal["add", "replace", "remove"]
    section: Section
    index: int | None = None
    text: str | None = None
    evidence: list[int] = Field(default_factory=list)


class NotePatch(BaseModel):
    ops: list[PatchOp]


def empty_note() -> SOAPNote:
    return SOAPNote(subjective=[], objective=[], assessment=[], plan=[])


def apply_patch(note: SOAPNote, patch: NotePatch, known_ids: set[int]) -> tuple[SOAPNote, int]:
    """Apply valid ops; drop ops that cite unknown utterances or bad indices. Returns (note, applied)."""
    data = {s: list(getattr(note, s)) for s in SOAPNote.SECTIONS}
    applied = 0
    removals: list[tuple[str, int]] = []
    for op in patch.ops:
        items = data[op.section]
        if op.op in {"add", "replace"}:
            evidence = [e for e in op.evidence if e in known_ids]
            if not op.text or not evidence:
                continue
            sentence = NoteSentence(text=op.text.strip(), evidence=evidence)
            if op.op == "add":
                items.append(sentence)
                applied += 1
            elif op.index is not None and 0 <= op.index < len(items):
                items[op.index] = sentence
                applied += 1
        elif op.op == "remove" and op.index is not None and 0 <= op.index < len(items):
            removals.append((op.section, op.index))
    for section, index in sorted(set(removals), key=lambda r: -r[1]):  # highest index first
        del data[section][index]
        applied += 1
    return SOAPNote(**data), applied


@dataclass
class UpdateMetric:
    batch_size: int
    model: str
    route_reason: str
    ttft_s: float | None
    llm_s: float
    speech_to_note_s: float  # oldest utterance in the batch became final -> note updated
    ops_applied: int
    output_tokens: int | None
    within_budget: bool


@dataclass
class LiveReport:
    note: SOAPNote
    transcript: Transcript
    updates: list[UpdateMetric]
    flagged: set[tuple[str, int]]
    budget_s: float
    final_result: object | None = None
    final_note_after_last_word_s: float | None = None

    def summary(self) -> dict:
        lat = sorted(m.speech_to_note_s for m in self.updates)

        def pct(p: float) -> float | None:
            if not lat:
                return None
            k = min(len(lat) - 1, max(0, round(p / 100 * (len(lat) - 1))))
            return round(lat[k], 3)

        return {
            "updates": len(self.updates),
            "utterances": len(self.transcript.utterances),
            "speech_to_note_p50_s": pct(50),
            "speech_to_note_p95_s": pct(95),
            "speech_to_note_max_s": round(lat[-1], 3) if lat else None,
            "within_budget_share": round(sum(m.within_budget for m in self.updates) / len(self.updates), 3)
            if self.updates
            else None,
            "mean_batch_size": round(statistics.mean(m.batch_size for m in self.updates), 2) if self.updates else None,
            "models_used": sorted({m.model for m in self.updates}),
            "live_sentences_flagged": len(self.flagged),
            "budget_s": self.budget_s,
            "final_note_after_last_word_s": self.final_note_after_last_word_s,
        }


@dataclass
class LiveScribe:
    chat: StreamingChat
    router: LatencyRouter
    on_update: Callable[[LiveScribe, UpdateMetric], Awaitable[None] | None] | None = None
    utterances: list[Utterance] = field(default_factory=list)
    note: SOAPNote = field(default_factory=empty_note)
    updates: list[UpdateMetric] = field(default_factory=list)
    flagged: set[tuple[str, int]] = field(default_factory=set)
    _pending: list[tuple[Utterance, float]] = field(default_factory=list)
    _worker: asyncio.Task | None = None

    def transcript(self) -> Transcript:
        return Transcript(utterances=list(self.utterances), source="live")

    def _note_prompt(self) -> str:
        return json.dumps(
            {s: [{"index": i, "text": x.text} for i, x in enumerate(getattr(self.note, s))] for s in SOAPNote.SECTIONS}
        )

    def _regrade(self) -> None:
        utt = {u.id: u for u in self.utterances}
        flagged = set()
        for section, idx, sent in self.note.sentences():
            evidence = " ".join(utt[e].text for e in sent.evidence if e in utt)
            score, _ = lexical_score(sent.text, evidence)
            if score < 0.45:
                flagged.add((section, idx))
        self.flagged = flagged

    async def _update_loop(self) -> None:
        while self._pending:
            batch, self._pending = self._pending, []
            decision = self.router.choose(len(batch))
            lines = "\n".join(f"[U{u.id}] {u.speaker}: {u.text}" for u, _ in batch)
            messages = [{"role": "user", "content": PATCH_PROMPT.format(note=self._note_prompt(), utterances=lines)}]
            try:
                res = await self.chat.complete(
                    decision.model, messages, max_tokens=decision.max_tokens, json_schema=NotePatch.model_json_schema()
                )
                patch = NotePatch.model_validate(extract_json(res.text))
            except Exception as err:  # keep listening even if one update fails
                patch, res = NotePatch(ops=[]), None
                print(f"[live scribe] update failed: {err}")
            self.note, applied = apply_patch(self.note, patch, {u.id for u in self.utterances})
            self._regrade()
            done = time.perf_counter()
            llm_s = res.total_s if res else 0.0
            if res is None:
                self.router.observe_failure(decision.model)
            else:
                self.router.observe(decision.model, llm_s)
            metric = UpdateMetric(
                batch_size=len(batch),
                model=decision.model,
                route_reason=decision.reason,
                ttft_s=res.ttft_s if res else None,
                llm_s=llm_s,
                speech_to_note_s=done - min(t for _, t in batch),
                ops_applied=applied,
                output_tokens=res.completion_tokens if res else None,
                within_budget=(done - min(t for _, t in batch)) <= self.router.budget_s,
            )
            self.updates.append(metric)
            if self.on_update:
                try:
                    maybe = self.on_update(self, metric)
                    if asyncio.iscoroutine(maybe):
                        await maybe
                except Exception as err:  # a display error must not stop the scribe
                    print(f"[live scribe] on_update failed: {err}")

    def _feed(self, ev: SpeechEvent) -> None:
        if not ev.is_final or not ev.text:
            return
        u = Utterance(
            id=len(self.utterances) + 1,
            speaker=ev.speaker if ev.speaker in {"doctor", "patient"} else "other",
            text=ev.text,
            end_s=ev.audio_end_s,
        )
        self.utterances.append(u)
        self._pending.append((u, ev.wall_time))
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._update_loop())

    async def run(
        self,
        events: AsyncIterator[SpeechEvent],
        finalize: Callable[[Transcript], Awaitable[object]] | None = None,
    ) -> LiveReport:
        async for ev in events:
            self._feed(ev)
        if self._worker:
            await self._worker
        report = LiveReport(self.note, self.transcript(), self.updates, self.flagged, self.router.budget_s)
        if finalize is not None:
            start = time.perf_counter()
            report.final_result = await finalize(self.transcript())
            report.final_note_after_last_word_s = round(time.perf_counter() - start, 2)
        return report
