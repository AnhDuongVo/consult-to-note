"""The Consult-to-Note agent as a LangGraph state machine.

    extract -> draft -> ground --(unsupported sentences, revisions left)--> revise -> ground ...
                           |
                           +--> safety (NeMo Guardrails, optional) -> fhir -> trials (optional)
                                -> [human review interrupt] -> finalize

The note only becomes `final` in FHIR after a clinician approves it at the interrupt.
State is kept as plain JSON (pydantic `model_dump`) so it checkpoints cleanly.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ..fhir import build_bundle
from ..grounding import check_note
from ..llm import LLM, CallRecord, complete_structured
from ..prompts import DRAFT, EXTRACT, REVISE, SYSTEM_SCRIBE
from ..schemas import Extraction, GroundingReport, SOAPNote, Transcript, TrialAssessment


def _sum_timings(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    out = dict(a or {})
    for k, v in (b or {}).items():
        out[k] = out.get(k, 0.0) + v
    return out


class NoteState(TypedDict, total=False):
    transcript: dict
    extraction: dict
    note: dict
    grounding: dict
    revisions: int
    safety: dict
    fhir: dict
    trials: list[dict]
    approved: bool
    reviewer: str
    timings: Annotated[dict[str, float], _sum_timings]


@dataclass
class PipelineOptions:
    max_revisions: int = 1
    include_trials: bool = False
    trial_location: str | None = None
    safety_check: bool = False
    human_review: bool = False


def _timed(name: str, fn: Callable) -> Callable:
    async def wrapper(state: NoteState) -> dict:
        start = time.perf_counter()
        update = await fn(state)
        update["timings"] = {name: time.perf_counter() - start}
        return update

    wrapper.__name__ = name
    return wrapper


def build_graph(llm: LLM, options: PipelineOptions | None = None, checkpointer: Any | None = None):
    opts = options or PipelineOptions()

    def transcript_of(state: NoteState) -> Transcript:
        return Transcript.model_validate(state["transcript"])

    async def extract(state: NoteState) -> dict:
        t = transcript_of(state)
        ex = await complete_structured(
            llm,
            [
                {"role": "system", "content": SYSTEM_SCRIBE},
                {"role": "user", "content": EXTRACT.format(transcript=t.to_prompt())},
            ],
            Extraction,
            role="fast",
            step="extract",
        )
        return {"extraction": ex.model_dump(), "revisions": 0}

    async def draft(state: NoteState) -> dict:
        t = transcript_of(state)
        note = await complete_structured(
            llm,
            [
                {"role": "system", "content": SYSTEM_SCRIBE},
                {
                    "role": "user",
                    "content": DRAFT.format(extraction=json.dumps(state["extraction"]), transcript=t.to_prompt()),
                },
            ],
            SOAPNote,
            role="reasoning",
            step="draft",
        )
        return {"note": note.model_dump()}

    async def ground(state: NoteState) -> dict:
        report = await check_note(SOAPNote.model_validate(state["note"]), transcript_of(state), llm)
        return {"grounding": report.model_dump()}

    async def revise(state: NoteState) -> dict:
        report = GroundingReport.model_validate(state["grounding"])
        problems = "\n".join(f'- {c.section}[{c.index}]: "{c.sentence}" ({c.reason})' for c in report.unsupported)
        note = await complete_structured(
            llm,
            [
                {"role": "system", "content": SYSTEM_SCRIBE},
                {
                    "role": "user",
                    "content": REVISE.format(
                        problems=problems, note=json.dumps(state["note"]), transcript=transcript_of(state).to_prompt()
                    ),
                },
            ],
            SOAPNote,
            role="reasoning",
            step="revise",
        )
        return {"note": note.model_dump(), "revisions": state.get("revisions", 0) + 1}

    def after_ground(state: NoteState) -> str:
        report = GroundingReport.model_validate(state["grounding"])
        if report.unsupported and state.get("revisions", 0) < opts.max_revisions:
            return "revise"
        return "safety" if opts.safety_check else "fhir"

    async def safety(state: NoteState) -> dict:
        from ..guard import check_note_output

        result = await check_note_output(SOAPNote.model_validate(state["note"]).to_markdown(citations=False))
        return {"safety": {"allowed": result.allowed, "rail": result.rail}}

    async def fhir(state: NoteState) -> dict:
        bundle = build_bundle(
            SOAPNote.model_validate(state["note"]), Extraction.model_validate(state["extraction"]), final=False
        )
        return {"fhir": bundle}

    async def trials(state: NoteState) -> dict:
        from ..trials import prescreen

        try:
            results = await prescreen(
                transcript_of(state), Extraction.model_validate(state["extraction"]), llm, location=opts.trial_location
            )
        except Exception as err:  # network problems must not lose the note
            return {"trials": [{"error": str(err)}]}
        return {"trials": [r.model_dump() for r in results]}

    async def finalize(state: NoteState) -> dict:
        if state.get("approved") and state.get("safety", {}).get("allowed", True):
            bundle = build_bundle(
                SOAPNote.model_validate(state["note"]),
                Extraction.model_validate(state["extraction"]),
                final=True,
                author=state.get("reviewer") or "Reviewing clinician",
            )
            return {"fhir": bundle}
        return {}

    g = StateGraph(NoteState)
    for name, fn in [
        ("extract", extract),
        ("draft", draft),
        ("ground", ground),
        ("revise", revise),
        ("safety", safety),
        ("fhir", fhir),
        ("trials", trials),
        ("finalize", finalize),
    ]:
        g.add_node(name, _timed(name, fn))
    g.add_edge(START, "extract")
    g.add_edge("extract", "draft")
    g.add_edge("draft", "ground")
    g.add_conditional_edges("ground", after_ground, {"revise": "revise", "safety": "safety", "fhir": "fhir"})
    g.add_edge("revise", "ground")
    g.add_edge("safety", "fhir")
    g.add_edge("fhir", "trials" if opts.include_trials else "finalize")
    g.add_edge("trials", "finalize")
    g.add_edge("finalize", END)

    if opts.human_review:
        return g.compile(checkpointer=checkpointer or MemorySaver(), interrupt_before=["finalize"])
    return g.compile(checkpointer=checkpointer)


@dataclass
class PipelineResult:
    transcript: Transcript
    extraction: Extraction
    note: SOAPNote
    grounding: GroundingReport
    revisions: int
    fhir: dict
    trials: list[TrialAssessment] = field(default_factory=list)
    trial_errors: list[str] = field(default_factory=list)
    safety: dict | None = None
    timings: dict[str, float] = field(default_factory=dict)
    llm_calls: list[CallRecord] = field(default_factory=list)

    @classmethod
    def from_state(cls, state: dict, calls: list[CallRecord] | None = None) -> PipelineResult:
        trials, errors = [], []
        for t in state.get("trials", []) or []:
            if "error" in t:
                errors.append(t["error"])
            else:
                trials.append(TrialAssessment.model_validate(t))
        return cls(
            transcript=Transcript.model_validate(state["transcript"]),
            extraction=Extraction.model_validate(state["extraction"]),
            note=SOAPNote.model_validate(state["note"]),
            grounding=GroundingReport.model_validate(state["grounding"]),
            revisions=state.get("revisions", 0),
            fhir=state["fhir"],
            trials=trials,
            trial_errors=errors,
            safety=state.get("safety"),
            timings=state.get("timings", {}),
            llm_calls=list(calls or []),
        )

    def flagged(self) -> set[tuple[str, int]]:
        return {(c.section, c.index) for c in self.grounding.unsupported}


async def run_pipeline(transcript: Transcript, llm: LLM, options: PipelineOptions | None = None) -> PipelineResult:
    """Run end to end without the human-review pause (the FHIR note stays `preliminary`)."""
    opts = replace(options or PipelineOptions(), human_review=False)
    graph = build_graph(llm, opts)
    start_calls = len(llm.calls)
    state = await graph.ainvoke({"transcript": transcript.model_dump(), "timings": {}})
    return PipelineResult.from_state(state, llm.calls[start_calls:])
