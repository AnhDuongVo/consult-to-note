"""NeMo Agent Toolkit plugin.

Registers two components (discovered through the `nat.components` entry point in pyproject.toml):

* `consult_to_note`: the full pipeline as a workflow/function. Input: a transcript ([doctor]/[patient]
  lines). Output: the SOAP note in Markdown with citations, a grounding summary and FHIR resource counts.
* `ground_note_sentence`: a small tool that checks one note sentence against cited transcript text.
  Useful for other agents (for example via `nat mcp serve`).

LLMs come from the YAML config (`llms:` with `_type: nim`) through `builder.get_llm(...)`, so the toolkit's
profiler and evaluator see every call.
"""

from __future__ import annotations

from nat.builder.builder import Builder
from nat.builder.framework_enum import LLMFrameworkEnum
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.data_models.component_ref import LLMRef
from nat.data_models.function import FunctionBaseConfig
from pydantic import Field


class ConsultToNoteConfig(FunctionBaseConfig, name="consult_to_note"):
    llm_name: LLMRef = Field(description="Reasoning LLM for drafting and revising")
    fast_llm_name: LLMRef | None = Field(default=None, description="Fast LLM for extraction and judging")
    max_revisions: int = 1
    include_trials: bool = False
    trial_location: str | None = None
    output: str = Field(default="markdown", description="'markdown' (note + summary) or 'json' (full result)")


@register_function(config_type=ConsultToNoteConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN])
async def consult_to_note(config: ConsultToNoteConfig, builder: Builder):
    import json

    from ..agent import PipelineOptions, run_pipeline
    from ..llm import LangChainAdapter
    from ..schemas import Transcript

    reasoning = await builder.get_llm(config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
    fast = (
        await builder.get_llm(config.fast_llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN)
        if config.fast_llm_name
        else reasoning
    )
    llm = LangChainAdapter(reasoning, fast)

    async def _run(transcript: str) -> str:
        """Generate a preliminary, reviewable SOAP draft; this invocation does not complete clinician approval."""
        result = await run_pipeline(
            Transcript.from_text(transcript, source="nat"),
            llm,
            PipelineOptions(
                max_revisions=config.max_revisions,
                include_trials=config.include_trials,
                trial_location=config.trial_location,
            ),
        )
        if config.output == "json":
            return json.dumps(
                {
                    "note": result.note.model_dump(),
                    "grounding": result.grounding.model_dump(),
                    "fhir": result.fhir,
                    "trials": [t.model_dump() for t in result.trials],
                }
            )
        g = result.grounding
        counts: dict[str, int] = {}
        for entry in result.fhir["entry"]:
            rtype = entry["resource"]["resourceType"]
            counts[rtype] = counts.get(rtype, 0) + 1
        footer = (
            f"\n---\nPreliminary draft: clinician review required.\nGrounding: {len(g.checks) - len(g.unsupported)}/{len(g.checks)} sentences supported; "
            f"revisions: {result.revisions}; FHIR: {counts}"
        )
        return result.note.to_markdown(flags=result.flagged()) + footer

    yield FunctionInfo.from_fn(_run, description=_run.__doc__)


class GroundSentenceConfig(FunctionBaseConfig, name="ground_note_sentence"):
    pass


@register_function(config_type=GroundSentenceConfig)
async def ground_note_sentence(config: GroundSentenceConfig, builder: Builder):
    from ..grounding import lexical_score, quantities, risk_flags

    async def _check(sentence: str, evidence: str) -> str:
        """Check whether a clinical note sentence is supported by the given transcript evidence text."""
        score, reason = lexical_score(sentence, evidence)
        flags = risk_flags(sentence, evidence)
        if quantities(sentence) - quantities(evidence):
            flags.append("quantity/unit mismatch")
        if flags:
            score = 0.0
            reason += "; " + "; ".join(flags)
        return f"score={score:.2f} ({reason}); lexical screening only, clinician review required"

    yield FunctionInfo.from_fn(_check, description=_check.__doc__)
