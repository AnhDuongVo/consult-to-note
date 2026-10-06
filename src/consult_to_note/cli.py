"""Command line interface: `c2n --help`."""

from __future__ import annotations

import asyncio
import json
import uuid
from importlib import resources
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from .agent import PipelineOptions, PipelineResult, build_graph
from .config import get_settings
from .llm import NIMClient
from .realtime import cli as realtime_cli
from .schemas import SOAPNote, Transcript

app = typer.Typer(add_completion=False, help="Grounded clinical notes from consultations, batch and real-time.")
console = Console()


def _load_transcript(path: Path) -> Transcript:
    if path.suffix == ".json":
        return Transcript.model_validate_json(path.read_text(encoding="utf-8"))
    return Transcript.from_text(path.read_text(encoding="utf-8"), source=path.name)


def _sample_path() -> Path:
    return Path(str(resources.files("consult_to_note") / "samples" / "diabetes_followup.txt"))


def _save(result: PipelineResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transcript.json").write_text(result.transcript.model_dump_json(indent=2))
    (out_dir / "extraction.json").write_text(result.extraction.model_dump_json(indent=2))
    (out_dir / "note.json").write_text(result.note.model_dump_json(indent=2))
    (out_dir / "note.md").write_text(result.note.to_markdown(flags=result.flagged()))
    (out_dir / "grounding.json").write_text(result.grounding.model_dump_json(indent=2))
    (out_dir / "fhir_bundle.json").write_text(json.dumps(result.fhir, indent=2))
    if result.trials or result.trial_errors:
        (out_dir / "trials.json").write_text(
            json.dumps({"trials": [t.model_dump() for t in result.trials], "errors": result.trial_errors}, indent=2)
        )
    (out_dir / "profile.json").write_text(
        json.dumps(
            {
                "step_seconds": result.timings,
                "llm_calls": [c.__dict__ for c in result.llm_calls],
            },
            indent=2,
        )
    )


def _print_summary(result: PipelineResult) -> None:
    console.print(Markdown(result.note.to_markdown(flags=result.flagged())))
    g = result.grounding
    console.print(
        f"[bold]Grounding:[/] {len(g.checks) - len(g.unsupported)}/{len(g.checks)} sentences supported "
        f"({g.support_rate:.0%}), revisions: {result.revisions}"
    )
    if result.safety is not None:
        console.print(f"[bold]Safety rail:[/] {'passed' if result.safety.get('allowed') else 'BLOCKED'}")
    table = Table(title="Latency per step (s)")
    table.add_column("step")
    table.add_column("seconds", justify="right")
    for step, sec in result.timings.items():
        table.add_row(step, f"{sec:.2f}")
    table.add_row("total", f"{sum(result.timings.values()):.2f}")
    console.print(table)
    tok_in = sum(c.prompt_tokens or 0 for c in result.llm_calls)
    tok_out = sum(c.completion_tokens or 0 for c in result.llm_calls)
    console.print(f"[bold]Tokens:[/] {tok_in} in, {tok_out} out over {len(result.llm_calls)} LLM calls")
    for t in result.trials:
        console.print(f"[bold]Trial {t.nct_id}[/] {t.title}: {t.verdict}")
    for e in result.trial_errors:
        console.print(f"[yellow]Trial search failed: {e}[/]")


async def _run(transcript: Transcript, opts: PipelineOptions, deidentify: bool, out_dir: Path) -> PipelineResult:
    settings = get_settings()
    llm = NIMClient(settings)
    if deidentify:
        from .guard import mask_transcript

        transcript = await mask_transcript(transcript)
        console.print("[green]Identifiers masked with NeMo Guardrails + Presidio before any LLM call.[/]")
    graph = build_graph(llm, opts)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state = await graph.ainvoke({"transcript": transcript.model_dump(), "timings": {}}, config)

    if opts.human_review:
        snapshot = await graph.aget_state(config)
        draft = PipelineResult.from_state(snapshot.values, llm.calls)
        console.rule("Draft for clinician review")
        _print_summary(draft)
        if draft.safety is not None and not draft.safety.get("allowed", True):
            console.print(
                f"[red]The safety rail blocked this draft ({draft.safety.get('rail')}). "
                "It stays 'preliminary'; edit the transcript or prompts and run again.[/]"
            )
        elif typer.confirm("Approve and sign this note?", default=False):
            reviewer = typer.prompt("Your name", default="Reviewing clinician")
            await graph.aupdate_state(config, {"approved": True, "reviewer": reviewer})
        else:
            console.print("[yellow]Not approved: the FHIR Composition stays 'preliminary'.[/]")
        state = await graph.ainvoke(None, config)

    result = PipelineResult.from_state(state, llm.calls)
    _save(result, out_dir)
    return result


@app.command()
def note(
    transcript_file: Path = typer.Argument(
        ..., exists=True, help="Transcript .txt (lines starting with doctor:/patient:) or .json"
    ),
    out_dir: Path = typer.Option(Path("runs/latest"), help="Where to write note, FHIR bundle and profile"),
    review: bool = typer.Option(True, help="Pause for clinician approval before finalising"),
    trials: bool = typer.Option(False, help="Pre-screen recruiting trials on ClinicalTrials.gov"),
    trial_location: str = typer.Option(None, help="Limit the trial search to a location, e.g. Switzerland"),
    deidentify: bool = typer.Option(False, help="Mask identifiers with NeMo Guardrails before LLM calls"),
    safety: bool = typer.Option(False, help="Run the NeMo Guardrails output self-check on the draft"),
    max_revisions: int = typer.Option(1, help="Grounding repair rounds"),
):
    """Turn a consultation transcript into a grounded SOAP note and a FHIR bundle."""
    opts = PipelineOptions(
        max_revisions=max_revisions,
        include_trials=trials,
        trial_location=trial_location,
        safety_check=safety,
        human_review=review,
    )
    result = asyncio.run(_run(_load_transcript(transcript_file), opts, deidentify, out_dir))
    if not review:
        _print_summary(result)
    console.print(f"[green]Saved to {out_dir}/[/]")


@app.command()
def demo(
    out_dir: Path = typer.Option(Path("runs/demo")),
    trials: bool = typer.Option(True),
    review: bool = typer.Option(False),
):
    """Run the bundled synthetic diabetes follow-up consultation end to end."""
    opts = PipelineOptions(include_trials=trials, trial_location=None, human_review=review)
    result = asyncio.run(_run(_load_transcript(_sample_path()), opts, False, out_dir))
    if not review:
        _print_summary(result)
    console.print(f"[green]Saved to {out_dir}/[/]")


@app.command()
def transcribe(
    audio: Path = typer.Argument(..., exists=True, help="16-bit PCM WAV. With --patient-audio: the doctor channel"),
    patient_audio: Path = typer.Option(None, exists=True, help="Separate patient channel (e.g. PriMock57)"),
    out: Path = typer.Option(Path("runs/transcript.json")),
    boost: str = typer.Option("", help="Comma-separated words to boost, e.g. drug names"),
):
    """Speech to transcript with Parakeet (Riva) on build.nvidia.com or a local NIM."""
    from .asr import transcribe_file, transcribe_two_channel

    settings = get_settings()
    boost_words = [w.strip() for w in boost.split(",") if w.strip()] or None
    if patient_audio:
        transcript = transcribe_two_channel(audio, patient_audio, settings, boost_words)
    else:
        transcript = asyncio.run(transcribe_file(audio, settings, NIMClient(settings), boost_words=boost_words))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(transcript.model_dump_json(indent=2))
    console.print(transcript.to_prompt())
    console.print(f"[green]Saved {len(transcript.utterances)} utterances to {out}[/]")


@app.command(name="eval")
def eval_cmd(
    split: str = typer.Option("valid", help="ACI-Bench split: train, valid, test1, test2, test3"),
    n: int = typer.Option(10, help="Number of encounters"),
    data_dir: Path = typer.Option(Path("data/aci-bench")),
    out_dir: Path = typer.Option(Path("runs/eval")),
    judge: bool = typer.Option(True, help="Use the reasoning model as LLM-as-judge"),
    max_revisions: int = typer.Option(1),
):
    """Evaluate on ACI-Bench: ROUGE, grounding, LLM-judge omissions and hallucinations, latency, tokens."""
    from .evaluation import evaluate

    summary = asyncio.run(evaluate(NIMClient(get_settings()), split, n, data_dir, out_dir, judge, max_revisions))
    table = Table(title=f"ACI-Bench {split} (n={summary['n']})")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in summary.items():
        if k not in {"split", "n"}:
            table.add_row(k, str(v))
    console.print(table)


@app.command()
def models():
    """List the model IDs your key can use (check these against .env)."""
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(base_url=s.base_url, api_key=s.require_api_key())
    ids = sorted(m.id for m in client.models.list().data)
    for mid in ids:
        mark = " <- reasoning" if mid == s.model_reasoning else " <- fast" if mid == s.model_fast else ""
        console.print(f"{mid}{mark}")
    for role, mid in (("C2N_MODEL_REASONING", s.model_reasoning), ("C2N_MODEL_FAST", s.model_fast)):
        if mid not in ids:
            console.print(f"[red]{role}={mid} is not in the list. Pick another ID in .env[/]")


@app.command()
def show(run_dir: Path = typer.Argument(Path("runs/latest"))):
    """Print a saved note with its grounding flags."""
    from .schemas import GroundingReport

    note_ = SOAPNote.model_validate_json((run_dir / "note.json").read_text())
    flags: set[tuple[str, int]] = set()
    grounding_file = run_dir / "grounding.json"
    if grounding_file.exists():
        report = GroundingReport.model_validate_json(grounding_file.read_text())
        flags = {(c.section, c.index) for c in report.unsupported}
    console.print(Markdown(note_.to_markdown(flags=flags)))


# Real-time commands (live, bench, plot, aiperf) share the same entry point.
app.registered_commands.extend(realtime_cli.app.registered_commands)


if __name__ == "__main__":
    app()
