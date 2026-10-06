"""Real-time commands registered on `c2n`: live, bench, plot, aiperf."""

from __future__ import annotations

import asyncio
import json
import shlex
from pathlib import Path

import typer
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel

from ..config import get_settings
from ..schemas import Transcript
from .chat import StreamingChat
from .live_scribe import LiveScribe, UpdateMetric
from .router import Candidate, LatencyRouter

app = typer.Typer(add_completion=False, help="Real-time clinical note drafting and NIM / Dynamo benchmarking.")
console = Console()

DEFAULT_MODELS = "nvidia/nemotron-3-super-120b-a12b,nvidia/nemotron-3.5-lightning-30b-a3b"


@app.command()
def live(
    transcript: Path = typer.Option(None, exists=True, help="Replay a text transcript at speaking pace (no audio)"),
    wav: Path = typer.Option(None, exists=True, help="Stream one mixed WAV through Parakeet streaming ASR"),
    doctor_wav: Path = typer.Option(None, exists=True, help="Doctor channel (use with --patient-wav)"),
    patient_wav: Path = typer.Option(None, exists=True),
    mic: bool = typer.Option(False, help="Use the microphone (requires the 'mic' extra)"),
    budget: float = typer.Option(1.0, help="Latency budget in seconds from end of utterance to updated note"),
    models: str = typer.Option(DEFAULT_MODELS, help="Comma-separated model IDs, best quality first"),
    speed: float = typer.Option(1.0, help="Replay speed for --transcript (2.0 = twice as fast)"),
    final: bool = typer.Option(True, help="Run the full grounded pipeline after the last word"),
    out_dir: Path = typer.Option(Path("runs/live")),
):
    """Draft the note while the consultation is running, within a latency budget."""
    from . import streaming_asr

    settings = get_settings()
    if transcript:
        events = streaming_asr.simulate_from_transcript(Transcript.from_text(transcript.read_text()), speed=speed)
    elif doctor_wav and patient_wav:
        events = streaming_asr.stream_two_channel(str(doctor_wav), str(patient_wav), settings)
    elif wav:
        events = streaming_asr.stream_wav(str(wav), settings)
    elif mic:
        events = streaming_asr.stream_microphone(settings)
    else:
        raise typer.BadParameter("Choose one input: --transcript, --wav, --doctor-wav/--patient-wav or --mic")

    router = LatencyRouter([Candidate(m.strip()) for m in models.split(",") if m.strip()], budget_s=budget)
    chat = StreamingChat(settings.base_url, settings.require_api_key())

    async def main():
        with Live(console=console, refresh_per_second=8) as view:

            def render(scribe: LiveScribe, m: UpdateMetric | None):
                status = (
                    "waiting for speech"
                    if m is None
                    else (
                        f"update {len(scribe.updates)}: {m.speech_to_note_s:.2f}s speech-to-note "
                        f"({'OK' if m.within_budget else 'OVER BUDGET'}), model {m.model}, batch {m.batch_size}, "
                        f"TTFT {m.ttft_s or 0:.2f}s"
                    )
                )
                view.update(
                    Group(
                        Panel(Markdown(scribe.note.to_markdown(flags=scribe.flagged)), title="Live note (preview)"),
                        status,
                    )
                )

            scribe = LiveScribe(chat, router, on_update=render)
            render(scribe, None)

            finalize = None
            if final:
                from ..agent import run_pipeline
                from ..llm import NIMClient

                async def finalize(t: Transcript):
                    llm = NIMClient(settings)
                    if any(u.speaker == "other" for u in t.utterances):
                        # Mixed audio has no speaker roles: label doctor/patient before the final note.
                        from ..asr import label_speakers

                        t = await label_speakers(t, llm)
                    return await run_pipeline(t, llm)

            return await scribe.run(events, finalize)

    report = asyncio.run(main())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "live_note.md").write_text(report.note.to_markdown(flags=report.flagged))
    (out_dir / "transcript.json").write_text(report.transcript.model_dump_json(indent=2))
    (out_dir / "updates.json").write_text(json.dumps([m.__dict__ for m in report.updates], indent=2))
    summary = report.summary()
    if report.final_result is not None:
        res = report.final_result
        (out_dir / "final_note.md").write_text(res.note.to_markdown(flags=res.flagged()))
        (out_dir / "fhir_bundle.json").write_text(json.dumps(res.fhir, indent=2))
        summary["final_support_rate"] = round(res.grounding.support_rate, 3)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    console.print_json(json.dumps(summary))
    console.print(f"[green]Saved to {out_dir}/[/]")


@app.command()
def bench(
    config: Path = typer.Argument(Path("deploy/targets.yml"), exists=True),
    out_dir: Path = typer.Option(Path("runs/bench")),
):
    """Benchmark the targets in a YAML file (hosted NIM, local NIM, Dynamo, vLLM)."""
    from .bench.benchmark import BenchConfig, run_benchmark, to_markdown

    rows = asyncio.run(run_benchmark(BenchConfig.from_yaml(config), out_dir))
    console.print(Markdown(to_markdown(rows)))


@app.command()
def plot(summary_json: Path = typer.Argument(..., exists=True)):
    """Charts from a benchmark summary.json (requires the 'plot' extra)."""
    from .bench.plot import plot_summary

    for p in plot_summary(summary_json):
        console.print(f"saved {p}")


@app.command()
def aiperf(
    url: str = typer.Option("http://localhost:8000"),
    model: str = typer.Option("nvidia/nemotron-3-nano-30b-a3b"),
    tokenizer: str = typer.Option("", help="Hugging Face tokenizer id matching the model"),
    concurrency: str = typer.Option("1,4,16"),
    input_tokens: int = typer.Option(900, help="Mean prompt length (a note-update prompt is ~600-1200)"),
    output_tokens: int = typer.Option(200),
):
    """Print NVIDIA AIPerf commands that reproduce the benchmark with NVIDIA's own tool."""
    for c in concurrency.split(","):
        cmd = [
            "aiperf",
            "profile",
            "--model",
            model,
            "--url",
            url,
            "--endpoint-type",
            "chat",
            "--streaming",
            "--concurrency",
            c.strip(),
            "--request-count",
            str(max(32, 8 * int(c))),
            "--synthetic-input-tokens-mean",
            str(input_tokens),
            "--output-tokens-mean",
            str(output_tokens),
        ]
        if tokenizer:
            cmd += ["--tokenizer", tokenizer]
        console.print(shlex.join(cmd))
