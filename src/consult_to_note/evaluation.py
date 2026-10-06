"""Evaluation on ACI-Bench (doctor-patient dialogues with reference notes, CC BY 4.0).

Metrics per encounter:
* ROUGE-1/2/L F1 against the reference note (cheap, comparable with the literature, but shallow)
* grounding support rate (share of note sentences supported by their cited utterances)
* LLM-as-judge: omitted critical facts and unsupported facts, plus 1-5 scores
* latency and tokens per pipeline step
"""

from __future__ import annotations

import csv
import json
import re
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .agent import PipelineOptions, run_pipeline
from .llm import LLM, complete_structured
from .schemas import Transcript

ACI_FILES = {
    "train": "train.csv",
    "valid": "valid.csv",
    "test1": "clinicalnlp_taskB_test1.csv",
    "test2": "clinicalnlp_taskC_test2.csv",
    "test3": "clef_taskC_test3.csv",
}


def load_aci(split: str, data_dir: str | Path = "data/aci-bench") -> list[dict[str, str]]:
    path = Path(data_dir) / ACI_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/download_data.py")
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


# ---------- ROUGE (dependency-free) ----------


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _f1(overlap: int, a: int, b: int) -> float:
    if overlap == 0 or a == 0 or b == 0:
        return 0.0
    p, r = overlap / a, overlap / b
    return 2 * p * r / (p + r)


def rouge_n(pred: str, ref: str, n: int) -> float:
    def grams(t):
        return Counter(tuple(t[i : i + n]) for i in range(len(t) - n + 1))

    gp, gr = grams(_toks(pred)), grams(_toks(ref))
    return _f1(sum((gp & gr).values()), sum(gp.values()), sum(gr.values()))


def rouge_l(pred: str, ref: str) -> float:
    a, b = _toks(pred), _toks(ref)
    if not a or not b:
        return 0.0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return _f1(prev[-1], len(a), len(b))


# ---------- LLM judge ----------


class NoteJudgement(BaseModel):
    omissions: list[str] = Field(description="Clinically important facts in the reference note missing from the draft")
    unsupported: list[str] = Field(description="Facts in the draft that are not supported by the transcript")
    completeness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    conciseness: int = Field(ge=1, le=5)


JUDGE_NOTE = """You are an experienced physician grading an AI-drafted clinical note.
Compare the DRAFT with the REFERENCE note written by a clinician and with the TRANSCRIPT.
List clinically important omissions and any unsupported (hallucinated) facts. Ignore style and ordering.
Score completeness, correctness and conciseness from 1 (poor) to 5 (excellent).

TRANSCRIPT:
{transcript}

REFERENCE:
{reference}

DRAFT:
{draft}
"""


async def judge_note(llm: LLM, transcript: Transcript, reference: str, draft: str) -> NoteJudgement:
    return await complete_structured(
        llm,
        [
            {
                "role": "user",
                "content": JUDGE_NOTE.format(transcript=transcript.to_prompt(), reference=reference, draft=draft),
            }
        ],
        NoteJudgement,
        role="reasoning",
        step="judge_note",
    )


@dataclass
class EncounterResult:
    encounter_id: str
    rouge1: float
    rouge2: float
    rougeL: float
    support_rate: float
    sentences: int
    unsupported_after_grounding: int
    revisions: int
    omissions: int | None
    judge_unsupported: int | None
    completeness: int | None
    correctness: int | None
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    step_latency: dict


async def evaluate(
    llm: LLM,
    split: str = "valid",
    n: int = 10,
    data_dir: str | Path = "data/aci-bench",
    out_dir: str | Path = "runs/eval",
    judge: bool = True,
    max_revisions: int = 1,
) -> dict:
    rows = load_aci(split, data_dir)[:n]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[EncounterResult] = []
    failures: list[dict] = []
    with (out / f"{split}_results.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            transcript = Transcript.from_text(row["dialogue"], source=f"aci:{row['encounter_id']}")
            start = time.perf_counter()
            try:
                res = await run_pipeline(transcript, llm, PipelineOptions(max_revisions=max_revisions))
                latency = time.perf_counter() - start
                draft = res.note.to_markdown(citations=False)
                j = await judge_note(llm, transcript, row["note"], draft) if judge else None
            except Exception as err:  # keep going; failures are counted in the summary
                failures.append({"encounter_id": row["encounter_id"], "error": str(err)[:500]})
                fh.write(json.dumps(failures[-1]) + "\n")
                continue
            r = EncounterResult(
                encounter_id=row["encounter_id"],
                rouge1=rouge_n(draft, row["note"], 1),
                rouge2=rouge_n(draft, row["note"], 2),
                rougeL=rouge_l(draft, row["note"]),
                support_rate=res.grounding.support_rate,
                sentences=len(res.grounding.checks),
                unsupported_after_grounding=len(res.grounding.unsupported),
                revisions=res.revisions,
                omissions=len(j.omissions) if j else None,
                judge_unsupported=len(j.unsupported) if j else None,
                completeness=j.completeness if j else None,
                correctness=j.correctness if j else None,
                latency_s=latency,
                prompt_tokens=sum(c.prompt_tokens or 0 for c in res.llm_calls),
                completion_tokens=sum(c.completion_tokens or 0 for c in res.llm_calls),
                step_latency=res.timings,
            )
            results.append(r)
            fh.write(json.dumps({**asdict(r), "draft": draft, "judge": j.model_dump() if j else None}) + "\n")
            fh.flush()

    def mean(key: str) -> float | None:
        vals = [getattr(r, key) for r in results if getattr(r, key) is not None]
        return round(statistics.mean(vals), 4) if vals else None

    lat = sorted(r.latency_s for r in results)
    summary = {
        "split": split,
        "n": len(results),
        "failed": len(failures),
        **{
            k: mean(k)
            for k in (
                "rouge1",
                "rouge2",
                "rougeL",
                "support_rate",
                "revisions",
                "omissions",
                "judge_unsupported",
                "completeness",
                "correctness",
                "prompt_tokens",
                "completion_tokens",
            )
        },
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_max_s": round(lat[-1], 2) if lat else None,
    }
    (out / f"{split}_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
