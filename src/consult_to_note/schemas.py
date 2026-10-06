"""Data models shared by the pipeline. Every clinical statement carries evidence: utterance ids."""

from __future__ import annotations

import re
from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, BeforeValidator, Field


def _coerce_ids(value: Any) -> list[int]:
    """Accept [3, "4", "U5", "[U6]"] from models and keep only parseable utterance ids."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        value = [value]
    out: list[int] = []
    for v in value:
        m = re.search(r"\d+", str(v))
        if m and int(m.group()) not in out:
            out.append(int(m.group()))
    return out


EvidenceIds = Annotated[list[int], BeforeValidator(_coerce_ids)]

Speaker = Literal["doctor", "patient", "other"]


class Utterance(BaseModel):
    id: int
    speaker: Speaker
    text: str
    start_s: float | None = None
    end_s: float | None = None


class Transcript(BaseModel):
    utterances: list[Utterance]
    source: str = "unknown"
    language: str = "en"

    def to_prompt(self) -> str:
        """Render as `[U3] doctor: ...` lines; the ids are what the model must cite."""
        return "\n".join(f"[U{u.id}] {u.speaker}: {' '.join(u.text.split())}" for u in self.utterances)

    def by_id(self) -> dict[int, Utterance]:
        return {u.id: u for u in self.utterances}

    @classmethod
    def from_text(cls, text: str, source: str = "text") -> Transcript:
        """Parse ACI-Bench style (`[doctor] hi`) or plain style (`Doctor: hi`) dialogues.

        Lines without a speaker label are appended to the previous utterance.
        """
        pattern = re.compile(
            r"^\s*(?:\[(?P<b>[a-z_ ]+)\]|(?P<p>doctor|patient|dr\.?|pt|clinician|nurse|other)\s*:)\s*(?P<t>.*)$",
            re.IGNORECASE,
        )
        utterances: list[Utterance] = []
        for raw in re.split(r"\r?\n", text.strip()):
            line = raw.strip()
            if not line:
                continue
            m = pattern.match(line)
            if m:
                label = (m.group("b") or m.group("p") or "").strip().lower().rstrip(".")
                speaker: Speaker = (
                    "doctor"
                    if label in {"doctor", "dr", "clinician", "nurse"}
                    else "patient"
                    if label in {"patient", "pt"}
                    else "other"
                )
                utterances.append(Utterance(id=len(utterances) + 1, speaker=speaker, text=m.group("t").strip()))
            elif utterances:
                utterances[-1].text += " " + line
            else:
                utterances.append(Utterance(id=1, speaker="other", text=line))
        return cls(utterances=utterances, source=source)


# ---------- extraction ----------

FindingCategory = Literal[
    "chief_complaint",
    "symptom",
    "history",
    "medication",
    "allergy",
    "social",
    "family",
    "vital",
    "exam",
    "result",
    "assessment",
    "plan",
]


class Finding(BaseModel):
    category: FindingCategory
    text: str = Field(description="Short clinical statement in third person")
    evidence: EvidenceIds = Field(description="Ids of the utterances that support this finding")
    negated: bool = Field(default=False, description="True if the finding is explicitly denied, e.g. 'no fever'")


class Condition(BaseModel):
    name: str
    status: Literal["active", "resolved", "suspected"] = "active"
    evidence: EvidenceIds = Field(default_factory=list)


class Medication(BaseModel):
    name: str
    dose: str | None = None
    frequency: str | None = None
    status: Literal["active", "started", "stopped", "changed"] = "active"
    evidence: EvidenceIds = Field(default_factory=list)


class Observation(BaseModel):
    name: str
    value: str
    unit: str | None = None
    evidence: EvidenceIds = Field(default_factory=list)


class Demographics(BaseModel):
    age_years: int | None = None
    sex: Literal["female", "male", "unknown"] = "unknown"


class Extraction(BaseModel):
    demographics: Demographics = Field(default_factory=Demographics)
    findings: list[Finding] = Field(default_factory=list)
    conditions: list[Condition] = Field(default_factory=list)
    medications: list[Medication] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)


# ---------- note ----------


class NoteSentence(BaseModel):
    text: str
    evidence: EvidenceIds = Field(description="Utterance ids that support this sentence; must not be empty")


class SOAPNote(BaseModel):
    subjective: list[NoteSentence] = Field(default_factory=list)
    objective: list[NoteSentence] = Field(default_factory=list)
    assessment: list[NoteSentence] = Field(default_factory=list)
    plan: list[NoteSentence] = Field(default_factory=list)

    SECTIONS: ClassVar[tuple[str, ...]] = ("subjective", "objective", "assessment", "plan")

    def sentences(self) -> list[tuple[str, int, NoteSentence]]:
        """(section, index, sentence) for every sentence in the note."""
        return [(sec, i, s) for sec in self.SECTIONS for i, s in enumerate(getattr(self, sec))]

    def to_markdown(self, citations: bool = True, flags: set[tuple[str, int]] | None = None) -> str:
        flags = flags or set()
        out: list[str] = []
        for sec in self.SECTIONS:
            out.append(f"## {sec.capitalize()}")
            items = getattr(self, sec)
            if not items:
                out.append("- (nothing documented)")
            for i, s in enumerate(items):
                cite = f" [{', '.join(f'U{e}' for e in s.evidence)}]" if citations and s.evidence else ""
                flag = " **[CHECK: not supported by transcript]**" if (sec, i) in flags else ""
                out.append(f"- {s.text}{cite}{flag}")
            out.append("")
        return "\n".join(out).strip() + "\n"


# ---------- grounding ----------


class GroundingCheck(BaseModel):
    section: str
    index: int
    sentence: str
    supported: bool
    score: float
    method: Literal["lexical", "llm_judge", "no_evidence"]
    reason: str = ""


class GroundingReport(BaseModel):
    checks: list[GroundingCheck]

    @property
    def support_rate(self) -> float:
        return sum(c.supported for c in self.checks) / len(self.checks) if self.checks else 1.0

    @property
    def unsupported(self) -> list[GroundingCheck]:
        return [c for c in self.checks if not c.supported]


class JudgeVerdict(BaseModel):
    supported: bool
    reason: str


# ---------- trials ----------


class CriterionAssessment(BaseModel):
    criterion: str
    kind: Literal["inclusion", "exclusion"]
    status: Literal["met", "not_met", "unknown"]
    evidence: EvidenceIds = Field(default_factory=list)
    rationale: str = ""


class TrialAssessment(BaseModel):
    nct_id: str
    title: str
    verdict: Literal["possibly_eligible", "likely_ineligible", "insufficient_information"]
    criteria: list[CriterionAssessment]
    url: str = ""
