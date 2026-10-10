import pytest

from consult_to_note.grounding import check_note
from consult_to_note.llm import FakeLLM
from consult_to_note.schemas import SOAPNote, Transcript


@pytest.mark.parametrize(
    "claim,evidence,ids",
    [
        ("Metformin 250 mg daily", "Metformin 25 mg daily", [1]),
        ("Metformin 25 g daily", "Metformin 25 mg daily", [1]),
        ("Patient has chest pain", "Patient has no chest pain", [1]),
        ("Pain in left knee", "Pain in right knee", [1]),
        ("Metformin 25 mg daily", "Metformin 25 mg daily", [1, 99]),
    ],
)
async def test_hard_failure_cannot_be_overridden(claim, evidence, ids):
    def unexpected_judge(messages):
        pytest.fail("Judge must never see deterministic failures")

    llm = FakeLLM({"ground_judge": unexpected_judge})
    note = SOAPNote.model_validate({"subjective": [{"text": claim, "evidence": ids}]})
    report = await check_note(note, Transcript.from_text("Patient: " + evidence), llm)
    assert len(report.unsupported) == 1
    assert report.checks[0].method != "llm_judge"


async def test_ambiguous_paraphrase_can_use_judge():
    llm = FakeLLM({"ground_judge": {"supported": True, "reason": "paraphrase"}})
    note = SOAPNote.model_validate({"subjective": [{"text": "Reports dyspnoea", "evidence": [1]}]})
    report = await check_note(note, Transcript.from_text("Patient: I am short of breath"), llm)
    assert report.checks[0].supported and report.checks[0].method == "llm_judge"
