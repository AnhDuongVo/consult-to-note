import pytest

from consult_to_note.llm import FakeLLM
from consult_to_note.schemas import Transcript

DIALOGUE = """[doctor] What brings you in today?
[patient] My right knee has been hurting for two weeks after a hike.
[doctor] Any swelling or locking?
[patient] Some swelling, no locking. I take ibuprofen 400 milligrams when it hurts.
[doctor] On exam there is mild swelling of the right knee, full range of motion, ligaments stable.
[doctor] I think this is a mild sprain. Rest, ice, and come back in four weeks if it is not better."""


@pytest.fixture
def transcript() -> Transcript:
    return Transcript.from_text(DIALOGUE)


EXTRACTION = {
    "demographics": {"age_years": None, "sex": "unknown"},
    "findings": [
        {"category": "chief_complaint", "text": "Right knee pain for two weeks after a hike", "evidence": [2]},
        {"category": "symptom", "text": "Knee swelling, no locking", "evidence": [4]},
    ],
    "conditions": [{"name": "Right knee sprain", "status": "suspected", "evidence": [6]}],
    "medications": [
        {"name": "ibuprofen", "dose": "400 mg", "frequency": "as needed", "status": "active", "evidence": [4]}
    ],
    "observations": [],
    "allergies": [],
}

GOOD_NOTE = {
    "subjective": [
        {"text": "Right knee pain for two weeks after a hike.", "evidence": [2]},
        {"text": "Swelling of the knee, no locking.", "evidence": [4]},
        {"text": "Takes ibuprofen 400 milligrams as needed.", "evidence": [4]},
    ],
    "objective": [
        {"text": "Mild swelling of the right knee, full range of motion, ligaments stable.", "evidence": [5]}
    ],
    "assessment": [{"text": "Mild sprain of the right knee.", "evidence": [5, 6]}],
    "plan": [{"text": "Rest and ice; return in four weeks if not better.", "evidence": [6]}],
}

# Same note with a hallucinated dose (800 mg) that the grounding check must catch.
BAD_NOTE = {
    **GOOD_NOTE,
    "subjective": GOOD_NOTE["subjective"][:2]
    + [{"text": "Takes ibuprofen 800 milligrams three times daily.", "evidence": [4]}],
}


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM(
        {
            "extract": EXTRACTION,
            "draft": BAD_NOTE,
            "revise": GOOD_NOTE,
            # Judge: rejects the hallucinated 800 mg dose, accepts paraphrases.
            "ground_judge": lambda msgs: {"supported": "800" not in msgs[-1]["content"], "reason": "fake judge"},
        }
    )
