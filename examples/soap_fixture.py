"""Synthetic consultation and scripted model responses; no live generation."""

DIALOGUE = """[doctor] What brings you in today?
[patient] My right knee has been hurting for two weeks after a hike.
[doctor] Any swelling or locking?
[patient] Some swelling, no locking. I take ibuprofen 400 milligrams when it hurts.
[doctor] On exam there is mild swelling of the right knee, full range of motion, ligaments stable.
[doctor] I think this is a mild sprain. Rest, ice, and come back in four weeks if it is not better."""

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

BAD_NOTE = {
    **GOOD_NOTE,
    "subjective": GOOD_NOTE["subjective"][:2]
    + [{"text": "Takes ibuprofen 800 milligrams three times daily.", "evidence": [4]}],
}
