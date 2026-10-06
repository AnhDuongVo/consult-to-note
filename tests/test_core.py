import pytest

from conftest import BAD_NOTE, EXTRACTION, GOOD_NOTE
from consult_to_note.evaluation import rouge_l, rouge_n
from consult_to_note.fhir import build_bundle, validate_bundle
from consult_to_note.grounding import check_note, lexical_score
from consult_to_note.llm import extract_json
from consult_to_note.schemas import CriterionAssessment, Extraction, SOAPNote, Transcript
from consult_to_note.trials import split_criteria, verdict_from


def test_transcript_parsing_formats():
    t = Transcript.from_text("Doctor: hello\nPatient: my head hurts\nsince yesterday\n[doctor] okay")
    assert [u.speaker for u in t.utterances] == ["doctor", "patient", "doctor"]
    assert t.utterances[1].text == "my head hurts since yesterday"
    assert t.to_prompt().splitlines()[0] == "[U1] doctor: hello"


def test_extract_json_handles_think_and_fences():
    assert extract_json('<think>hmm {"a": 0}</think>```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"b": [1, 2], "c": "}"} done') == {"b": [1, 2], "c": "}"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_lexical_numbers_are_strict():
    ev = "I take ibuprofen 400 milligrams when it hurts"
    assert lexical_score("Takes ibuprofen 400 milligrams", ev)[0] >= 0.6
    assert lexical_score("Takes ibuprofen 800 milligrams", ev)[0] == 0.0
    assert lexical_score("Metformin 1,000 mg twice daily", "metformin 1000 mg twice a day")[0] >= 0.6
    assert lexical_score("Follow up in four weeks", "come back in 4 weeks")[0] > 0


async def test_grounding_flags_hallucinated_dose(transcript):
    bad = await check_note(SOAPNote.model_validate(BAD_NOTE), transcript)
    assert [(c.section, c.index) for c in bad.unsupported] == [("subjective", 2)]
    good = await check_note(SOAPNote.model_validate(GOOD_NOTE), transcript)
    assert good.support_rate == 1.0


async def test_grounding_missing_evidence(transcript):
    note = SOAPNote.model_validate({**GOOD_NOTE, "plan": [{"text": "Start physiotherapy.", "evidence": [99]}]})
    report = await check_note(note, transcript)
    assert report.unsupported[0].method == "no_evidence"


def test_fhir_bundle_is_valid():
    bundle = build_bundle(SOAPNote.model_validate(GOOD_NOTE), Extraction.model_validate(EXTRACTION))
    validate_bundle(bundle)
    types = [e["resource"]["resourceType"] for e in bundle["entry"]]
    assert types[0] == "Composition" and "Condition" in types and "MedicationStatement" in types
    assert bundle["entry"][0]["resource"]["status"] == "preliminary"


def test_split_criteria_and_verdict():
    text = """Inclusion Criteria:

* Adults aged 18 to 75
* Type 2 diabetes with HbA1c 7.5% or higher

Exclusion Criteria:

* Type 1 diabetes
* Pregnancy"""
    crit = split_criteria(text)
    assert crit[0] == ("inclusion", "Adults aged 18 to 75")
    assert ("exclusion", "Pregnancy") in crit

    def mk(kind, status):
        return CriterionAssessment(criterion="x", kind=kind, status=status)

    assert verdict_from([mk("inclusion", "met"), mk("exclusion", "not_met")]) == "possibly_eligible"
    assert verdict_from([mk("inclusion", "met"), mk("exclusion", "met")]) == "likely_ineligible"
    assert verdict_from([mk("inclusion", "unknown")]) == "insufficient_information"


async def test_grounding_negation_and_side_need_the_judge():
    t = Transcript.from_text("[patient] I have no chest pain.\n[patient] My right knee hurts, not the left.")
    note = SOAPNote.model_validate(
        {
            "subjective": [
                {"text": "Patient reports chest pain.", "evidence": [1]},
                {"text": "Pain in the left knee.", "evidence": [2]},
                {"text": "Denies chest pain.", "evidence": [1]},
            ]
        }
    )
    report = await check_note(note, t)  # no judge available: risky sentences are flagged
    assert [c.index for c in report.unsupported] == [0, 1]
    assert "negation" in report.checks[0].reason and "side" not in report.checks[2].reason


def test_lenient_evidence_parsing():
    note = SOAPNote.model_validate({"plan": [{"text": "x", "evidence": ["U3", "[U4]", 5, "5"]}]})
    assert note.plan[0].evidence == [3, 4, 5]
    assert note.subjective == []


def test_rouge():
    assert rouge_n("the knee hurts", "the knee hurts", 1) == 1.0
    assert 0 < rouge_l("right knee pain two weeks", "knee pain for two weeks") < 1


def test_guardrails_follow_pipeline_settings(monkeypatch):
    from types import SimpleNamespace

    from consult_to_note import guard

    monkeypatch.setenv("NIM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("C2N_MODEL_FAST", "local/fast-model")
    main = SimpleNamespace(
        type="main", model="default", parameters={"chat_template_kwargs": {"enable_thinking": False}}
    )
    other = SimpleNamespace(type="embeddings", model="emb", parameters={})
    guard._apply_settings(SimpleNamespace(models=[main, other]))
    assert main.model == "local/fast-model"
    assert main.parameters["base_url"] == "http://localhost:8000/v1"
    assert main.parameters["chat_template_kwargs"] == {"enable_thinking": False}
    assert other.model == "emb"
