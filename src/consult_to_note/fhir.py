"""Export the note and extracted entities as a FHIR R4 document Bundle.

The Bundle holds a Composition (the SOAP note, one section per part, LOINC-coded) plus Patient,
Practitioner, Condition, MedicationStatement, Observation and AllergyIntolerance resources.
Every derived resource keeps its provenance: the utterance ids it came from (in `note`).
Free-text codes only: mapping to SNOMED CT / RxNorm / LOINC is a deliberate next step, not done here.
"""

from __future__ import annotations

import html
import uuid
from datetime import UTC, datetime
from typing import Any

from .schemas import Extraction, SOAPNote

LOINC = "http://loinc.org"
SECTION_CODES = {
    "subjective": ("61150-9", "Subjective Narrative"),
    "objective": ("61149-1", "Objective Narrative"),
    "assessment": ("51848-0", "Evaluation note"),
    "plan": ("18776-5", "Plan of care note"),
}
MED_STATUS = {"active": "active", "started": "active", "changed": "active", "stopped": "stopped"}
COND_CLINICAL = {"active": "active", "suspected": "active", "resolved": "resolved"}
COND_VERIFY = {"active": "confirmed", "resolved": "confirmed", "suspected": "provisional"}


def _urn() -> str:
    return f"urn:uuid:{uuid.uuid4()}"


def _provenance(evidence: list[int]) -> list[dict[str, str]]:
    return [{"text": "Source utterances: " + ", ".join(f"U{e}" for e in evidence)}] if evidence else []


def _cc(system: str, code: str, display: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code, "display": display}], "text": display}


def _status_cc(system: str, code: str) -> dict[str, Any]:
    return {"coding": [{"system": system, "code": code}]}


def build_bundle(
    note: SOAPNote,
    extraction: Extraction,
    *,
    final: bool = False,
    author: str = "Consult-to-Note draft (AI-assisted)",
    patient_display: str = "Synthetic patient",
) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat(timespec="seconds")
    patient_ref, practitioner_ref = _urn(), _urn()
    sex = extraction.demographics.sex
    patient = {
        "resourceType": "Patient",
        "name": [{"text": patient_display}],
        "gender": sex if sex in {"female", "male"} else "unknown",
    }
    practitioner = {"resourceType": "Practitioner", "name": [{"text": author}]}
    subject = {"reference": patient_ref}

    entries: list[tuple[str, dict[str, Any]]] = []
    for c in extraction.conditions:
        entries.append(
            (
                _urn(),
                {
                    "resourceType": "Condition",
                    "clinicalStatus": _status_cc(
                        "http://terminology.hl7.org/CodeSystem/condition-clinical", COND_CLINICAL[c.status]
                    ),
                    "verificationStatus": _status_cc(
                        "http://terminology.hl7.org/CodeSystem/condition-ver-status", COND_VERIFY[c.status]
                    ),
                    "code": {"text": c.name},
                    "subject": subject,
                    "note": _provenance(c.evidence),
                },
            )
        )
    for m in extraction.medications:
        dosage = " ".join(x for x in (m.dose, m.frequency) if x)
        res: dict[str, Any] = {
            "resourceType": "MedicationStatement",
            "status": MED_STATUS[m.status],
            "medicationCodeableConcept": {"text": m.name},
            "subject": subject,
            "note": _provenance(m.evidence),
        }
        if dosage:
            res["dosage"] = [{"text": dosage}]
        entries.append((_urn(), res))
    for o in extraction.observations:
        entries.append(
            (
                _urn(),
                {
                    "resourceType": "Observation",
                    "status": "preliminary" if not final else "final",
                    "code": {"text": o.name},
                    "subject": subject,
                    "valueString": f"{o.value} {o.unit}".strip() if o.unit else o.value,
                    "note": _provenance(o.evidence),
                },
            )
        )
    for a in extraction.allergies:
        entries.append(
            (
                _urn(),
                {
                    "resourceType": "AllergyIntolerance",
                    "code": {"text": a},
                    "patient": subject,
                },
            )
        )

    sections = []
    for sec in SOAPNote.SECTIONS:
        code, display = SECTION_CODES[sec]
        items = getattr(note, sec)
        lis = "".join(f"<li>{html.escape(s.text)}</li>" for s in items) or "<li>Nothing documented.</li>"
        sections.append(
            {
                "title": sec.capitalize(),
                "code": _cc(LOINC, code, display),
                "text": {
                    "status": "generated",
                    "div": f'<div xmlns="http://www.w3.org/1999/xhtml"><ul>{lis}</ul></div>',
                },
            }
        )
    composition = {
        "resourceType": "Composition",
        "status": "final" if final else "preliminary",
        "type": _cc(LOINC, "11488-4", "Consult note"),
        "subject": subject,
        "date": now,
        "author": [{"reference": practitioner_ref}],
        "title": "Consultation note (SOAP)",
        "section": sections,
    }

    bundle_entries = [
        {"fullUrl": _urn(), "resource": composition},
        {"fullUrl": patient_ref, "resource": patient},
        {"fullUrl": practitioner_ref, "resource": practitioner},
    ]
    bundle_entries += [{"fullUrl": url, "resource": res} for url, res in entries]
    return {
        "resourceType": "Bundle",
        "type": "document",
        "identifier": {"system": "urn:ietf:rfc:3986", "value": _urn()},
        "timestamp": now,
        "entry": bundle_entries,
    }


def validate_bundle(bundle: dict[str, Any]) -> None:
    """Validate the structure with `fhir.resources` (R4B models, compatible with these R4 resources)."""
    from fhir.resources.R4B.bundle import Bundle

    Bundle.model_validate(bundle) if hasattr(Bundle, "model_validate") else Bundle.parse_obj(bundle)
