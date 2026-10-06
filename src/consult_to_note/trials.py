"""Clinical trial pre-screening from the consultation (research demo, not a medical device).

1. Search ClinicalTrials.gov (API v2, no key needed) for recruiting trials for the patient's conditions.
2. Split each trial's eligibility text into inclusion and exclusion criteria.
3. Ask the fast model to assess each criterion against the transcript, citing utterances.
The verdict is deliberately conservative: anything unclear stays "insufficient_information" and a
human research coordinator decides.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx
from pydantic import BaseModel

from .llm import LLM, complete_structured
from .prompts import SYSTEM_SCRIBE, TRIAL_CRITERIA
from .schemas import CriterionAssessment, Extraction, Transcript, TrialAssessment

log = logging.getLogger(__name__)
CTGOV_URL = "https://clinicaltrials.gov/api/v2/studies"


async def search_trials(
    condition: str, *, location: str | None = None, page_size: int = 5, client: httpx.AsyncClient | None = None
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "query.cond": condition,
        "filter.overallStatus": "RECRUITING",
        "pageSize": page_size,
        "format": "json",
    }
    if location:
        params["query.locn"] = location
    own = client is None
    client = client or httpx.AsyncClient(timeout=30)
    try:
        resp = await client.get(CTGOV_URL, params=params)
        resp.raise_for_status()
        return resp.json().get("studies", [])
    finally:
        if own:
            await client.aclose()


def parse_study(study: dict[str, Any]) -> dict[str, Any]:
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    elig = proto.get("eligibilityModule", {})
    nct = ident.get("nctId", "")
    return {
        "nct_id": nct,
        "title": ident.get("briefTitle", ""),
        "criteria_text": elig.get("eligibilityCriteria", ""),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex", "ALL"),
        "url": f"https://clinicaltrials.gov/study/{nct}" if nct else "",
    }


_HEADING = re.compile(r"(?im)^\s*(?:key\s+|main\s+|major\s+)?(inclusion|exclusion)\s+criteria\b[^\n]*$")
_BULLET = re.compile(r"^(\s*)(?:(?:[-*\u2022\u25e6o]|\d+[.)]|[a-z][.)])\s+)?")


def parse_criteria(text: str) -> list[tuple[str, str]]:
    """All criteria as [(kind, text)], inclusion first.

    Handles "Key Inclusion Criteria:" style headings, and folds nested sub-bullets under a parent line that
    ends with ":" (e.g. "One of the following:") into one criterion: "One of the following: (a; b; c)".
    """
    pieces = _HEADING.split(text)
    out: list[tuple[str, str]] = []
    kind = "inclusion"
    # re.split with one group gives [before, kind1, body1, kind2, body2, ...]
    sections = [("inclusion", pieces[0])] + [(pieces[i].lower(), pieces[i + 1]) for i in range(1, len(pieces) - 1, 2)]
    for kind, body in sections:
        parent: list | None = None
        parent_indent = -1
        for line in body.splitlines():
            if not line.strip():
                continue
            m = _BULLET.match(line)
            indent = len(m.group(1).replace("\t", "    "))
            item = line[m.end() :].strip()
            if len(item) <= 3:
                continue
            if parent is not None and indent > parent_indent:
                parent[2].append(item)
                continue
            if parent is not None:
                out.append((parent[0], f"{parent[1]} ({'; '.join(parent[2])})" if parent[2] else parent[1]))
                parent = None
            if item.endswith(":"):
                parent, parent_indent = [kind, item, []], indent
            else:
                out.append((kind, item))
        if parent is not None:
            out.append((parent[0], f"{parent[1]} ({'; '.join(parent[2])})" if parent[2] else parent[1]))
    return [c for c in out if c[0] == "inclusion"] + [c for c in out if c[0] == "exclusion"]


def split_criteria(text: str, max_items: int = 12) -> list[tuple[str, str]]:
    """Criteria limited to `max_items` (inclusion first, then exclusion) to bound cost and latency."""
    out = parse_criteria(text)
    inc = [c for c in out if c[0] == "inclusion"][: max_items // 2 + max_items % 2]
    exc = [c for c in out if c[0] == "exclusion"][: max_items // 2]
    return inc + exc


class _CriteriaList(BaseModel):
    criteria: list[CriterionAssessment]


def verdict_from(criteria: list[CriterionAssessment]) -> str:
    inclusion = [c for c in criteria if c.kind == "inclusion"]
    exclusion = [c for c in criteria if c.kind == "exclusion"]
    if any(c.status == "not_met" for c in inclusion) or any(c.status == "met" for c in exclusion):
        return "likely_ineligible"
    if inclusion and all(c.status == "met" for c in inclusion) and all(c.status == "not_met" for c in exclusion):
        return "possibly_eligible"
    return "insufficient_information"


async def assess_trial(
    trial: dict[str, Any], transcript: Transcript, extraction: Extraction, llm: LLM
) -> TrialAssessment:
    criteria = split_criteria(trial["criteria_text"])
    criteria_txt = "\n".join(f"- ({kind}) {text}" for kind, text in criteria)
    facts = {
        "demographics": extraction.demographics.model_dump(),
        "conditions": [c.model_dump() for c in extraction.conditions],
        "medications": [m.model_dump() for m in extraction.medications],
        "observations": [o.model_dump() for o in extraction.observations],
        "trial_age_range": [trial.get("minimum_age"), trial.get("maximum_age")],
        "trial_sex": trial.get("sex"),
    }
    result = await complete_structured(
        llm,
        [
            {"role": "system", "content": SYSTEM_SCRIBE},
            {
                "role": "user",
                "content": TRIAL_CRITERIA.format(
                    facts=json.dumps(facts),
                    transcript=transcript.to_prompt(),
                    title=trial["title"],
                    nct_id=trial["nct_id"],
                    criteria=criteria_txt,
                ),
            },
        ],
        _CriteriaList,
        role="fast",
        step="trial_criteria",
    )
    return TrialAssessment(
        nct_id=trial["nct_id"],
        title=trial["title"],
        url=trial.get("url", ""),
        verdict=verdict_from(result.criteria),
        criteria=result.criteria,
    )


async def prescreen(
    transcript: Transcript, extraction: Extraction, llm: LLM, *, max_trials: int = 3, location: str | None = None
) -> list[TrialAssessment]:
    conditions = [c.name for c in extraction.conditions if c.status != "resolved"]
    if not conditions:
        return []
    studies = await search_trials(conditions[0], location=location, page_size=max_trials)
    results = []
    for study in studies[:max_trials]:
        try:
            results.append(await assess_trial(parse_study(study), transcript, extraction, llm))
        except Exception as err:  # one bad trial must not hide the others
            info = parse_study(study)
            results.append(
                TrialAssessment(
                    nct_id=info["nct_id"],
                    title=info["title"],
                    url=info["url"],
                    verdict="insufficient_information",
                    criteria=[],
                )
            )
            log.warning("trial %s could not be assessed: %s", info["nct_id"], err)
    return results
