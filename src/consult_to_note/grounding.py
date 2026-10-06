"""Grounding check: is every note sentence supported by the utterances it cites?

Two stages, cheapest first:
1. Lexical check (no LLM): numbers must match, content words must overlap, and negation and
   laterality (left/right) must agree. Clear passes stop here, which keeps the check fast and auditable.
2. LLM judge (fast model) for everything else: paraphrases ("dyspnoea" for "short of breath"),
   negation or side mismatches, number mismatches. Without a judge those sentences are flagged.
"""

from __future__ import annotations

import asyncio
import re

from .llm import LLM, complete_structured
from .prompts import JUDGE, SYSTEM_SCRIBE
from .schemas import GroundingCheck, GroundingReport, JudgeVerdict, SOAPNote, Transcript

STOPWORDS = set(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "of",
        "to",
        "in",
        "on",
        "at",
        "for",
        "with",
        "without",
        "by",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "his",
        "her",
        "their",
        "they",
        "them",
        "he",
        "she",
        "patient",
        "patients",
        "pt",
        "reports",
        "reported",
        "report",
        "states",
        "stated",
        "says",
        "said",
        "notes",
        "noted",
        "describes",
        "described",
        "mentions",
        "mentioned",
        "denies",
        "denied",
        "endorses",
        "endorsed",
        "also",
        "currently",
        "today",
        "now",
        "very",
        "some",
        "any",
        "about",
        "over",
        "into",
        "than",
        "then",
        "there",
        "here",
        "which",
        "who",
        "whom",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "per",
        "due",
    ]
)
NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "once": "1",
    "twice": "2",
    "bid": "2",
    "tid": "3",
    "qid": "4",
    "od": "1",
    "qd": "1",
}
NEGATIONS = {"no", "not", "denies", "denied", "deny", "without", "never", "negative", "none", "nor", "absent"}
SIDES = {"left", "right", "bilateral"}
SYNONYMS = {
    "bp": "blood",
    "htn": "hypertension",
    "dm": "diabetes",
    "sob": "breath",
    "hx": "history",
    "qd": "daily",
    "yo": "year",
    "y/o": "year",
    "f/u": "follow",
    "followup": "follow",
    "kilos": "kg",
    "milligrams": "mg",
}
_TOKEN_RE = re.compile(r"[a-z]+(?:/[a-z]+)?|\d+(?:[.,]\d+)?")

LEXICAL_PASS = 0.6
LEXICAL_FAIL = 0.25


def _stem(tok: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if len(tok) > 4 and tok.endswith(suffix):
            return tok[: -len(suffix)]
    return tok


def _tokens(text: str) -> list[str]:
    text = re.sub(r"(\d)[,'](\d{3})\b", r"\1\2", text.lower())  # 1,000 / 1'000 -> 1000
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)  # decimal comma (2,5 mg) -> 2.5
    return _TOKEN_RE.findall(text)


def _numbers(text: str) -> set[str]:
    nums = set()
    for tok in _tokens(text):
        if tok[0].isdigit():
            nums.add(f"{float(tok):g}")
        elif tok in NUMBER_WORDS:
            nums.add(NUMBER_WORDS[tok])
    return nums


def _content(text: str) -> set[str]:
    out = set()
    for tok in _tokens(text):
        if tok[0].isdigit() or tok in STOPWORDS or (len(tok) < 3 and tok not in {"mg", "kg", "bp", "hr"}):
            continue
        out.add(_stem(SYNONYMS.get(tok, tok)))
    return out


def _negated_words(text: str, window: int = 4) -> tuple[set[str], set[str]]:
    """(content words, content words within `window` tokens after a negation such as 'no' or 'denies')."""
    words, negated = set(), set()
    countdown = 0
    # Negation scope ends at a sentence boundary or "but" ("no swelling. I take ibuprofen").
    marked = re.sub(r"[.;!?]+(\s|$)|\bbut\b", " <eos> ", text.lower())
    for tok in _tokens(marked.replace("<eos>", " zzeos ")):
        if tok == "zzeos":
            countdown = 0
            continue
        if tok in NEGATIONS:
            countdown = window
            continue
        stem = _stem(SYNONYMS.get(tok, tok))
        words.add(stem)
        if countdown > 0:
            negated.add(stem)
            countdown -= 1
    return words, negated


def risk_flags(sentence: str, evidence_text: str) -> list[str]:
    """Cheap signals that word overlap alone cannot judge: negation polarity and laterality."""
    s_words, s_neg = _negated_words(sentence)
    e_words, e_neg = _negated_words(evidence_text)
    shared = (s_words & e_words) - STOPWORDS - SIDES
    flips = sorted(w for w in shared if (w in s_neg) != (w in e_neg))
    flags = []
    if flips:
        flags.append(f"negation differs for: {flips}")
    s_sides, e_sides = set(_tokens(sentence)) & SIDES, set(_tokens(evidence_text)) & SIDES
    wrong_side = (s_sides - e_sides) | {x for x in s_sides & e_neg if x not in s_neg}
    if wrong_side:
        flags.append(f"side not in evidence: {sorted(wrong_side)}")
    return flags


def lexical_score(sentence: str, evidence_text: str) -> tuple[float, str]:
    """Return (score in [0, 1], reason). A number missing from the evidence scores 0."""
    missing_numbers = _numbers(sentence) - _numbers(evidence_text)
    if missing_numbers:
        return 0.0, f"numbers not in evidence: {sorted(missing_numbers)}"
    words = _content(sentence)
    if not words:
        return 1.0, "no content words to check"
    evidence_words = _content(evidence_text)
    hit = words & evidence_words
    return len(hit) / len(words), f"{len(hit)}/{len(words)} content words found"


async def check_note(
    note: SOAPNote,
    transcript: Transcript,
    llm: LLM | None = None,
    *,
    pass_threshold: float = LEXICAL_PASS,
    fail_threshold: float = LEXICAL_FAIL,
) -> GroundingReport:
    utt = transcript.by_id()
    checks: list[GroundingCheck] = []
    pending: list[tuple[int, str]] = []  # (index into checks, evidence text) needing the judge

    for section, idx, sent in note.sentences():
        cited = [utt[e] for e in sent.evidence if e in utt]
        if not cited:
            checks.append(
                GroundingCheck(
                    section=section,
                    index=idx,
                    sentence=sent.text,
                    supported=False,
                    score=0.0,
                    method="no_evidence",
                    reason="no valid utterance cited",
                )
            )
            continue
        evidence_text = " ".join(u.text for u in cited)
        score, reason = lexical_score(sent.text, evidence_text)
        flags = risk_flags(sent.text, evidence_text)
        if flags:
            reason = f"{reason}; {'; '.join(flags)}"
        clear_pass = score >= pass_threshold and not flags
        if clear_pass or llm is None:
            # Without a judge: pass only clear cases, everything else is flagged for the clinician.
            supported = clear_pass or (score >= (pass_threshold + fail_threshold) / 2 and not flags)
            checks.append(
                GroundingCheck(
                    section=section,
                    index=idx,
                    sentence=sent.text,
                    supported=supported,
                    score=score,
                    method="lexical",
                    reason=reason,
                )
            )
        else:
            # Paraphrases, negations, laterality and number mismatches all go to the judge.
            checks.append(
                GroundingCheck(
                    section=section,
                    index=idx,
                    sentence=sent.text,
                    supported=False,
                    score=score,
                    method="llm_judge",
                    reason=reason,
                )
            )
            pending.append((len(checks) - 1, "\n".join(f"[U{u.id}] {u.speaker}: {u.text}" for u in cited)))

    async def judge(i: int, evidence: str) -> None:
        verdict = await complete_structured(
            llm,  # type: ignore[arg-type]
            [
                {"role": "system", "content": SYSTEM_SCRIBE},
                {"role": "user", "content": JUDGE.format(sentence=checks[i].sentence, evidence=evidence)},
            ],
            JudgeVerdict,
            role="fast",
            step="ground_judge",
            max_tokens=256,
        )
        checks[i].supported = verdict.supported
        checks[i].reason = f"{checks[i].reason}; judge: {verdict.reason}"

    if pending:
        await asyncio.gather(*(judge(i, ev) for i, ev in pending))
    return GroundingReport(checks=checks)
