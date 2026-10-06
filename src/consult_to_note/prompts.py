"""Prompts for each pipeline step. Kept in one place so they are easy to read, version and evaluate."""

SYSTEM_SCRIBE = (
    "You are a careful clinical documentation assistant supporting a physician. "
    "You only document what was said in the consultation. You never invent findings, doses, values or diagnoses. "
    "Every statement you produce must cite the transcript utterance ids (the numbers in [U#]) that support it."
)

EXTRACT = """Extract the clinically relevant facts from this doctor-patient consultation.

Rules:
- Use third person, concise clinical language ("Patient reports ...").
- Cite the utterance ids that support each item in `evidence` (numbers only, e.g. [3, 4]).
- Mark explicitly denied findings with negated=true ("denies chest pain").
- Only include medications, doses, vitals and results that were actually said.
- If age or sex are not stated, leave them null / "unknown".

Transcript:
{transcript}
"""

DRAFT = """Write a SOAP note for this consultation using the extracted facts and the transcript.

Sections:
- subjective: chief complaint, history of present illness, relevant history, medications, allergies, social and family history
- objective: vitals, examination findings, results mentioned in the visit
- assessment: the physician's assessment and diagnoses as stated or clearly implied by the physician
- plan: tests, treatments, medication changes, referrals, follow-up, patient education

Rules:
- One fact per sentence. Each sentence must have a non-empty `evidence` list of utterance ids.
- Do not add anything that is not supported by the cited utterances. No new doses, numbers or diagnoses.
- Leave a section empty rather than guessing.

Extracted facts (JSON):
{extraction}

Transcript:
{transcript}
"""

REVISE = """Some sentences in your SOAP note are not supported by the utterances they cite.
Fix ONLY those sentences: correct the citation, correct the wording so it matches what was said,
or remove the sentence. Keep every other sentence unchanged. Return the full corrected note.

Problems:
{problems}

Current note (JSON):
{note}

Transcript:
{transcript}
"""

JUDGE = """Does the cited transcript evidence fully support the note sentence?
Answer supported=true only if every clinical fact in the sentence (including numbers, doses, laterality,
negations and timing) is stated in the evidence. Paraphrasing is fine.

Note sentence: {sentence}

Cited evidence:
{evidence}
"""

LABEL_SPEAKERS = """These transcript segments come from a doctor-patient consultation, but the speakers are unknown.
Label each segment as "doctor" or "patient" based on content and conversational flow.

Segments:
{segments}
"""

TRIAL_CRITERIA = """Assess whether this patient may be eligible for the clinical trial, criterion by criterion.

For each criterion return status:
- "met" if the transcript clearly shows it is satisfied (for exclusion criteria: the exclusion applies),
- "not_met" if the transcript clearly shows it is not satisfied (for exclusion criteria: the exclusion does not apply),
- "unknown" if the consultation does not contain enough information.
Cite utterance ids as evidence. Be conservative: prefer "unknown" over guessing.

Patient facts (JSON):
{facts}

Transcript:
{transcript}

Trial: {title} ({nct_id})
Criteria:
{criteria}
"""
