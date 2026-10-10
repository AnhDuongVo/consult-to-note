"""Full consult-to-note workflow + agenteval, using synthetic fixtures and scripted models.

Run from an environment with both repositories installed and pytest available.
Pass the consult-to-note checkout path as the first argument.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

from agenteval.clinical.metrics import citation_integrity_rate, number_accuracy
from agenteval.clinical.schemas import Claim, NumberClaim, Record

from consult_to_note.agent import PipelineOptions, run_pipeline
from consult_to_note.llm import FakeLLM
from consult_to_note.schemas import Transcript


async def main():
    repo = Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("synthetic_fixture", repo / "tests/conftest.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    print("FULL WORKFLOW · SYNTHETIC INPUTS · SCRIPTED MODELS · NO LIVE API")
    print("\nSOURCE TRANSCRIPT\n" + fixture.DIALOGUE)
    records = []
    for budget in (0, 1):
        model = FakeLLM(
            {
                "extract": fixture.EXTRACTION,
                "draft": fixture.BAD_NOTE,
                "revise": fixture.GOOD_NOTE,
                "ground_judge": {"supported": True, "reason": "deliberately permissive scripted judge"},
            }
        )
        result = await run_pipeline(
            Transcript.from_text(fixture.DIALOGUE), model, PipelineOptions(max_revisions=budget)
        )
        print(f"\nREVISION BUDGET {budget} · actual LangGraph workflow")
        print("Model steps:", ", ".join(c.step for c in model.calls))
        print("Flags:", sorted(result.flagged()), "revisions:", result.revisions)
        print(result.note.to_markdown(flags=result.flagged()))
        print("FHIR status:", result.fhir["entry"][0]["resource"]["status"])
        dose = 800 if budget == 0 else 400
        assert ("800" in result.note.to_markdown()) == (budget == 0)
        assert bool(result.flagged()) == (budget == 0)
        records.append(
            Record(
                id=f"budget-{budget}",
                task="consult-to-note",
                sources=["U4"],
                claims=[
                    Claim(
                        text=f"Ibuprofen {dose} mg",
                        citations=["U4"],
                        numbers=[NumberClaim(value=dose, source_id="U4", source_value=400)],
                    )
                ],
            )
        )
    print("\nAGENTEVAL · same selected dose before and after revision")
    for record in records:
        print(
            json.dumps(
                {
                    "id": record.id,
                    "citation_integrity_rate": citation_integrity_rate([record]),
                    "selected_dose_number_accuracy": number_accuracy([record], rel_tol=0, abs_tol=0),
                }
            )
        )
    print("Citation existence remains 1.0 despite the wrong dose; number accuracy catches it.")
    print("Limits: one synthetic consultation, scripted extraction/drafting/judging, selected-dose evaluation.")
    print("This demonstrates full software control flow, not live model quality or clinical validation.")


if __name__ == "__main__":
    asyncio.run(main())
