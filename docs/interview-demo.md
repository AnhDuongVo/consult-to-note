# Interview walkthrough: full workflow and selected-dose evaluation

Install consult-to-note and agenteval in the same Python environment, with consult-to-note's dev dependencies. From the consult-to-note repository root:

```bash
python examples/interview_walkthrough.py .
```

The script loads the bundled synthetic test fixture, executes the actual LangGraph pipeline with zero and one revision, and passes the same selected dose to agenteval. It deliberately supplies a permissive scripted judge: the 800 mg draft must stay flagged against 400 mg evidence until the scripted revision repairs it. Assertions check that behavior. FHIR output stays preliminary.

Walk through `src/consult_to_note/agent.py`, `src/consult_to_note/grounding.py`, the fixture in `tests/conftest.py`, and the example script. Pause at source evidence, deterministic failure, revision and metrics. Citation integrity stays 1.0 while selected-dose numerical accuracy changes from 0.0 to 1.0, illustrating why citation existence alone is insufficient.

This exercises the full software workflow with scripted models, not live model generation, all-claim evaluation or clinical validation. The separate slower edited video gives reading time; the script can be run live.
