# Validation evidence — 9 October 2026

`pip install -e ".[dev]"` succeeded in a fresh virtual environment for this repository, independently of the other projects. Tests ran on Python 3.12.14 / Darwin arm64 CPU. Other Python versions in CI have not been executed locally here.

| Check | Result |
|---|---|
| Offline test suite | 26 passed |
| Ruff lint | Passed |
| Ruff formatting | Passed |
| Mocked/simulated integration | Tested within the scope below |
| Live NVIDIA hosted endpoint | Not executed |
| Self-hosted GPU endpoint | Not executed |
| Clinical/scientific domain validation | Not completed |

## Tested scope

Deterministic numeric, unit, citation, polarity and laterality failures cannot be overridden by a judge; ambiguous paraphrases can still use a judge. Existing graph/FHIR/realtime/mocked-client tests.

The GitHub workflows have been added or retained, but their remote execution has not been verified after these changes. Unit tests establish behavior on fixtures; they do not establish semantic or clinical correctness.

## NAT configuration

`nat validate --config_file configs/workflow.yml` passed with the installed toolkit. This checks configuration/schema validity and plugin discovery; it does not execute a live model workflow or verify the example model IDs.

## Small offline evaluation

| Dataset / test | Result |
|---|---|
| Bundled synthetic knee consultation | 2 scripted notes × 6 sentences |
| Correct note screen pass rate | 100% |
| Planted wrong dose with a permissive model judge | 1 detected, cannot be overridden |
| Median time to check the good/bad pair, 30 repetitions | 0.78 ms on the CPU above |
| Live ACI-Bench / latency benchmark | Not run |

[Machine-readable results](offline-evaluation.json). This is a checker microbenchmark on one fixture, not a clinical evaluation or model-generation latency result. It does not test wrong medication/value associations exhaustively.
