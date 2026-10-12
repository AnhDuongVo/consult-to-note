import sys
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

repo = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo / "examples"))


def test_soap_demo_generation_revision_and_backend_change():
    app = AppTest.from_file(str(repo / "examples/soap_workflow.py")).run(timeout=30)
    assert not app.exception, app.exception
    assert app.button[1].disabled
    app.button[0].click().run(timeout=30)
    assert not app.exception, app.exception
    r = app.session_state.result
    assert all(getattr(r.note, s) for s in r.note.SECTIONS)
    assert ("subjective", 2) in r.flagged()
    assert "800" in r.note.to_markdown()
    assert r.fhir["entry"][0]["resource"]["status"] == "preliminary"
    app.button[1].click().run(timeout=30)
    assert not app.exception, app.exception
    r = app.session_state.result
    assert not r.flagged()
    assert "400" in r.note.to_markdown() and "800" not in r.note.to_markdown()
    assert r.revisions == 1
    app.selectbox[0].select("Configured model endpoint · live").run()
    assert app.button[1].disabled
    assert len(app.info) == 1
