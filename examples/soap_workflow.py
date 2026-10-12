"""Run: streamlit run examples/soap_workflow.py (install .[demo] first)."""

import asyncio
import html
import json

import streamlit as st
from soap_fixture import BAD_NOTE, DIALOGUE, EXTRACTION, GOOD_NOTE

from consult_to_note.agent import PipelineOptions, run_pipeline
from consult_to_note.config import Settings
from consult_to_note.llm import FakeLLM, NIMClient
from consult_to_note.schemas import Transcript

st.set_page_config(page_title="consult-to-note | Transcript to SOAP", layout="wide")
st.markdown(
    """<style>
.block-container{padding:1.4rem 2rem;max-width:1500px}h1{font-size:2rem!important}h3{font-size:1.12rem!important;margin-top:.5rem!important}
.soap-line{padding:5px 9px;margin:5px 0;border-left:3px solid #b3c1c5;background:#f6f7f7;color:#243137;font-size:16px}.flag{border-color:#b8563a;background:#fff1ec}.cite{color:#86614e;font-size:13px}.stCaption{font-size:14px}textarea:disabled{-webkit-text-fill-color:#344054!important;color:#344054!important;opacity:1!important}
</style>""",
    unsafe_allow_html=True,
)
st.title("consult-to-note: transcript → SOAP note")
backend = st.selectbox("Generation backend", ["Scripted model responses · offline", "Configured model endpoint · live"])
is_live = backend.endswith("live")
st.caption(
    "Synthetic consultation · Actual extract → draft → check → revise → FHIR pipeline · "
    + ("Live model calls" if is_live else "Scripted extraction, drafting and judging; no live model generation")
)
left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("Input · consultation transcript")
    text = st.text_area("Consultation text", value=DIALOGUE, height=280, disabled=not is_live)
    transcript = Transcript.from_text(text)
    with st.expander("Transcript evidence IDs", expanded=False):
        st.text(transcript.to_prompt())
    generate = st.button("Generate SOAP draft", type="primary", use_container_width=True)
    repair = st.button(
        "Run with automatic revision",
        use_container_width=True,
        disabled=st.session_state.get("result_key") != (backend, text),
    )
    st.caption(
        "The offline draft deliberately contains an 800 mg error. The source says 400 mg. "
        "Automatic revision uses the same pipeline with one revision allowed."
    )
    if generate or repair:
        try:
            if not transcript.utterances:
                raise ValueError("Enter a consultation transcript first.")
            if is_live:
                model = NIMClient(Settings())
            else:
                if text != DIALOGUE:
                    raise ValueError("Scripted responses apply only to the bundled synthetic consultation.")
                model = FakeLLM(
                    {
                        "extract": EXTRACTION,
                        "draft": BAD_NOTE,
                        "revise": GOOD_NOTE,
                        "ground_judge": {
                            "supported": True,
                            "reason": "Scripted judge; deterministic failures remain flagged",
                        },
                    }
                )
            with st.spinner("Running SOAP workflow…"):
                result = asyncio.run(run_pipeline(transcript, model, PipelineOptions(max_revisions=1 if repair else 0)))
            st.session_state.result = result
            st.session_state.result_key = (backend, text)
            st.rerun()
        except Exception as exc:
            st.error(
                f"Workflow did not complete ({type(exc).__name__}). Check the local endpoint configuration and logs."
            )
with right:
    st.subheader("Output · SOAP note")
    if st.session_state.get("result_key") != (backend, text):
        st.info("Run the workflow to see Subjective, Objective, Assessment and Plan here.")
    else:
        result = st.session_state.result
        flags = result.flagged()
        st.caption(f"{len(flags)} flagged sentence(s) · {result.revisions} revision(s) · FHIR status: preliminary")
        for section in result.note.SECTIONS:
            st.markdown(f"### {section.capitalize()}")
            sentences = getattr(result.note, section)
            if not sentences:
                st.caption("Nothing documented")
            for i, sentence in enumerate(sentences):
                cls = "soap-line flag" if (section, i) in flags else "soap-line"
                refs = ", ".join(f"U{n}" for n in sentence.evidence)
                label = " · CHECK: unsupported" if (section, i) in flags else ""
                st.markdown(
                    f'<div class="{cls}">{html.escape(sentence.text)} <span class="cite">[{refs}]{label}</span></div>',
                    unsafe_allow_html=True,
                )
        with st.expander("Inspect citations and check findings"):
            for check in result.grounding.checks:
                st.markdown(f"**{check.section.capitalize()}:** {check.sentence}")
                st.caption(f"{check.method}: {check.reason}")
            st.text(result.transcript.to_prompt())
        st.download_button("Download SOAP note (.md)", result.note.to_markdown(flags=flags), "note.md")
        st.download_button("Download preliminary FHIR (.json)", json.dumps(result.fhir, indent=2), "fhir_bundle.json")
st.caption(
    "Research prototype. Synthetic example; checks are incomplete. Clinician approval is not performed in this view."
)
