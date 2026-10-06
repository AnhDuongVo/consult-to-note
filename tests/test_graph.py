import uuid

from consult_to_note.agent import PipelineOptions, PipelineResult, build_graph, run_pipeline


async def test_pipeline_repairs_hallucination(transcript, fake_llm):
    result = await run_pipeline(transcript, fake_llm, PipelineOptions(max_revisions=1))
    steps = [c.step for c in fake_llm.calls]
    assert steps[:2] == ["extract", "draft"] and "revise" in steps
    assert result.revisions == 1
    assert result.grounding.support_rate == 1.0
    assert "800" not in result.note.to_markdown()
    assert result.fhir["entry"][0]["resource"]["status"] == "preliminary"
    assert {"extract", "draft", "ground", "revise", "fhir"} <= set(result.timings)


async def test_no_revision_budget_keeps_flag(transcript, fake_llm):
    result = await run_pipeline(transcript, fake_llm, PipelineOptions(max_revisions=0))
    assert result.revisions == 0
    assert result.flagged() == {("subjective", 2)}
    assert "[CHECK" in result.note.to_markdown(flags=result.flagged())


async def test_human_review_interrupt_then_approve(transcript, fake_llm):
    graph = build_graph(fake_llm, PipelineOptions(human_review=True))
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    await graph.ainvoke({"transcript": transcript.model_dump(), "timings": {}}, config)
    snapshot = await graph.aget_state(config)
    assert snapshot.next == ("finalize",)  # paused for the clinician
    assert snapshot.values["fhir"]["entry"][0]["resource"]["status"] == "preliminary"

    await graph.aupdate_state(config, {"approved": True, "reviewer": "Dr. Test"})
    final = await graph.ainvoke(None, config)
    result = PipelineResult.from_state(final)
    composition = result.fhir["entry"][0]["resource"]
    assert composition["status"] == "final"
    assert result.fhir["entry"][2]["resource"]["name"][0]["text"] == "Dr. Test"
