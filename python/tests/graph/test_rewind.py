import pytest
from timbal import Flow
from timbal.state.context import RunContext
from timbal.state.savers import InMemorySaver


@pytest.mark.asyncio
async def test_invalid_run_id():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    context = RunContext(id="1")

    prompt = "Hello my name is David"
    await flow.complete(context=context, prompt=prompt)

    # This doesn't raise an error, but the final length of the snapshots will be 1.
    prompt = "What is my name?"
    await flow.complete(context=context, prompt=prompt)
    assert len(flow.state_saver.snapshots) == 1


@pytest.mark.asyncio
async def test_no_parent_id():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.output["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_parent_id():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    flow_output_event = await flow.complete(prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(parent_id=flow_output_event.run_id),
        prompt=prompt,
    ):
        if event.type == "FLOW_OUTPUT":
            assert "david" in event.output["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_different_parent_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(parent_id="wrong_parent_id"),
        prompt=prompt,
    ):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.output["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_rewind():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello"
    flow_output_event_1 = await flow.complete(prompt=prompt)

    prompt = "My name is David"
    await flow.complete(
        context=RunContext(parent_id=flow_output_event_1.run_id),
        prompt=prompt,
    )

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(parent_id=flow_output_event_1.run_id),
        prompt=prompt,
    ):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.output["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_rewind_2():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "message")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello"
    flow_output_event_1 = await flow.complete(prompt=prompt)

    prompt = "My name is David"
    flow_output_event_2 = await flow.complete(
        context=RunContext(parent_id=flow_output_event_1.run_id),
        prompt=prompt,
    )

    prompt = "What is my name?"
    flow_output_event_3 = await flow.complete(
        context=RunContext(parent_id=flow_output_event_2.run_id), 
        prompt=prompt,
    )
    assert "david" in flow_output_event_3.output["message"].content[0].text.lower()

    prompt = "My name is John"
    flow_output_event_4 = await flow.complete(
        context=RunContext(parent_id=flow_output_event_1.run_id),
        prompt=prompt,
    )

    prompt = "What is my name?"
    flow_output_event_5 = await flow.complete(
        context=RunContext(parent_id=flow_output_event_4.run_id),
        prompt=prompt,
    )
    assert "john" in flow_output_event_5.output["message"].content[0].text.lower()
