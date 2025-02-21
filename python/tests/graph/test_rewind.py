import pytest
from timbal import Flow
from timbal.state.savers import InMemorySaver


@pytest.mark.asyncio
async def test_invalid_run_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_id="1", prompt=prompt)

    with pytest.raises(ValueError):
        prompt = "What is my name?"
        await flow.complete(run_id="1", prompt=prompt)


@pytest.mark.asyncio
async def test_no_parent_id_no_group_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_id="1", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_id="2", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_no_parent_id_different_group_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_id="1", run_group_id="group1", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_id="2", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_no_parent_id_same_group_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_group_id="group1", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_group_id="group1", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_different_parent_id_same_group_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_id="1", run_group_id="group1", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_id="2", run_parent_id="wrong_parent_id", run_group_id="group1", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_same_parent_id_same_group_id():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello my name is David"
    await flow.complete(run_id="1", run_group_id="group1", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_id="2", run_parent_id="1", run_group_id="group1", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_rewind():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello"
    await flow.complete(run_id="1", prompt=prompt)

    prompt = "My name is David"
    await flow.complete(run_id="2", prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(run_id="3", run_parent_id="1", prompt=prompt):
        if event.type == "FLOW_OUTPUT":
            assert "david" not in event.outputs["message"].content[0].text.lower()


@pytest.mark.asyncio
async def test_rewind_2():

    flow = (
        Flow(id="flow")
        .add_llm(id="llm", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    prompt = "Hello"
    await flow.complete(run_id="1", prompt=prompt)

    prompt = "My name is David"
    await flow.complete(run_id="2", prompt=prompt)

    prompt = "What is my name?"
    response = await flow.complete(run_id="3", prompt=prompt)
    assert "david" in response["message"].content[0].text.lower()

    prompt = "My name is John"
    await flow.complete(run_id="4", run_parent_id="1", prompt=prompt)

    prompt = "What is my name?"
    response = await flow.complete(run_id="5", prompt=prompt)
    assert "john" in response["message"].content[0].text.lower()
