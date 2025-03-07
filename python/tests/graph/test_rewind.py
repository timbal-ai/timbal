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
        .set_output("message", "llm.return")
        .compile(state_saver=InMemorySaver())
    )

    context = RunContext(id="1")

    prompt = "Hello my name is David"
    await flow.complete(context=context, prompt=prompt)

    with pytest.raises(ValueError):
        prompt = "What is my name?"
        await flow.complete(context=context, prompt=prompt)


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
    await flow.complete(prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(prompt=prompt):
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
    await flow.complete(context=RunContext(group_id="some_other_group_id"), prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(prompt=prompt):
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
    await flow.complete(prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(prompt=prompt):
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
    await flow.complete(
        context=RunContext(
            id="1",
            group_id="same_group_id",
        ),
        prompt=prompt,
    )

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(
            id="2",
            parent_id="wrong_parent_id",
            group_id="same_group_id",
        ),
        prompt=prompt,
    ):
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
    await flow.complete(
        context=RunContext(
            id="1",
            group_id="same_group_id",
        ),
        prompt=prompt,
    )

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(
            id="2",
            parent_id="1",
            group_id="same_group_id",
        ),
        prompt=prompt,
    ):
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
    await flow.complete(context=RunContext(id="1"), prompt=prompt)

    prompt = "My name is David"
    await flow.complete(context=RunContext(id="2"), prompt=prompt)

    prompt = "What is my name?"
    async for event in flow.run(
        context=RunContext(
            id="3",
            parent_id="1",
        ),
        prompt=prompt,
    ):
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
    await flow.complete(context=RunContext(id="1"), prompt=prompt)

    prompt = "My name is David"
    await flow.complete(context=RunContext(id="2"), prompt=prompt)

    prompt = "What is my name?"
    response = await flow.complete(context=RunContext(id="3"), prompt=prompt)
    assert "david" in response["message"].content[0].text.lower()

    prompt = "My name is John"
    await flow.complete(
        context=RunContext(
            id="4",
            parent_id="1",
        ),
        prompt=prompt,
    )

    prompt = "What is my name?"
    response = await flow.complete(context=RunContext(id="5"), prompt=prompt)
    assert "john" in response["message"].content[0].text.lower()
