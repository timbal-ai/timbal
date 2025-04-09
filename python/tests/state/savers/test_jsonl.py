from pathlib import Path

import pytest
from timbal import Agent, Flow
from timbal.state import RunContext
from timbal.state.savers import JSONLSaver


def test_jsonl_init():
    test_file_dir = Path(__file__).parent
    jsonl_path = test_file_dir / "test_jsonl_init.jsonl"
    JSONLSaver(jsonl_path)

    assert jsonl_path.exists()
    assert jsonl_path.stat().st_size == 0

    # Cleanup
    jsonl_path.unlink()


@pytest.mark.asyncio
async def test_jsonl_put():
    test_file_dir = Path(__file__).parent
    jsonl_path = test_file_dir / "test_jsonl_put.jsonl"
    jsonl_saver = JSONLSaver(jsonl_path)

    with open(jsonl_path) as f:
        jsonl_lines_cnt = len(f.readlines())

    flow = (
        Flow()
        .add_llm(model="gpt-4o-mini", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "response")
        .compile(state_saver=jsonl_saver)
    )

    await flow.complete(prompt="Hello, world!")

    # Assert there is exactly one line in the file
    with open(jsonl_path) as f:
        assert len(f.readlines()) == jsonl_lines_cnt + 1

    # Cleanup
    jsonl_path.unlink()


@pytest.mark.asyncio
async def test_jsonl_flow():
    test_file_dir = Path(__file__).parent
    jsonl_path = test_file_dir / "test_jsonl.jsonl"
    jsonl_saver = JSONLSaver(jsonl_path)

    flow = (
        Flow()
        .add_llm(model="gpt-4o-mini", memory_id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "response")
        .compile(state_saver=jsonl_saver)
    )

    flow_output_event = await flow.complete(prompt="My name is David")

    flow_output_event = await flow.complete(
        context=RunContext(parent_id=flow_output_event.run_id),
        prompt="What is my name?"
    )

    flow_output_text = flow_output_event.output["response"].content[0].text
    assert "david" in flow_output_text.lower(), "Response should mention that my name is David"

    # Cleanup
    jsonl_path.unlink()


@pytest.mark.asyncio
async def test_jsonl_agent():
    test_file_dir = Path(__file__).parent
    jsonl_path = test_file_dir / "test_jsonl.jsonl"
    jsonl_saver = JSONLSaver(jsonl_path)

    agent = Agent(state_saver=jsonl_saver)

    flow_output_event = await agent.complete(prompt="My name is David")

    flow_output_event = await agent.complete(
        context=RunContext(parent_id=flow_output_event.run_id),
        prompt="What is my name?"
    )

    flow_output_text = flow_output_event.output.content[0].text
    assert "david" in flow_output_text.lower(), "Response should mention that my name is David"

    # Cleanup
    jsonl_path.unlink()
