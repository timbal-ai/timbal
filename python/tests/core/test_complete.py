import pytest
from timbal import Flow
from timbal.types import Message


@pytest.mark.asyncio
async def test_complete():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("llm.return", "response")
    )

    flow_output_event = await flow.complete(prompt="What is the capital of France?")
    flow_output = flow_output_event.output

    assert "response" in flow_output and isinstance(flow_output["response"], Message)
