import pytest
from timbal import Flow
from timbal.types import Message


@pytest.mark.asyncio
async def test_complete():
    flow = (
        Flow(id="flow")
        .add_llm(id="llm")
        .set_data_map("llm.prompt", "prompt")
        .set_output("response", "llm.return")
    )

    result = await flow.complete(prompt="What is the capital of France?")

    assert "response" in result and isinstance(result["response"], Message)
