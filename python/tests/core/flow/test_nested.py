import pytest

from timbal import Flow


@pytest.mark.asyncio
async def test_nested_paths():
    flow_0 = Flow(id="flow_0")
    flow_1 = Flow(id="flow_1").add_step(flow_0)
    flow_2 = Flow(id="flow_2").add_step(flow_1)

    assert len(flow_2.steps) == 1
    flow_2_flow_1 = flow_2.steps["flow_1"]
    assert flow_2_flow_1.path == "flow_2.flow_1"

    assert len(flow_2_flow_1.steps) == 1
    assert flow_2_flow_1.steps["flow_0"].path == "flow_2.flow_1.flow_0"
