from typing import Any

import pytest
from timbal import Flow
from timbal.types import Field


def identity_handler(x: Any = Field(default=None)) -> Any:
    return x


@pytest.mark.asyncio
async def test_remove_step():
    flow = (
        Flow(id="test_remove_step")
        .add_step("step_1", identity_handler)
        .add_step("step_2", identity_handler)
        .add_step("step_3", identity_handler)
        .set_data_map("step_1.x", "x1")
        .set_data_map("step_2.x", "x2")
        .set_output("step_1.return", "x1")
        .set_output("step_2.return", "x2")
        .set_data_map("step_3.x", "step_1.return")
    )
    
    flow.remove_step("step_1")

    assert "step_1" not in flow.steps

    # Ensure we're cleaning up links correctly
    assert "step_1-step_2" not in flow.links

    # Ensure we're cleaning up outputs correctly
    assert "x1" not in flow.outputs
    assert "x2" in flow.outputs

    # Ensure we're cleaning up data correctly
    assert "step_1.x" not in flow.data
    assert "step_2.x" in flow.data
    assert "step_3.x" not in flow.data
