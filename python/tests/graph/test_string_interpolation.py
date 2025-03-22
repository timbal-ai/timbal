from typing import Any

import pytest
from timbal import Flow
from timbal.types import Field


def identity_handler(x: Any = Field(default=None)) -> Any:
    return x


# TODO Test interpolating LLM messages.


@pytest.mark.asyncio
async def test_string_interpolation():
    flow = (
        Flow()
        .add_step("step_1", identity_handler)
        .add_step("step_2", identity_handler)
        .add_step("step_3", identity_handler)
        .add_link("step_1", "step_3")
        .add_link("step_2", "step_3")
        .set_data_map("step_1.x", "x1")
        .set_data_map("step_2.x", "x2")
        .set_data_value("step_3.x",  "{{step_1.return}} {{step_2.return}}")
        .set_output("step_3.x", "step_3_x")
        .set_output("step_3.return", "step_3_return")
    )

    x1 = "Hello"
    x2 = "World"

    async for event in flow.run(x1=x1, x2=x2):
        if event.type == "FLOW_OUTPUT":
            flow_output = event.output
            assert flow_output["step_3_x"] == f"{x1} {x2}"
            assert flow_output["step_3_return"] == f"{x1} {x2}"
