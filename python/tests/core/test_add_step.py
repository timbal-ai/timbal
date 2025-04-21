from typing import Any

import pytest
from timbal import Flow
from timbal.errors import StepKeyError
from timbal.core.step import Step
from timbal.state.data import DataMap
from timbal.types import Field


def identity_handler(x: Any = Field(default=None)) -> Any:
    return x


def identity_handler_2(x: Any = Field(default=None)) -> Any:
    return x


@pytest.mark.asyncio
async def test_add_step_with_handler():
    flow = Flow(id="test_add_step_with_handler")

    flow.add_step("step_1", identity_handler)
    assert "step_1.x" not in flow.data

    flow.add_step("step_2", identity_handler, x=1)
    assert flow.data["step_2.x"].resolve() == 1

    with pytest.raises(StepKeyError):
        flow.add_step("step_3", identity_handler, y=1)


@pytest.mark.asyncio
async def test_add_step_with_step():
    flow = Flow(id="test_add_step_with_step")

    step = Step(id="this_will_be_overriden", handler_fn=identity_handler)
    flow.add_step("step_1", step)

    assert "step_1" in flow.steps
    assert "this_will_be_overriden" not in flow.steps


@pytest.mark.asyncio
async def test_add_step_override():
    flow = Flow(id="test_add_step_override")

    flow.add_step("step_1", identity_handler)

    with pytest.raises(ValueError):
        flow.add_step("step_1", identity_handler)


@pytest.mark.asyncio
async def test_add_step_without_id():
    flow = Flow(id="test_add_step_without_id")

    flow.add_step(identity_handler, x=1)

    assert "identity_handler" in flow.steps
    assert flow.data["identity_handler.x"].resolve() == 1


@pytest.mark.asyncio
async def test_automatic_link_creation():
    flow = (
        Flow(id="test_automatic_link_creation")
        .add_step("step_1", identity_handler, x=1)
        .add_step("step_2", identity_handler_2)
        .set_data_map("step_2.x", "step_1.return")
        .set_output("step_2.return", "result")
    )

    result = await flow.complete()

    assert len(flow.links) == 1 and flow.links["step_1-step_2"]
    assert "step_2" in flow.steps
    assert result.output["result"] == 1


@pytest.mark.asyncio
async def test_add_step_data_map():
    flow = (
        Flow(id="test_add_step_data_map")
        .add_step("step_1", identity_handler, x=1)
        .add_step("step_2", identity_handler_2, x=DataMap(key="step_1.return"))
        .set_output("step_2.return", "result")
    )

    result = await flow.complete()

    assert len(flow.links) == 1 and flow.links["step_1-step_2"]
    assert "step_2" in flow.steps
    assert result.output["result"] == 1
