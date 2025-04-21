from typing import Any

import pytest
from timbal import Flow
from timbal.types import Field


def identity_handler(x: Any = Field(default=None)) -> Any:
    return x


@pytest.mark.asyncio
async def test_flow_conditions():
    flow = (
        Flow(id="test_flow_conditions")
        .add_step("step_1", identity_handler)
        .add_step("step_2", identity_handler)
        .add_link("step_1", "step_2", condition="{{step_1.return}} * {{x}} > 5")
        .set_data_map("step_1.x", "x")
    )

    executed_steps = set()
    async for event in flow.run(x=1):
        if event.type == "START": 
            executed_steps.add(event.path)
    assert "test_flow_conditions.step_2" not in executed_steps

    executed_steps.clear()
    async for event in flow.run(x=3):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_flow_conditions.step_2" in executed_steps


@pytest.mark.asyncio
async def test_flow_positive_negative_conditions():
    flow = (
        Flow(id="test_flow_positive_negative_conditions")
        .add_step("step_1", identity_handler)
        .add_step("step_2", identity_handler)
        .add_step("step_3", identity_handler)
        .add_link("step_1", "step_2", condition="{{step_1.return}} > 0")
        .add_link("step_1", "step_3", condition="{{step_1.return}} <= 0")
        .set_data_map("step_1.x", "x")
    )

    # Test positive path
    steps_executed = set()
    async for event in flow.run(x=1):
        if event.type == "START":
            steps_executed.add(event.path)
    assert "test_flow_positive_negative_conditions.step_2" in steps_executed
    assert "test_flow_positive_negative_conditions.step_3" not in steps_executed

    # Test negative path
    steps_executed.clear()
    async for event in flow.run(x=-1):
        if event.type == "START":
            steps_executed.add(event.path)
    assert "test_flow_positive_negative_conditions.step_2" not in steps_executed
    assert "test_flow_positive_negative_conditions.step_3" in steps_executed


@pytest.mark.asyncio
async def test_flow_string_conditions():
    flow = (
        Flow(id="test_flow_string_conditions")
        .add_step("input", identity_handler)
        .add_step("path_a", identity_handler)
        .add_step("path_b", identity_handler)
        .add_link("input", "path_a", condition="{{input.return}} == 'test'")
        .add_link("input", "path_b", condition="{{input.return}} != 'test'")
        .set_data_map("input.x", "input_str")
    )

    executed_steps = set()
    async for event in flow.run(input_str="test"):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_flow_string_conditions.path_a" in executed_steps
    assert "test_flow_string_conditions.path_b" not in executed_steps


@pytest.mark.asyncio
async def test_flow_list_conditions():
    flow = (
        Flow(id="test_flow_list_conditions")
        .add_step("list_step", identity_handler)
        .add_step("int_step", identity_handler)
        .add_link("list_step", "int_step", condition="{{list_step.return}} == ['TEST']")
        .set_data_map("list_step.x", "input_list")
    )

    executed_steps = set()
    async for event in flow.run(input_list=["TEST"]):
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_flow_list_conditions.int_step" in executed_steps


@pytest.mark.asyncio
async def test_many_to_one():
    flow = (
        Flow(id="test_many_to_one")
        .add_step("step_1", identity_handler)
        .add_step("step_2", identity_handler)
        .add_step("step_3", identity_handler)
        .add_link("step_1", "step_3", condition="1 > 0")
        .add_link("step_2", "step_3", condition="1 < 0")
    )

    executed_steps = set()
    async for event in flow.run():
        if event.type == "START":
            executed_steps.add(event.path)
    assert "test_many_to_one.step_3" in executed_steps
