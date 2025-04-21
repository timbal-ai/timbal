from typing import Any

import pytest
from timbal import Flow
from timbal.errors import InvalidLinkError
from timbal.types import Field


def identity_handler(x: Any = Field(default=None)) -> Any:
    return x


@pytest.fixture
def flow():
    return (
        Flow(id="test_add_link")
        .add_step("step_1", identity_handler)
        .add_llm("step_2")
        .add_step("step_3", identity_handler)
    )


def test_add_link(flow: Flow):
    flow.add_link("step_1", "step_2")
    assert "step_1-step_2" in flow.links

    # Test invalid steps ids
    with pytest.raises(InvalidLinkError):
        flow.add_link("step_1", "step_4")
    with pytest.raises(InvalidLinkError):
        flow.add_link("step_4", "step_2")


def test_already_linked(flow: Flow):
    flow.add_link("step_1", "step_2")
    flow.add_link("step_2", "step_3")

    with pytest.raises(InvalidLinkError):
        flow.add_link("step_1", "step_3")


def test_cycle(flow: Flow):
    flow.add_link("step_1", "step_2")
    flow.add_link("step_2", "step_3")

    with pytest.raises(InvalidLinkError):
        flow.add_link("step_3", "step_1")

    with pytest.raises(InvalidLinkError):
        flow.add_link("step_3", "step_3")


def test_tool(flow: Flow):
    with pytest.raises(InvalidLinkError):
        flow.add_link("step_1", "step_3", is_tool=True)

    flow.add_link("step_2", "step_3", is_tool=True)
    step_2_tools = flow.data["step_2.tools"].resolve(context_data=None)
    assert all(tool.name == "step_3" for tool in step_2_tools)


def test_tool_result(flow: Flow):
    flow.add_link("step_1", "step_2", is_tool_result=True)

    with pytest.raises(InvalidLinkError):
        flow.add_link("step_2", "step_3", is_tool_result=True)
