import asyncio
from collections.abc import AsyncGenerator, Generator

import pytest
from timbal import Tool, Workflow
from timbal.state import get_run_context
from timbal.types.events import OutputEvent

from .conftest import (
    async_handler,
    sync_handler,
)

# ==============================================================================
# Test Handler Functions for Workflow Steps
# ==============================================================================

def step1_handler(x: str) -> str:
    """First step in workflow."""
    return f"step1:{x}"


def step2_handler(x: str, step1_result: str) -> str:
    """Second step that depends on step1."""
    return f"step2:{x}:{step1_result}"


def step3_handler(x: str) -> str:
    """Third step that runs independently."""
    return f"step3:{x}"


async def async_step_handler(x: str) -> str:
    """Async step handler."""
    await asyncio.sleep(0.01)
    return f"async_step:{x}"


def error_step_handler(x: str) -> str:
    """Step that raises an error."""
    raise ValueError(f"Error in step: {x}")


def conditional_step_handler(x: str) -> str:
    """Step that can be conditionally executed."""
    return f"conditional:{x}"


def gen_step_handler(count: int) -> Generator[int, None, None]:
    """Generator step handler."""
    for i in range(count):
        yield i


async def async_gen_step_handler(count: int) -> AsyncGenerator[int, None]:
    """Async generator step handler."""
    for i in range(count):
        await asyncio.sleep(0.01)
        yield i


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def simple_workflow():
    """Create a simple workflow for testing."""
    return Workflow(name="simple_workflow")


@pytest.fixture
def step1_tool():
    """Create a tool for step 1."""
    return Tool(name="step1", handler=step1_handler)


@pytest.fixture
def step2_tool():
    """Create a tool for step 2."""
    return Tool(name="step2", handler=step2_handler)


@pytest.fixture
def step3_tool():
    """Create a tool for step 3."""
    return Tool(name="step3", handler=step3_handler)


@pytest.fixture
def async_step_tool():
    """Create an async tool for testing."""
    return Tool(name="async_step", handler=async_step_handler)


@pytest.fixture
def error_step_tool():
    """Create a tool that raises errors."""
    return Tool(name="error_step", handler=error_step_handler)


@pytest.fixture
def conditional_step_tool():
    """Create a tool for conditional execution."""
    return Tool(name="conditional_step", handler=conditional_step_handler)


@pytest.fixture
def gen_step_tool():
    """Create a generator tool."""
    return Tool(name="gen_step", handler=gen_step_handler)


@pytest.fixture
def async_gen_step_tool():
    """Create an async generator tool."""
    return Tool(name="async_gen_step", handler=async_gen_step_handler)


@pytest.fixture
def linear_workflow():
    """Create a linear workflow: step1 -> step2 -> step3."""
    workflow = Workflow(name="linear_workflow")
    workflow.step(step1_handler, x="input1")
    workflow.step(step2_handler, x="input2", step1_result="step1_handler.result")
    workflow.step(step3_handler, x="input3")
    return workflow


@pytest.fixture
def parallel_workflow():
    """Create a workflow with parallel steps."""
    workflow = Workflow(name="parallel_workflow")
    workflow.step(step1_handler, x="input1")
    workflow.step(step3_handler, x="input3")  # Independent of step1
    return workflow


@pytest.fixture
def complex_workflow():
    """Create a complex workflow with multiple dependencies."""
    workflow = Workflow(name="complex_workflow")
    workflow.step(step1_handler, x="input1")
    workflow.step(step2_handler, x="input2", step1_result="step1_handler.result")
    workflow.step(step3_handler, x="input3")  # Independent
    workflow.step(async_step_handler, x="input4", depends_on=["step2_handler", "step3_handler"])
    return workflow


class TestWorkflowCreation:
    """Test Workflow instantiation and validation."""

    def test_missing_name(self):
        """Test that Workflow requires a name."""
        with pytest.raises(Exception, match="Field required"):
            Workflow()
    
    def test_invalid_name(self):
        """Test that Workflow requires a valid name."""
        with pytest.raises(Exception, match="Input should be a valid string"):
            Workflow(name=123)


class TestStepManagement:
    def test_step_without_explicit_name(self):
        """Test that Workflow requires a step name."""
        wf = Workflow(name="workflow").step(async_handler)
        assert wf._steps["async_handler"].name == "async_handler"

    def test_steps_same_functions(self):
        with pytest.raises(Exception, match="Step async_handler already exists in the workflow."):
            Workflow(name="workflow").step(async_handler).step(async_handler)

    def tests_steps_same_handler_different_instances(self):
        tool = Tool(name="async_handler_2", handler=async_handler)
        wf = (Workflow(name="workflow")
            .step(async_handler)
            .step(tool)
        )
        assert wf._steps["async_handler"].name == "async_handler"
        assert wf._steps["async_handler_2"].name == "async_handler_2"
    
    @pytest.mark.asyncio
    async def test_steps_parameter_behavior(self):
        """Test that workflow parameters override step default parameters during execution."""
        # Create a workflow with a step that has a default parameter
        wf = Workflow(name="workflow").step(async_handler, x="step_param")
        result = await wf().collect()
        assert result.output == "async:step_param"

        # Check parameter is overriden by workflow
        result = await wf(x="workflow_param").collect()
        assert result.output == "async:workflow_param"

    async def test_parameters_same_name_different_steps(self):
        tool = Tool(name="async_handler_2", handler=async_handler)
        wf = (Workflow(name="workflow")
            .step(async_handler, x="step_param")
            .step(tool, x="step2_param", depends_on=["async_handler"])
        )
        result = await wf(x="workflow_param").collect()
        assert get_run_context()._tracing.as_records()[0].input['x'] == "workflow_param"
        assert get_run_context()._tracing.as_records()[1].input['x'] == "workflow_param"
        assert get_run_context()._tracing.as_records()[2].input['x'] == "workflow_param"
        assert result.output == "async:workflow_param"

    def test_add_step_with_dict(self, simple_workflow):
        """Test adding a step using a dictionary (converted to Tool)."""
        step_dict = {
            "name": "dict_step",
            "handler": step1_handler
        }
        simple_workflow.step(step_dict)
        
        assert "dict_step" in simple_workflow._steps
        assert isinstance(simple_workflow._steps["dict_step"], Tool)
        assert simple_workflow._steps["dict_step"].handler == step1_handler

    def test_add_step_with_callable(self, simple_workflow):
        """Test adding a step using a callable (converted to Tool)."""
        simple_workflow.step(step1_handler)
        
        assert "step1_handler" in simple_workflow._steps
        assert isinstance(simple_workflow._steps["step1_handler"], Tool)
        assert simple_workflow._steps["step1_handler"].handler == step1_handler
    

class TestContext:
    """Test context management."""

    @staticmethod
    def context_function(x: str) -> str:
        assert get_run_context().get_data(".input.x") == x
        get_run_context().set_data(".input.x", "new_input")
        assert get_run_context().get_data(".input.x") == "new_input"
        get_run_context().set_data(".new_parameter", "new_parameter")
        get_run_context().set_data(".second_new_parameter", "second_new_parameter")
        return x 

    @staticmethod
    def check_neighbor_parameters() -> str:
        assert get_run_context().get_data("context_function.new_parameter") == "new_parameter"
        assert get_run_context().get_data("context_function.second_new_parameter") == "second_new_parameter"

    @staticmethod
    def multiple_outputs() -> str:
        return "first_output", "second_output"

    async def test_context_levels(self):
        """Test context levels."""
        wf = Workflow(name="first_workflow").step(async_handler, x="step_param")
        _ = await wf().collect()
        assert get_run_context()._tracing.as_records()[0].input == {}
        assert get_run_context()._tracing.as_records()[1].input['x'] == "step_param"

        # Value overridden
        wf = Workflow(name="second_workflow").step(async_handler, x="step_param")
        _ = await wf(x='workflow_param').collect()
        assert get_run_context()._tracing.as_records()[0].input['x'] == "workflow_param"
        assert get_run_context()._tracing.as_records()[1].input['x'] == "workflow_param"

        # Creating new parameter step
        wf = Workflow(name="third_workflow").step(TestContext.context_function, x="old_input")
        _ = await wf().collect()
        assert get_run_context()._tracing.as_records()[1].new_parameter == "new_parameter"   # 0: workflow, 1: step
        assert get_run_context()._tracing.as_records()[1].second_new_parameter == "second_new_parameter"

        # Checking neighbor parameters
        wf = (Workflow(name="fourth_workflow")
            .step(TestContext.context_function, x="step_param")
            .step(TestContext.check_neighbor_parameters, x="step_param")
        )
        _ = await wf().collect()

    async def test_workflow_final_output(self):
        """Test that the final output is the output of the last step."""
        tool = Tool(name="async_handler_2", handler=async_handler)
        wf = (Workflow(name="workflow")
            .step(async_handler, x="step_param")
            .step(tool, x="step2_param", depends_on=["async_handler"])
        )
        result = await wf().collect()
        assert result.output == "async:step2_param"
    
    async def test_multiple_outputs(self):
        wf = (Workflow(name="multiple_outputs_wf")
            .step(TestContext.multiple_outputs)
        )
        _ = await wf().collect()
        assert len(get_run_context()._tracing.as_records()[0].output) == 2
        assert get_run_context()._tracing.as_records()[0].output[0] == "first_output"
        assert get_run_context()._tracing.as_records()[0].output[1] == "second_output"

    def test_nonexistent_dependency_error(self, simple_workflow):
        """Test that depending on nonexistent step raises an error."""
        with pytest.raises(ValueError, match="Source step nonexistent not found"):
            simple_workflow.step(step2_handler, depends_on=["nonexistent"])


class TestControlFlow:
    # Parallel execution, sequencial when dependencies, depends_on, when

    @staticmethod
    def context_function() -> str:
        get_run_context().set_data(".new_parameter", "new_parameter_value")

    @staticmethod
    def dependent_function() -> str:
        return get_run_context().get_data("context_function.new_parameter")

    async def test_parallel_execution(self):
        """Test parallel execution."""
        wf = (Workflow(name="parallel_workflow")
            .step(async_handler, x="output")
            .step(sync_handler, x="output")
        )
        _ = await wf().collect()
        assert get_run_context()._tracing.as_records()[0].output == "async:output"

    async def test_depends_on(self):
        wf = (Workflow(name="parallel_workflow")
            .step(async_handler, x="output")
            .step(sync_handler, x="output", depends_on=["async_handler"])
        )
        _ = await wf().collect()
        assert get_run_context()._tracing.as_records()[0].output == "sync:output"

    async def test_when(self):
        wf = (Workflow(name="parallel_workflow")
            .step(async_handler, x="output")
            .step(sync_handler, x="output", when=lambda: True)
        )
        _ = await wf().collect()
        assert get_run_context()._tracing.as_records()[0].output == "async:output"

    async def test_when_false(self):
        wf = (Workflow(name="parallel_workflow")
            .step(async_handler, x="output")
            .step(sync_handler, x="output", when=lambda: False)
        )
        _ = await wf().collect()
        # print(get_run_context()._tracing.as_records())
        assert len(get_run_context()._tracing.as_records()) == 2 # sync step never called
        assert get_run_context()._tracing.as_records()[0].output == "async:output"

    async def test_automatic_dependency_linking(self):
        wf = (Workflow(name="dependency_linking_wf")
            .step(TestControlFlow.context_function)
            .step(TestControlFlow.dependent_function)
        )
        _ = await wf().collect()
        print(get_run_context()._tracing.as_records())
        assert get_run_context()._tracing.as_records()[0].output == "new_parameter_value"

    async def test_automatic_dependency_linking_incorrect_order(self):
        with pytest.raises(ValueError, match="Source step context_function not found in workflow"):
            wf = (Workflow(name="dependency_linking_wf")
                .step(TestControlFlow.dependent_function)
                .step(TestControlFlow.context_function)
            )
            _ = await wf().collect()
            print(get_run_context()._tracing.as_records())

    def test_invalid_depends_on_type(self, simple_workflow):
        """Test that invalid depends_on type raises an error."""
        simple_workflow.step(step1_handler)
        
        with pytest.raises(ValueError, match="depends_on must be a list"):
            simple_workflow.step(step2_handler, depends_on="step1_handler")

    def test_cycle_detection_prevents_cycles(self, simple_workflow):
        """Test that cycle detection prevents creating cycles."""
        simple_workflow.step(step1_handler)
        simple_workflow.step(step2_handler)
        
        # Create a cycle: step1 -> step2 -> step1
        simple_workflow._steps["step1_handler"].next_steps.add("step2_handler")
        simple_workflow._steps["step2_handler"].previous_steps.add("step1_handler")
        
        with pytest.raises(ValueError, match="would create a cycle"):
            simple_workflow._link("step2_handler", "step1_handler")

    def test_complex_dependency_structure(self, simple_workflow):
        """Test complex dependency structures."""
        simple_workflow.step(step1_handler)
        simple_workflow.step(step2_handler, depends_on=["step1_handler"])
        simple_workflow.step(step3_handler)
        simple_workflow.step(async_step_handler, depends_on=["step2_handler", "step3_handler"])
        
        assert simple_workflow._is_dag() is True
        
        # Check that all dependencies are correctly set
        step1 = simple_workflow._steps["step1_handler"]
        step2 = simple_workflow._steps["step2_handler"]
        step3 = simple_workflow._steps["step3_handler"]
        async_step = simple_workflow._steps["async_step_handler"]
        
        assert "step2_handler" in step1.next_steps
        assert "async_step_handler" in step2.next_steps
        assert "async_step_handler" in step3.next_steps
        assert "step1_handler" in step2.previous_steps
        assert "step2_handler" in async_step.previous_steps
        assert "step3_handler" in async_step.previous_steps

    async def test_workflow_parameter_passing(self, simple_workflow):
        """Test that parameters are correctly passed to workflow steps."""
        simple_workflow.step(step1_handler, x="workflow_input")
        
        events = []
        async for event in simple_workflow(x="override_input"):
            events.append(event)
        
        output_events = [e for e in events if isinstance(e, OutputEvent)]
        assert len(output_events) == 2
        assert output_events[-1].error is None
        assert "step1:override_input" in str(output_events[-1].output)


class TestParameterAndNesting:
    def test_params_model_validation(self, simple_workflow):
        """Test that params_model validates input correctly."""
        simple_workflow.step(step1_handler)
        
        params_model = simple_workflow.params_model
        
        # Valid input should work
        valid_input = {"x": "test_value"}
        validated = params_model.model_validate(valid_input)
        assert validated.x == "test_value"
        
        # Invalid input should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            params_model.model_validate({"invalid_param": "value"})


class TestWorkflowIntegration:
    def test_workflow_with_tool_objects(self, simple_workflow, step1_tool, step2_tool):
        """Test workflow with Tool objects."""
        simple_workflow.step(step1_tool)
        simple_workflow.step(step2_tool, depends_on=["step1"])
        
        assert "step1" in simple_workflow._steps
        assert "step2" in simple_workflow._steps
        
        # Check dependency
        step1 = simple_workflow._steps["step1"]
        step2 = simple_workflow._steps["step2"]
        assert "step2" in step1.next_steps
        assert "step1" in step2.previous_steps
    
    def test_workflow_schema_generation(self, simple_workflow):
        """Test workflow schema generation."""
        simple_workflow.step(step1_handler)
        
        # Should generate valid schemas
        openai_schema = simple_workflow.openai_schema
        anthropic_schema = simple_workflow.anthropic_schema
        
        assert "function" in openai_schema
        assert "name" in anthropic_schema
        assert openai_schema["function"]["name"] == "simple_workflow"
        assert anthropic_schema["name"] == "simple_workflow"

    


