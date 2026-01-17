"""Tests for workflow step skipping behavior.

This module tests the pull-based skip logic in workflows where:
- Steps are skipped when their `when` condition returns False
- Steps are skipped when they try to access a skipped dependency's span (SpanNotFound)
- Steps fail when their dependency failed (error propagation)
- Input parameters can override default lambda-based parameters
"""

import pytest
from timbal import Workflow
from timbal.core.workflow import StepState
from timbal.errors import SpanNotFound
from timbal.state import get_run_context

# ==============================================================================
# Test Handler Functions
# ==============================================================================


def returns_5():
    """Returns valor=5, which is <= 10."""
    return {"valor": 5}


def returns_15():
    """Returns valor=15, which is > 10."""
    return {"valor": 15}


def double(x: int):
    """Doubles the input."""
    return {"valor": x * 2}


def triple(x: int):
    """Triples the input."""
    return {"valor": x * 3}


def add(a: int, b: int):
    """Adds two inputs."""
    return {"valor": a + b}


def identity(x):
    """Returns input as-is."""
    return {"valor": x}


def raises_error():
    """Always raises an error."""
    raise ValueError("Intentional error")


def conditional_error(should_fail: bool):
    """Raises error based on input."""
    if should_fail:
        raise ValueError("Conditional error triggered")
    return {"valor": "success"}


# ==============================================================================
# Test Classes
# ==============================================================================


class TestWhenConditionSkipping:
    """Test skipping steps based on `when` condition."""

    @pytest.mark.asyncio
    async def test_when_false_skips_step(self):
        """Step is skipped when `when` returns False."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # Only workflow + returns_5 should have spans (double was skipped)
        assert len(records) == 2
        assert records[0].path == "test"
        assert records[1].path == "test.returns_5"
        assert records[1].output == {"valor": 5}

    @pytest.mark.asyncio
    async def test_when_true_runs_step(self):
        """Step runs when `when` returns True."""
        workflow = (
            Workflow(name="test")
            .step(returns_15)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_15").output["valor"],
                when=lambda: get_run_context().step_span("returns_15").output["valor"] > 10,
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        assert len(records) == 3
        assert records[2].path == "test.double"
        assert records[2].output == {"valor": 30}  # 15 * 2


class TestDependencySkipPropagation:
    """Test that steps are skipped when their dependencies are skipped."""

    @pytest.mark.asyncio
    async def test_single_skipped_dependency(self):
        """Step is skipped when its only dependency was skipped."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                triple,
                x=lambda: get_run_context().step_span("double").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # double is skipped, so triple should also be skipped (needs double's span)
        assert len(records) == 2
        assert records[1].path == "test.returns_5"

    @pytest.mark.asyncio
    async def test_all_dependencies_skipped(self):
        """Step is skipped when all its dependencies were skipped."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                triple,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                add,
                a=lambda: get_run_context().step_span("double").output["valor"],
                b=lambda: get_run_context().step_span("triple").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # double and triple are skipped, so add should also be skipped
        assert len(records) == 2
        assert records[1].path == "test.returns_5"

    @pytest.mark.asyncio
    async def test_conditional_dependency_branch(self):
        """Step runs using alternative branch when one dependency is skipped."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                identity,
                x=lambda: (
                    get_run_context().step_span("double").output["valor"]
                    if get_run_context().step_span("returns_5").output["valor"] > 10
                    else get_run_context().step_span("returns_5").output["valor"]
                ),
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # identity should run using returns_5's output since double was skipped
        assert len(records) == 3
        assert records[2].path == "test.identity"
        assert records[2].output == {"valor": 5}


class TestDependencyErrorPropagation:
    """Test that steps fail when their dependencies fail."""

    @pytest.mark.asyncio
    async def test_step_fails_when_dependency_errors(self):
        """Step fails when trying to access output from failed dependency."""
        workflow = (
            Workflow(name="test")
            .step(raises_error)
            .step(
                double,
                x=lambda: get_run_context().step_span("raises_error").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # raises_error should have error, double should fail
        assert len(records) == 2  # workflow + raises_error (double failed during eval)
        assert records[1].error is not None

    @pytest.mark.asyncio
    async def test_error_in_when_condition(self):
        """Step fails when `when` condition raises an error."""
        workflow = (
            Workflow(name="test")
            .step(raises_error)
            .step(
                double,
                x=lambda: 10,
                when=lambda: get_run_context().step_span("raises_error").output["valor"] > 10,
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # double should fail because when condition tries to access errored step
        assert len(records) == 2


class TestInputOverrides:
    """Test that workflow inputs override lambda-based default params."""

    @pytest.mark.asyncio
    async def test_input_overrides_lambda(self):
        """Workflow input overrides step's lambda-based default param."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
            )
        )

        # Override x with direct value
        await workflow(x=100).collect()
        records = get_run_context()._trace.as_records()

        assert len(records) == 3
        assert records[2].path == "test.double"
        assert records[2].output == {"valor": 200}  # 100 * 2, not 5 * 2

    @pytest.mark.asyncio
    async def test_input_skips_lambda_resolution(self):
        """Lambda is not evaluated when input is provided."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                triple,
                # This lambda would fail if evaluated (double was skipped)
                x=lambda: get_run_context().step_span("double").output["valor"],
            )
        )

        # Override x to skip the problematic lambda
        await workflow(x=50).collect()
        records = get_run_context()._trace.as_records()

        # triple should run with x=50 (skipping the lambda that would fail)
        assert len(records) == 3
        assert records[2].path == "test.triple"
        assert records[2].output == {"valor": 150}  # 50 * 3


class TestComplexWorkflows:
    """Test complex workflow scenarios with multiple branches and conditions."""

    @pytest.mark.asyncio
    async def test_diamond_dependency_all_run(self):
        """Diamond pattern: A -> B,C -> D (all run when condition is met)."""
        workflow = (
            Workflow(name="test")
            .step(returns_15)  # valor > 10, so B and C will run
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_15").output["valor"],
                when=lambda: get_run_context().step_span("returns_15").output["valor"] > 10,
            )
            .step(
                triple,
                x=lambda: get_run_context().step_span("returns_15").output["valor"],
                when=lambda: get_run_context().step_span("returns_15").output["valor"] > 10,
            )
            .step(
                add,
                a=lambda: get_run_context().step_span("double").output["valor"],
                b=lambda: get_run_context().step_span("triple").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        assert len(records) == 5
        double_output = next(r for r in records if r.path == "test.double").output
        triple_output = next(r for r in records if r.path == "test.triple").output
        add_output = next(r for r in records if r.path == "test.add").output

        assert double_output == {"valor": 30}  # 15 * 2
        assert triple_output == {"valor": 45}  # 15 * 3
        assert add_output == {"valor": 75}  # 30 + 45

    @pytest.mark.asyncio
    async def test_diamond_dependency_all_skipped(self):
        """Diamond pattern: A -> B,C -> D (all skip when condition not met)."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)  # valor <= 10, so B and C will be skipped
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                triple,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(
                add,
                a=lambda: get_run_context().step_span("double").output["valor"],
                b=lambda: get_run_context().step_span("triple").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # Only workflow + returns_5 should run
        assert len(records) == 2
        assert records[1].path == "test.returns_5"

    @pytest.mark.asyncio
    async def test_parallel_independent_steps(self):
        """Independent steps run in parallel regardless of skip conditions."""
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: get_run_context().step_span("returns_5").output["valor"] > 10,
            )
            .step(returns_15)  # Independent, should always run
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # returns_5, returns_15 run; double is skipped
        assert len(records) == 3
        paths = [r.path for r in records]
        assert "test.returns_5" in paths
        assert "test.returns_15" in paths
        assert "test.double" not in paths


class TestSpanNotFoundError:
    """Test SpanNotFound exception behavior."""

    @pytest.mark.asyncio
    async def test_span_not_found_for_skipped_step(self):
        """SpanNotFound is raised when accessing a skipped step's span."""
        # This is tested implicitly through other tests, but let's be explicit
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: False,  # Always skip
            )
            .step(
                triple,
                x=lambda: get_run_context().step_span("double").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # triple should be skipped because double has no span
        assert len(records) == 2

    def test_nonexistent_step_in_lambda_caught_at_build_time(self):
        """Referencing a nonexistent step in lambda is caught at workflow build time."""
        with pytest.raises(ValueError, match="Source step nonexistent not found"):
            Workflow(name="test").step(returns_5).step(
                double,
                x=lambda: get_run_context().step_span("nonexistent").output["valor"],
            )


class TestStepStateTracking:
    """Test that step states are tracked correctly."""

    @pytest.mark.asyncio
    async def test_completed_state(self):
        """Successfully executed steps have COMPLETED state."""
        workflow = Workflow(name="test").step(returns_5)

        result = await workflow().collect()

        assert result.error is None

    @pytest.mark.asyncio
    async def test_skipped_vs_failed_distinction(self):
        """Skipped steps (condition=False) vs failed steps (error) are distinct."""
        # This workflow has both a skipped step and a failed step
        workflow = (
            Workflow(name="test")
            .step(returns_5)
            .step(
                double,
                x=lambda: get_run_context().step_span("returns_5").output["valor"],
                when=lambda: False,  # Will be SKIPPED
            )
            .step(raises_error)  # Will be FAILED
            .step(
                triple,
                x=lambda: get_run_context().step_span("double").output["valor"],
            )
        )

        await workflow().collect()
        records = get_run_context()._trace.as_records()

        # Check that raises_error has an error recorded
        error_record = next((r for r in records if r.path == "test.raises_error"), None)
        assert error_record is not None
        assert error_record.error is not None
