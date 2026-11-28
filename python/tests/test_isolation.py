"""The suite's own isolation (``outside_any_run`` in ``python/tests/conftest.py``): no test starts inside a run
another test left set.

Two tests, in this order. The first sets a run context and call ids the way a sync test or fixture does (on the
main thread's context) and leaves them. The second is async: pytest-asyncio copies the main thread's context into
it, and without the fixture its first top-level run took the leftover context as its own, with that context's
tracing provider (none here), so the run was never stored.

Marked ``serial``: CI runs these after the parallel pass, in one process and in file order. Run apart (on two
xdist workers, or the second alone) the second passes without the fixture too: nothing was left for it.
"""

import asyncio

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.state import (
    get_call_id,
    get_parent_call_id,
    get_run_context,
    set_call_id,
    set_parent_call_id,
    set_run_context,
)
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider

pytestmark = pytest.mark.serial

# A hang guard, not a timing check: the run takes milliseconds.
DEADLINE_S = 30


def test_a_test_that_leaves_a_run_context_set():
    set_run_context(RunContext(tracing_provider=None))
    set_call_id("left-by-an-earlier-test")
    set_parent_call_id("left-by-an-earlier-test")


async def test_the_next_test_starts_outside_any_run():
    assert get_run_context() is None
    assert get_call_id() is None
    assert get_parent_call_id() is None

    agent = Agent(name="agent", model=TestModel(responses=["ok"]), tracing_provider=InMemoryTracingProvider)
    result = await asyncio.wait_for(agent(prompt="hi").collect(), DEADLINE_S)
    assert result.status.code == "success"
    assert result.parent_run_id is None
    # Stored by the agent's own provider, under the run's own id.
    assert result.run_id in InMemoryTracingProvider._storage
