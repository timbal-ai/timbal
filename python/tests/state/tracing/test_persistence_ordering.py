"""OUTPUT is a durable checkpoint for a turn that can immediately be resumed."""

import asyncio
import json

from timbal import Agent
from timbal.core.test_model import TestModel as OfflineModel
from timbal.state import RunContext, get_run_context, set_run_context
from timbal.state.tracing.providers.base import TracingProvider
from timbal.state.tracing.trace import Trace


async def test_parent_session_available_when_root_output_is_received():
    stored = {}

    class Storage(TracingProvider):
        @classmethod
        async def get(cls, run_context):
            records = stored.get(run_context.parent_id)
            return Trace(records) if records is not None else None

        @classmethod
        async def _store(cls, run_context):
            # Yield to the scheduler to exercise an asynchronous persistence
            # boundary, and copy as remote storage would (no shared objects).
            await asyncio.sleep(0)
            stored[run_context.id] = json.loads(json.dumps(run_context._trace.model_dump()))

    async def remember_route():
        session = await get_run_context().get_session()
        session["route"] = "support"

    original = get_run_context()
    try:
        set_run_context(RunContext(tracing_provider=Storage))
        agent = Agent(
            name="router",
            model=OfflineModel(responses=["support"]),
            tracing_provider=Storage,
            post_hook=remember_route,
        )
        saw_output = False
        async for event in agent(prompt="help"):
            if event.type != "OUTPUT" or event.parent_call_id is not None:
                continue
            assert event.status.code == "success", event.error
            # Read from a new context before requesting the generator's next
            # item. Moving persistence after yield would break this contract.
            child = RunContext(parent_id=event.run_id, tracing_provider=Storage)
            assert await child.get_session() == {"route": "support"}
            trace = await Storage.get(child)
            root = trace[trace._root_call_id]
            assert root.t1 is not None
            assert root.status["code"] == "success"
            saw_output = True
        assert saw_output
    finally:
        set_run_context(original)
