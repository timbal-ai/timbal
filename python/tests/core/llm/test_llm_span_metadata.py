"""LLM trace labels follow the model dispatched for each call."""

import asyncio

import pytest
from timbal import Agent, FallbackModel
from timbal.core.llm import router
from timbal.core.test_model import TestModel
from timbal.state import get_run_context
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.types.events import OutputEvent


@pytest.fixture
def provider_streams(monkeypatch):
    """Keep real routing/fallback/retry logic; replace provider network calls."""
    calls = []
    failures = set()

    def prepare(*, model_name, messages, **_kwargs):
        async def stream():
            calls.append(model_name)
            # Force overlapping calls when testing a shared Agent.
            await asyncio.sleep(0)
            if model_name in failures:
                raise ValueError(f"Unavailable: {model_name}")
            async for chunk in TestModel(responses=[model_name]).stream(messages=messages):
                yield chunk

        return stream, "mock provider"

    monkeypatch.setattr(router, "_resolve_client", lambda *_args: (None, None))
    for name in ("prepare_messages_request", "prepare_responses_request", "prepare_chat_completions_request"):
        monkeypatch.setattr(router, name, prepare)
    return calls, failures


async def collect_llm(agent, **kwargs):
    events = [event async for event in agent(prompt="hello", **kwargs)]
    assert events[-1].error is None
    llm_events = [event for event in events if isinstance(event, OutputEvent) and event.path == agent._llm._path]
    assert len(llm_events) == 1
    event = llm_events[0]
    trace = InMemoryTracingProvider._storage[event.run_id]
    # Both the streamed output event and persisted trace must carry the labels.
    assert trace[event.call_id].metadata == event.metadata
    return event


@pytest.mark.parametrize("configured", ["openai/default", FallbackModel("openai/default", "google/backup")])
async def test_runtime_override_updates_span(configured, provider_streams):
    calls, _ = provider_streams
    agent = Agent(name="agent", model=configured, max_tokens=100)
    configured_metadata = dict(agent._llm.metadata)

    event = await collect_llm(agent, model="anthropic/override")

    assert event.metadata == {"type": "LLM", "model_provider": "anthropic", "model_name": "override"}
    assert event.output.collect_text() == "override"
    assert calls == ["override"]
    assert agent._llm.metadata == configured_metadata

    # A later run without an override must go back to the configured primary.
    event = await collect_llm(agent)
    assert event.metadata == {"type": "LLM", "model_provider": "openai", "model_name": "default"}


@pytest.mark.parametrize("fail_primary", [False, True])
@pytest.mark.parametrize("runtime_override", [False, True])
async def test_fallback_span_names_selected_entry(fail_primary, runtime_override, provider_streams):
    calls, failures = provider_streams
    chain = FallbackModel("openai/primary", "google/backup")
    agent = Agent(name="agent", model="openai/default" if runtime_override else chain)
    if fail_primary:
        failures.add("primary")

    event = await collect_llm(agent, **({"model": chain} if runtime_override else {}))

    provider, name = ("google", "backup") if fail_primary else ("openai", "primary")
    assert event.metadata == {"type": "LLM", "model_provider": provider, "model_name": name}
    assert event.output.collect_text() == name
    assert calls == (["primary", "backup"] if fail_primary else ["primary"])


@pytest.mark.usefixtures("provider_streams")
async def test_concurrent_overrides_do_not_leak_metadata():
    agent = Agent(name="agent", model=FallbackModel("openai/default", "google/backup"), max_tokens=100)
    configured_metadata = dict(agent._llm.metadata)
    models = ["anthropic/first", "google/second", "openai/third"]

    events = await asyncio.gather(*(collect_llm(agent, model=model) for model in models))

    for event, model in zip(events, models, strict=True):
        provider, name = model.split("/", 1)
        assert event.metadata == {"type": "LLM", "model_provider": provider, "model_name": name}
    assert agent._llm.metadata == configured_metadata


async def test_test_model_override_updates_span():
    agent = Agent(name="agent", model="openai/default")

    event = await collect_llm(agent, model=TestModel(responses=["offline"]))

    assert event.metadata == {"type": "LLM", "model_provider": "test", "model_name": "model"}
    assert event.output.collect_text() == "offline"


async def test_failed_fallback_span_names_last_attempt(provider_streams):
    _, failures = provider_streams
    failures.update(["primary", "backup"])
    agent = Agent(name="agent", model=FallbackModel("openai/primary", "google/backup"))

    events = [event async for event in agent(prompt="hello")]

    assert events[-1].error is not None
    trace = InMemoryTracingProvider._storage[events[-1].run_id]
    span = trace.get_path(agent._llm._path)[0]
    assert span.error is not None
    assert span.metadata == {"type": "LLM", "model_provider": "google", "model_name": "backup"}


@pytest.mark.usefixtures("provider_streams")
async def test_router_without_active_span():
    from timbal.state import _call_id, _run_context_var

    ctx_token = _run_context_var.set(None)
    call_token = _call_id.set(None)
    try:
        chunks = [chunk async for chunk in router._llm_router(model="openai/direct")]
        assert chunks[0].collect_text() == "direct"
        assert not get_run_context()._trace
    finally:
        _call_id.reset(call_token)
        _run_context_var.reset(ctx_token)
