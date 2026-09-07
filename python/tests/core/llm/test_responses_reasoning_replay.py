"""OpenAI Responses: request `include` for encrypted reasoning, and replay across a tool loop.

Reasoning models (GPT-5.x, o-series, codex) keep no chain of thought between the calls
of a tool loop unless every request (a) asks for `reasoning.encrypted_content` and (b)
replays the `reasoning` items it got back, ahead of the function_call each one produced.
With `store: false` that is the only mechanism. Without it the model degrades with loop
depth — up to emitting tool calls as plain text (`to=functions.…`) and ending the turn.

These tests run entirely offline against a fake `client.responses.create` that yields
real SDK event objects, so they exercise collector → memory → serializer → next request.
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseContentPartAddedEvent,
    ResponseCreatedEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseFunctionToolCall,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_usage import InputTokensDetails, OutputTokensDetails, ResponseUsage
from timbal.core.llm.responses import (
    REASONING_ENCRYPTED_CONTENT_INCLUDE,
    prepare_responses_request,
    supports_encrypted_reasoning,
)
from timbal.types.content import ThinkingContent, ToolUseContent
from timbal.types.message import Message


@pytest.fixture(autouse=True)
def clean_context():
    from timbal.state import _call_id, _run_context_var

    token_ctx = _run_context_var.set(None)
    token_cid = _call_id.set(None)
    yield
    _run_context_var.reset(token_ctx)
    _call_id.reset(token_cid)


# ---------------------------------------------------------------------------
# Request preparation
# ---------------------------------------------------------------------------


class TestSupportsEncryptedReasoning:
    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5.6-luna",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5.1-codex",
            "GPT-5.6-luna",
            "gpt-6-astra",
            "o3",
            "o3-mini",
            "o4-mini",
            "o1",
            "codex-mini-latest",
        ],
    )
    def test_reasoning_models(self, model):
        assert supports_encrypted_reasoning(model)

    @pytest.mark.parametrize(
        "model", ["gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-nano", "grok-4.6", "grok-3", "", "   "]
    )
    def test_non_reasoning_models_and_xai(self, model):
        assert not supports_encrypted_reasoning(model)


def _prepare(model_name: str, provider_params: dict | None = None, messages: list[Message] | None = None):
    client = MagicMock()
    _, _ = prepare_responses_request(
        client=client,
        model_name=model_name,
        request_headers={},
        system_prompt=None,
        messages=messages or [Message.validate("hi")],
        tools=None,
        max_tokens=None,
        temperature=None,
        output_model=None,
        provider_params=provider_params or {},
    )
    # prepare returns a stream factory; grab the kwargs it will send by calling it.
    return client


async def _kwargs_sent(
    model_name: str, provider_params: dict | None = None, messages: list[Message] | None = None
) -> dict:
    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return _empty_stream()

    client = MagicMock()
    client.responses.create = fake_create
    create_stream, _ = prepare_responses_request(
        client=client,
        model_name=model_name,
        request_headers={},
        system_prompt=None,
        messages=messages or [Message.validate("hi")],
        tools=None,
        max_tokens=None,
        temperature=None,
        output_model=None,
        provider_params=provider_params or {},
    )
    async for _ in create_stream():
        pass
    return captured


async def _empty_stream():
    return
    yield


class TestPrepareResponsesRequestInclude:
    @pytest.mark.asyncio
    async def test_reasoning_model_requests_encrypted_content(self):
        kwargs = await _kwargs_sent("gpt-5.6-luna")
        assert kwargs["store"] is False
        assert REASONING_ENCRYPTED_CONTENT_INCLUDE in kwargs["include"]
        assert "web_search_call.action.sources" in kwargs["include"]

    @pytest.mark.asyncio
    async def test_non_reasoning_model_does_not(self):
        kwargs = await _kwargs_sent("gpt-4.1-mini")
        assert kwargs["include"] == ["web_search_call.action.sources"]

    @pytest.mark.asyncio
    async def test_xai_model_does_not(self):
        kwargs = await _kwargs_sent("grok-4.6")
        assert REASONING_ENCRYPTED_CONTENT_INCLUDE not in kwargs["include"]

    @pytest.mark.asyncio
    async def test_caller_include_is_merged_not_clobbered(self):
        kwargs = await _kwargs_sent("gpt-5.6-luna", provider_params={"include": ["message.output_text.logprobs"]})
        assert kwargs["include"] == [
            "web_search_call.action.sources",
            REASONING_ENCRYPTED_CONTENT_INCLUDE,
            "message.output_text.logprobs",
        ]

    @pytest.mark.asyncio
    async def test_caller_include_deduped(self):
        kwargs = await _kwargs_sent("gpt-5.6-luna", provider_params={"include": [REASONING_ENCRYPTED_CONTENT_INCLUDE]})
        assert kwargs["include"].count(REASONING_ENCRYPTED_CONTENT_INCLUDE) == 1

    @pytest.mark.asyncio
    async def test_other_provider_params_still_forwarded(self):
        kwargs = await _kwargs_sent("gpt-5.6-luna", provider_params={"reasoning": {"effort": "high"}})
        assert kwargs["reasoning"] == {"effort": "high"}

    @pytest.mark.asyncio
    async def test_memory_with_reasoning_items_is_serialized_top_level(self):
        memory = [
            Message.validate("find x"),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
                    ToolUseContent(id="call_1", name="search", input={"q": "x"}),
                ],
            ),
        ]
        kwargs = await _kwargs_sent("gpt-5.6-luna", messages=memory)
        types = [i.get("type") or i.get("role") for i in kwargs["input"]]
        assert types == ["user", "reasoning", "function_call"]
        assert kwargs["input"][1] == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1", "summary": []}


# ---------------------------------------------------------------------------
# Agent tool loop, offline
# ---------------------------------------------------------------------------


def _usage(inp=20, out=5):
    return ResponseUsage(
        input_tokens=inp,
        output_tokens=out,
        total_tokens=inp + out,
        input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )


def _response(model="gpt-5.6-luna", status="in_progress", usage=None):
    return Response(
        id="resp_001",
        created_at=1,
        model=model,
        object="response",
        output=[],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
        status=status,
        usage=usage,
        incomplete_details=None,
        instructions=None,
        metadata={},
        error=None,
        temperature=1.0,
        top_p=1.0,
        max_output_tokens=None,
        text=None,
        truncation="disabled",
    )


def _tool_call_turn(
    rs_id: str, encrypted: str, call_id: str, fn_item: str, name: str, arguments: str, *, summary: str = ""
):
    """Events for one model response: reasoning (with payload) → function_call."""
    seq = iter(range(100))
    reasoning_done = ResponseReasoningItem(
        type="reasoning",
        id=rs_id,
        status="completed",
        summary=[Summary(type="summary_text", text=summary)] if summary else [],
        encrypted_content=encrypted,
    )
    return [
        ResponseCreatedEvent(type="response.created", response=_response(), sequence_number=next(seq)),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=0,
            sequence_number=next(seq),
            item=ResponseReasoningItem(type="reasoning", id=rs_id, status="in_progress", summary=[]),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done", output_index=0, item=reasoning_done, sequence_number=next(seq)
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=1,
            sequence_number=next(seq),
            item=ResponseFunctionToolCall(
                type="function_call", id=fn_item, call_id=call_id, name=name, arguments="", status="in_progress"
            ),
        ),
        ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id=fn_item,
            output_index=1,
            delta=arguments,
            sequence_number=next(seq),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            output_index=1,
            sequence_number=next(seq),
            item=ResponseFunctionToolCall(
                type="function_call", id=fn_item, call_id=call_id, name=name, arguments=arguments, status="completed"
            ),
        ),
        ResponseCompletedEvent(
            type="response.completed", response=_response(status="completed", usage=_usage()), sequence_number=next(seq)
        ),
    ]


def _text_turn(rs_id: str, encrypted: str, text: str):
    """Events for the final model response: reasoning → assistant text."""
    seq = iter(range(100))
    return [
        ResponseCreatedEvent(type="response.created", response=_response(), sequence_number=next(seq)),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=0,
            sequence_number=next(seq),
            item=ResponseReasoningItem(type="reasoning", id=rs_id, status="in_progress", summary=[]),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            output_index=0,
            sequence_number=next(seq),
            item=ResponseReasoningItem(
                type="reasoning", id=rs_id, status="completed", summary=[], encrypted_content=encrypted
            ),
        ),
        ResponseOutputItemAddedEvent(
            type="response.output_item.added",
            output_index=1,
            sequence_number=next(seq),
            item=ResponseOutputMessage(type="message", id="msg_1", role="assistant", status="in_progress", content=[]),
        ),
        ResponseContentPartAddedEvent(
            type="response.content_part.added",
            item_id="msg_1",
            output_index=1,
            content_index=0,
            sequence_number=next(seq),
            part=ResponseOutputText(type="output_text", text="", annotations=[]),
        ),
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            item_id="msg_1",
            output_index=1,
            content_index=0,
            delta=text,
            logprobs=[],
            sequence_number=next(seq),
        ),
        ResponseOutputItemDoneEvent(
            type="response.output_item.done",
            output_index=1,
            sequence_number=next(seq),
            item=ResponseOutputMessage(
                type="message",
                id="msg_1",
                role="assistant",
                status="completed",
                content=[ResponseOutputText(type="output_text", text=text, annotations=[])],
            ),
        ),
        ResponseCompletedEvent(
            type="response.completed", response=_response(status="completed", usage=_usage()), sequence_number=next(seq)
        ),
    ]


class _ScriptedResponses:
    """`client.responses.create` that plays one scripted event list per call and records every request."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests: list[dict] = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        events = self.turns.pop(0)

        async def gen():
            for e in events:
                yield e

        return gen()


async def _run_agent(scripted: _ScriptedResponses, *, prompt: str = "look it up"):
    """Run a one-tool Agent against the scripted client; return (OutputEvent, tool calls made).

    Everything — including `collect()` — runs inside the patches: the collector is lazy,
    so returning it and awaiting outside would hit the real network.
    """
    from timbal import Agent

    calls: list[dict] = []

    def search(q: str) -> str:
        """Search for a query."""
        calls.append({"q": q})
        return f"result for {q}"

    agent = Agent(name="agent", model="openai/gpt-5.6-luna", tools=[search])
    client = MagicMock()
    client.responses.create = scripted.create
    with patch("timbal.core.llm.router._resolve_client", return_value=(client, None)):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "key"}):
            with patch("timbal.core.llm.router.TIMBAL_OPENAI_API", "responses"):
                result = await agent(prompt=prompt).collect()
    assert result.error is None, result.error
    return result, calls


class TestAgentToolLoopReplaysReasoning:
    @pytest.mark.asyncio
    async def test_second_request_carries_first_steps_reasoning_item(self):
        scripted = _ScriptedResponses(
            [
                _tool_call_turn("rs_1", "enc-1", "call_1", "fc_1", "search", '{"q": "x"}'),
                _text_turn("rs_2", "enc-2", "Done: x"),
            ]
        )
        result, calls = await _run_agent(scripted)

        assert calls == [{"q": "x"}]
        assert isinstance(result.output, Message)
        assert result.output.content[-1].text == "Done: x"
        assert len(scripted.requests) == 2

        first, second = scripted.requests
        for req in (first, second):
            assert REASONING_ENCRYPTED_CONTENT_INCLUDE in req["include"]
            assert req["store"] is False

        # Second request replays: user → reasoning(rs_1) → function_call(call_1) → function_call_output.
        kinds = [(i.get("type") or i.get("role")) for i in second["input"]]
        assert kinds == ["user", "reasoning", "function_call", "function_call_output"], kinds
        reasoning = second["input"][1]
        assert reasoning == {"type": "reasoning", "id": "rs_1", "encrypted_content": "enc-1", "summary": []}
        assert second["input"][2]["call_id"] == "call_1"
        assert second["input"][3]["call_id"] == "call_1"
        # The reasoning payload is never leaked as visible assistant text.
        assert not any(i.get("role") == "assistant" for i in second["input"])

    @pytest.mark.asyncio
    async def test_three_step_loop_accumulates_every_reasoning_item(self):
        scripted = _ScriptedResponses(
            [
                _tool_call_turn("rs_1", "enc-1", "call_1", "fc_1", "search", '{"q": "a"}', summary="step one"),
                _tool_call_turn("rs_2", "enc-2", "call_2", "fc_2", "search", '{"q": "b"}'),
                _text_turn("rs_3", "enc-3", "a and b"),
            ]
        )
        _, calls = await _run_agent(scripted)

        assert calls == [{"q": "a"}, {"q": "b"}]
        third = scripted.requests[2]["input"]
        kinds = [(i.get("type") or i.get("role")) for i in third]
        assert kinds == [
            "user",
            "reasoning",
            "function_call",
            "function_call_output",
            "reasoning",
            "function_call",
            "function_call_output",
        ], kinds
        assert [i["id"] for i in third if i.get("type") == "reasoning"] == ["rs_1", "rs_2"]
        assert [i["encrypted_content"] for i in third if i.get("type") == "reasoning"] == ["enc-1", "enc-2"]
        assert third[1]["summary"] == [{"type": "summary_text", "text": "step one"}]
        # Every reasoning item sits directly before the function_call it produced.
        for idx, item in enumerate(third):
            if item.get("type") == "reasoning":
                assert third[idx + 1]["type"] == "function_call"

    @pytest.mark.asyncio
    async def test_final_memory_keeps_reasoning_items_for_the_next_turn(self):
        scripted = _ScriptedResponses(
            [
                _tool_call_turn("rs_1", "enc-1", "call_1", "fc_1", "search", '{"q": "x"}'),
                _text_turn("rs_2", "enc-2", "Done"),
            ]
        )
        result, _ = await _run_agent(scripted)
        final = result.output
        # The last assistant message carries its own reasoning item too, so a follow-up
        # user turn replays the complete chain.
        assert [type(c).__name__ for c in final.content] == ["ThinkingContent", "TextContent"]
        assert final.content[0].id == "rs_2" and final.content[0].encrypted_content == "enc-2"
        assert final.to_openai_responses_input()[0]["type"] == "reasoning"

    @pytest.mark.asyncio
    async def test_without_encrypted_content_nothing_is_replayed_and_nothing_breaks(self):
        """A server that returns no payload (include ignored) must not produce empty items."""
        scripted = _ScriptedResponses(
            [
                _tool_call_turn("rs_1", None, "call_1", "fc_1", "search", '{"q": "x"}'),
                _text_turn("rs_2", None, "Done"),
            ]
        )
        result, calls = await _run_agent(scripted)
        assert calls == [{"q": "x"}]
        second = scripted.requests[1]["input"]
        kinds = [(i.get("type") or i.get("role")) for i in second]
        assert kinds == ["user", "function_call", "function_call_output"], kinds
        assert [type(c).__name__ for c in result.output.content] == ["TextContent"]


# ---------------------------------------------------------------------------
# Leaked tool calls (Harmony text) run like real ones; `phase` survives the loop
# ---------------------------------------------------------------------------


def _text_turn_with_phase(rs_id: str, encrypted: str, text: str, phase: str | None):
    """`_text_turn` whose message items carry a `phase` (GPT-5.4+)."""
    events = _text_turn(rs_id, encrypted, text)
    out = []
    for e in events:
        item = getattr(e, "item", None)
        if isinstance(item, ResponseOutputMessage) and phase:
            e = e.model_copy(update={"item": item.model_copy(update={"phase": phase})})
        out.append(e)
    return out


class TestAgentRecoversLeakedToolCalls:
    @pytest.mark.asyncio
    async def test_leaked_call_text_runs_the_tool_and_the_loop_continues(self):
        """Turn 1 is a `message` that says ` to=functions.search json {"q":"x"}` instead of a
        function_call item. The tool must run and turn 2 must see a real function_call +
        output in its input — never the leaked text as assistant prose."""
        leak = ' to=functions.search  (json 恒一\n{"q":"x"}ંалда'
        scripted = _ScriptedResponses(
            [
                _text_turn("rs_1", "enc-1", leak),
                _text_turn("rs_2", "enc-2", "Done: x"),
            ]
        )
        result, calls = await _run_agent(scripted)

        assert calls == [{"q": "x"}]
        assert result.output.content[-1].text == "Done: x"
        second = scripted.requests[1]["input"]
        kinds = [(i.get("type") or i.get("role")) for i in second]
        assert kinds == ["user", "reasoning", "function_call", "function_call_output"], kinds
        assert second[2]["name"] == "search"
        assert second[2]["call_id"].startswith("call_leak_")
        assert second[3]["call_id"] == second[2]["call_id"]
        assert not any("to=functions" in str(i) for i in second)

    @pytest.mark.asyncio
    async def test_assistant_phase_is_replayed_on_the_next_request(self):
        scripted = _ScriptedResponses(
            [
                _text_turn_with_phase("rs_1", "enc-1", "One moment.", "commentary"),
            ]
        )
        result, _ = await _run_agent(scripted)
        assert result.output.content[-1].phase == "commentary"
        # A follow-up turn replays the assistant message with its phase intact.
        items = result.output.to_openai_responses_input()
        assert items[-1] == {"role": "assistant", "content": [{"type": "output_text", "text": "One moment."}], "phase": "commentary"}


class TestAgentRetriesUnrecoverableLeak:
    @pytest.mark.asyncio
    async def test_leak_without_arguments_is_re_requested_and_the_loop_completes(self):
        """Turn 1 leaks `to=functions.search (json…` with no JSON: nothing to run. The agent
        must re-request (nudge appended, empty assistant turn not kept), turn 2 calls the
        tool properly, turn 3 answers."""
        scripted = _ScriptedResponses(
            [
                _text_turn("rs_1", "enc-1", " to=functions.search  (jsonеиҳәеит?)\n"),
                _tool_call_turn("rs_2", "enc-2", "call_1", "fc_1", "search", '{"q": "x"}'),
                _text_turn("rs_3", "enc-3", "Done: x"),
            ]
        )
        result, calls = await _run_agent(scripted)

        assert calls == [{"q": "x"}]
        assert result.output.content[-1].text == "Done: x"
        assert len(scripted.requests) == 3
        second = scripted.requests[1]["input"]
        kinds = [(i.get("type") or i.get("role")) for i in second]
        # user prompt, then the runtime nudge — no dangling reasoning item, no empty assistant message
        assert kinds == ["user", "user"], kinds
        assert "emitted as text" in second[1]["content"][0]["text"]
        assert not any(i.get("type") == "reasoning" for i in second)

    @pytest.mark.asyncio
    async def test_retry_budget_is_bounded(self):
        leak = " to=functions.search  (json\n"
        scripted = _ScriptedResponses([_text_turn(f"rs_{n}", f"enc-{n}", leak) for n in range(1, 5)])
        result, calls = await _run_agent(scripted)
        assert calls == []
        # 1 original + 2 retries (max_leaked_tool_call_retries) = 3 requests; then the turn ends
        assert len(scripted.requests) == 3
        assert result.output.metadata.get("kind") == "leaked_tool_call_unrecovered"
