"""End-to-end streaming guardrail tests with real DeltaEvents.

These drive the agent's actual delta handling — in-flight scrubbing with holdback
windows, per-content-block scrubbers, tail flushing, buffer-until-verdict, retry
clearing — through a model that streams like a real provider.
"""

import pytest
from timbal import Agent
from timbal.guardrails import DetectPII, GuardrailStage, Verdict, guardrail
from timbal.types.events import DeltaEvent, GuardrailEvent, OutputEvent
from timbal.types.events.delta import TextDelta, ThinkingDelta

from .conftest import StreamingTestModel, text_stream, thinking_stream, tool_use_item

SSN_TEXT = "the customer ssn is 123-45-6789 and that is all"


def _text_deltas(events, path_suffix=".llm"):
    return [
        e.item.text_delta
        for e in events
        if isinstance(e, DeltaEvent) and isinstance(e.item, TextDelta) and e.path.endswith(path_suffix)
    ]


def _thinking_deltas(events):
    return [e.item.thinking_delta for e in events if isinstance(e, DeltaEvent) and isinstance(e.item, ThinkingDelta)]


def _final(events):
    return next(e for e in reversed(events) if isinstance(e, OutputEvent))


class TestPassthroughControl:
    @pytest.mark.asyncio
    async def test_no_guardrails_streams_unmodified(self):
        agent = Agent(name="a", model=StreamingTestModel([text_stream(SSN_TEXT)]), tools=[])
        events = [e async for e in agent(prompt="go")]
        assert "".join(_text_deltas(events)) == SSN_TEXT
        assert _final(events).output.collect_text() == SSN_TEXT


class TestTransformMode:
    @pytest.mark.asyncio
    async def test_pattern_split_across_chunks_is_scrubbed_in_flight(self):
        """chunk_size=7 splits the SSN across multiple deltas — the holdback window
        must reassemble and scrub it before any chunk escapes."""
        agent = Agent(
            name="a",
            model=StreamingTestModel([text_stream(SSN_TEXT, chunk_size=7)]),
            tools=[],
            guardrails=[DetectPII(stages={GuardrailStage.MODEL_OUTPUT}, action="redact", types=["ssn"])],
        )
        events = [e async for e in agent(prompt="go")]
        streamed = "".join(_text_deltas(events))
        assert "123-45-6789" not in streamed
        assert "[REDACTED_SSN]" in streamed

    @pytest.mark.asyncio
    async def test_streamed_text_equals_final_text(self):
        """Deltas (including the tail flush) must reassemble into exactly the stored,
        scrubbed message — no dropped or duplicated characters."""
        agent = Agent(
            name="a",
            model=StreamingTestModel([text_stream(SSN_TEXT, chunk_size=5)]),
            tools=[],
            guardrails=[DetectPII(stages={GuardrailStage.MODEL_OUTPUT}, action="redact", types=["ssn"])],
        )
        events = [e async for e in agent(prompt="go")]
        assert "".join(_text_deltas(events)) == _final(events).output.collect_text()

    @pytest.mark.asyncio
    async def test_clean_stream_passes_through_completely(self):
        clean = "nothing sensitive in here at all, just a normal answer"
        agent = Agent(
            name="a",
            model=StreamingTestModel([text_stream(clean, chunk_size=9)]),
            tools=[],
            guardrails=["pii:redact"],
        )
        events = [e async for e in agent(prompt="go")]
        assert "".join(_text_deltas(events)) == clean

    @pytest.mark.asyncio
    async def test_thinking_deltas_scrubbed_with_separate_block_scrubbers(self):
        """Thinking and text stream as separate content blocks — each gets its own
        holdback buffer, so scrubbing one never stitches content into the other."""
        script = [
            *thinking_stream("note: ssn 123-45-6789 must stay hidden", chunk_size=6),
            *text_stream("I cannot share that information.", chunk_size=6),
        ]
        agent = Agent(
            name="a",
            model=StreamingTestModel([script]),
            tools=[],
            guardrails=[DetectPII(stages={GuardrailStage.MODEL_OUTPUT}, action="redact", types=["ssn"])],
        )
        events = [e async for e in agent(prompt="go")]
        thinking = "".join(_thinking_deltas(events))
        assert "123-45-6789" not in thinking
        assert "[REDACTED_SSN]" in thinking
        text = "".join(_text_deltas(events))
        assert text == "I cannot share that information."
        # the stored message's thinking block is scrubbed too
        final = _final(events)
        thinking_blocks = [c for c in final.output.content if getattr(c, "type", "") == "thinking"]
        assert "[REDACTED_SSN]" in thinking_blocks[0].thinking

    @pytest.mark.asyncio
    async def test_intermediate_tool_call_stream_is_scrubbed(self):
        """Transform mode scrubs the tool-calling step's prose too — memory must match
        the scrubbed deltas that already went out."""
        calls = []

        def lookup(q: str) -> str:
            calls.append(q)
            return "found it"

        scripts = [
            [
                *text_stream("checking ssn 123-45-6789 in the system", chunk_size=6),
                tool_use_item("lookup", {"q": "x"}),
            ],
            text_stream("done, no sensitive data shared"),
        ]
        agent = Agent(
            name="a",
            model=StreamingTestModel(scripts),
            tools=[lookup],
            guardrails=[DetectPII(stages={GuardrailStage.MODEL_OUTPUT}, action="redact", types=["ssn"])],
        )
        events = [e async for e in agent(prompt="go")]
        assert _final(events).status.code == "success"
        assert calls == ["x"], "the tool call must still execute"
        streamed = "".join(_text_deltas(events))
        assert "123-45-6789" not in streamed


class TestBufferUntilVerdict:
    def _blocking_agent(self, scripts, **agent_kwargs):
        return Agent(
            name="a",
            model=StreamingTestModel(scripts),
            tools=[],
            guardrails=[
                guardrail(
                    lambda t: Verdict.block("forbidden content") if "FORBIDDEN" in t else True,
                    stages=["model_output"],
                    name="forbidden",
                )
            ],
            **agent_kwargs,
        )

    @pytest.mark.asyncio
    async def test_blocked_response_leaks_zero_deltas(self):
        agent = self._blocking_agent([text_stream("this contains FORBIDDEN material", chunk_size=6)])
        events = [e async for e in agent(prompt="go")]
        assert _text_deltas(events) == [], "buffer mode must withhold every chunk of a blocked response"
        final = _final(events)
        assert final.status.code == "blocked"
        assert final.output.collect_text() == "The response was withheld by a content policy."
        assert any(isinstance(e, GuardrailEvent) and e.action == "block" for e in events)

    @pytest.mark.asyncio
    async def test_allowed_response_replays_deltas_in_order_before_output(self):
        clean = "perfectly acceptable answer streaming through"
        agent = self._blocking_agent([text_stream(clean, chunk_size=8)])
        events = [e async for e in agent(prompt="go")]
        assert "".join(_text_deltas(events)) == clean
        # deltas replay before the final OutputEvent
        last_delta_idx = max(i for i, e in enumerate(events) if isinstance(e, DeltaEvent))
        final_idx = events.index(_final(events))
        assert last_delta_idx < final_idx

    @pytest.mark.asyncio
    async def test_retry_drops_rejected_draft_deltas_entirely(self):
        def no_pineapple(text):
            return Verdict.retry("No pineapple.") if "pineapple" in text else True

        model = StreamingTestModel(
            [
                text_stream("try pizza with pineapple today", chunk_size=6),
                text_stream("try pizza with mushrooms today", chunk_size=6),
            ]
        )
        agent = Agent(
            name="a",
            model=model,
            tools=[],
            guardrails=[guardrail(no_pineapple, stages=["model_output"], name="no_pineapple")],
        )
        events = [e async for e in agent(prompt="go")]
        streamed = "".join(_text_deltas(events))
        assert "pineapple" not in streamed, "the rejected draft must never reach the stream"
        assert streamed == "try pizza with mushrooms today"
        assert model.call_count == 2
        assert _final(events).status.code == "success"

    @pytest.mark.asyncio
    async def test_thinking_deltas_are_withheld_too(self):
        script = [
            *thinking_stream("planning FORBIDDEN reveal", chunk_size=6),
            *text_stream("this contains FORBIDDEN material", chunk_size=6),
        ]
        agent = self._blocking_agent([script])
        events = [e async for e in agent(prompt="go")]
        assert _thinking_deltas(events) == []
        assert _text_deltas(events) == []
        assert _final(events).status.code == "blocked"


class TestModelStepStreaming:
    @pytest.mark.asyncio
    async def test_step_block_on_intermediate_withholds_its_stream(self):
        def lookup(q: str) -> str:  # noqa: ARG001
            return "data"

        scripts = [
            [
                *text_stream("leaking PROJECT_TITAN details now", chunk_size=6),
                tool_use_item("lookup", {"q": "x"}),
            ],
            text_stream("done"),
        ]
        agent = Agent(
            name="a",
            model=StreamingTestModel(scripts),
            tools=[lookup],
            guardrails=[
                guardrail(
                    lambda t: Verdict.block("codename") if "PROJECT_TITAN" in t else True,
                    stages=["model_step"],
                    name="codename",
                )
            ],
        )
        events = [e async for e in agent(prompt="go")]
        assert _text_deltas(events) == [], "the blocked intermediate step must not stream"
        final = _final(events)
        assert final.status.code == "blocked"
        assert final.status.reason == "guardrail:codename:model_step"
