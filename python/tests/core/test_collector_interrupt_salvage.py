"""Regression: interrupted runs must salvage partial output when ``collector.result()`` raises.

Before ``_collector_output_on_interrupt`` (and fallbacks), ``GeneratorExit`` used
``try/except Exception: pass`` around ``collector.result()`` — a raising ``result()``
left ``span.output`` unset even when ``MessageCollector``-style ``_message`` was
populated.  ``CancelledError`` called ``result()`` unguarded and could fail after
``span.status`` was already set.

Unit tests exercise the helper directly (would fail if fallbacks/logging regressed).
Integration tests use a registered collector that only matches a private marker type.

Note: a pure ``GeneratorExit`` integration test that stalls the handler is awkward
because CPython forbids ``aclose()`` while ``__anext__`` is running on the same agen
(``RuntimeError: ... already running``).  The cancel-then-``aclose()`` test below
primarily exercises the ``CancelledError`` salvage path; ``except GeneratorExit`` uses
the same helper and is covered by the unit tests plus existing tests in
``test_interruptions.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

import pytest
from timbal import Tool
from timbal.collectors import get_collector_registry
from timbal.collectors.base import BaseCollector
from timbal.core.runnable import _collector_output_on_interrupt
from timbal.state import set_run_context
from timbal.state.context import RunContext
from timbal.types.content import TextContent
from timbal.types.events import OutputEvent
from timbal.types.message import Message
from timbal.types.run_status import RunStatus


class _CollectorInterruptMarker:
    """First yield from test handler — only ``_BrokenResultCollector`` matches."""


class _BrokenResultCollector(BaseCollector):
    """``process`` fills ``_message``; ``result`` always raises (strict / incomplete API)."""

    @classmethod
    def can_handle(cls, event: Any) -> bool:
        return type(event) is _CollectorInterruptMarker

    def process(self, _event: Any) -> None:
        self._message = Message(
            role="assistant",
            content=[TextContent(text="salvaged-partial")],
        )
        return None

    def result(self) -> Any:
        raise RuntimeError("simulated: result() not valid until stream complete")


# Register once; marker type is unique to this module — no collision with other tests.
get_collector_registry().register(_BrokenResultCollector)


class TestCollectorOutputOnInterruptUnit:
    def test_result_success_message(self):
        class Ok:
            def result(self):
                return Message(role="assistant", content=[TextContent(text="ok")])

        out = _collector_output_on_interrupt(Ok())
        assert isinstance(out, Message)
        assert out.collect_text() == "ok"

    def test_result_success_unwraps_output_event(self):
        inner = Message(role="assistant", content=[TextContent(text="inner")])

        class OkEv:
            def result(self):
                return OutputEvent(
                    run_id="r",
                    path="p",
                    call_id="c",
                    parent_call_id=None,
                    input={},
                    status=RunStatus(code="success", reason="stop", message=None),
                    output=inner,
                    error=None,
                    t0=0,
                    t1=1,
                    usage={},
                    metadata={},
                )

        assert _collector_output_on_interrupt(OkEv()) is inner

    def test_result_raises_falls_back_to_message(self):
        class BadMsg:
            _message = Message(role="assistant", content=[TextContent(text="fb")])

            def result(self):
                raise RuntimeError("boom")

        out = _collector_output_on_interrupt(BadMsg())
        assert isinstance(out, Message)
        assert out.collect_text() == "fb"

    def test_result_raises_falls_back_to_output_event(self):
        inner = Message(role="assistant", content=[TextContent(text="ev")])

        class BadEv:
            _output_event = OutputEvent(
                run_id="r",
                path="p",
                call_id="c",
                parent_call_id=None,
                input={},
                status=RunStatus(code="success", reason="stop", message=None),
                output=inner,
                error=None,
                t0=0,
                t1=1,
                usage={},
                metadata={},
            )

            def result(self):
                raise RuntimeError("boom")

        assert _collector_output_on_interrupt(BadEv()) is inner

    def test_result_raises_no_fallback_returns_none(self):
        class BadEmpty:
            def result(self):
                raise RuntimeError("boom")

        assert _collector_output_on_interrupt(BadEmpty()) is None


@pytest.mark.asyncio
class TestCollectorInterruptRunnableIntegration:
    """Would fail if GeneratorExit / CancelledError paths swallowed or skipped salvage."""

    @staticmethod
    async def _stall_handler() -> Any:
        yield _CollectorInterruptMarker()
        await asyncio.sleep(3600)

    async def test_cancel_stalled_anext_then_aclose_salvages_span_output(self):
        """Cancel a stuck ``__anext__``, then ``aclose()`` — span keeps salvaged ``Message``.

        The second ``__anext__`` blocks inside the collector while the handler sleeps, so
        ``_execute_handler`` has created the collector and ``process()`` has set ``_message``.
        Cancelling the waiter surfaces ``CancelledError`` into ``Runnable.__call__`` (salvage
        path).  ``aclose()`` finishes cleanup without ``aclose(): ... already running``.
        """
        run_context = RunContext()
        set_run_context(run_context)

        tool = Tool(name="salvage_genexit", handler=self._stall_handler)
        gen = tool().__aiter__()
        await gen.__anext__()  # StartEvent
        waiter = asyncio.create_task(gen.__anext__())
        await asyncio.sleep(0.08)
        waiter.cancel()
        with contextlib.suppress(asyncio.CancelledError, RuntimeError):
            await waiter
        await gen.aclose()

        spans = [s for s in run_context._trace.values() if s.path == "salvage_genexit"]
        assert len(spans) == 1
        span = spans[0]
        assert span.status is not None
        assert span.status.code == "cancelled"
        assert isinstance(span.output, Message)
        assert span.output.collect_text() == "salvaged-partial"

    async def test_cancelled_collect_output_salvaged(self):
        run_context = RunContext()
        set_run_context(run_context)

        tool = Tool(name="salvage_cancel", handler=self._stall_handler)
        task = asyncio.create_task(tool().collect())
        await asyncio.sleep(0.15)
        task.cancel()
        out = await task

        assert isinstance(out, OutputEvent)
        assert out.status.code == "cancelled"
        assert isinstance(out.output, Message)
        assert out.output.collect_text() == "salvaged-partial"

        spans = [s for s in run_context._trace.values() if s.path == "salvage_cancel"]
        assert len(spans) == 1
        assert spans[0].output.collect_text() == "salvaged-partial"
