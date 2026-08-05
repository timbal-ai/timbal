"""Tests for state/tracing/trace.py — Trace container."""

import pytest

from timbal.state.tracing.span import Span
from timbal.state.tracing.trace import Trace


def _make_span(call_id: str, parent_call_id: str | None = None, path: str = "agent") -> Span:
    return Span(
        path=path,
        call_id=call_id,
        parent_call_id=parent_call_id,
        t0=0,
    )


class TestTraceInit:
    def test_empty_init(self):
        trace = Trace()
        assert len(trace) == 0
        assert trace._root_call_id is None

    def test_init_from_none(self):
        trace = Trace(None)
        assert len(trace) == 0

    def test_init_from_list(self):
        records = [
            {"path": "agent", "call_id": "c1", "parent_call_id": None, "t0": 0}
        ]
        trace = Trace(records)
        assert "c1" in trace
        assert trace._root_call_id == "c1"

    def test_init_from_list_missing_call_id_raises(self):
        with pytest.raises(ValueError, match="Missing call_id"):
            Trace([{"path": "agent", "parent_call_id": None, "t0": 0}])

    def test_init_from_dict(self):
        data = {
            "c1": {"path": "agent", "call_id": "c1", "parent_call_id": None, "t0": 0}
        }
        trace = Trace(data)
        assert "c1" in trace

    def test_init_from_dict_missing_call_id_raises(self):
        data = {"c1": {"path": "agent", "parent_call_id": None, "t0": 0}}
        with pytest.raises(ValueError, match="Missing call_id"):
            Trace(data)

    def test_init_invalid_type_raises(self):
        with pytest.raises(ValueError, match="Invalid trace data type"):
            Trace(42)  # type: ignore[arg-type]


class TestTraceSetItem:
    def test_set_span_directly(self):
        trace = Trace()
        span = _make_span("c1", parent_call_id=None)
        trace["c1"] = span
        assert trace["c1"] is span
        assert trace._root_call_id == "c1"

    def test_set_dict_value_converted_to_span(self):
        trace = Trace()
        trace["c1"] = {"path": "agent", "call_id": "c1", "parent_call_id": None, "t0": 0}
        assert isinstance(trace["c1"], Span)

    def test_non_string_key_raises(self):
        trace = Trace()
        with pytest.raises((ValueError, Exception)):
            trace[123] = _make_span("c1")  # type: ignore[index]

    def test_invalid_value_type_raises(self):
        trace = Trace()
        with pytest.raises(ValueError, match="Invalid trace value type"):
            trace["c1"] = "not a span"  # type: ignore[assignment]

    def test_dict_missing_parent_call_id_raises(self):
        trace = Trace()
        with pytest.raises((ValueError, TypeError)):
            trace["c1"] = {
                "type": "OUTPUT",
                "run_id": "r1",
                "path": "agent",
                "call_id": "c1",
                # missing parent_call_id
            }

    def test_multiple_root_calls_raises(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None)
        with pytest.raises(ValueError, match="Cannot set multiple root calls"):
            trace["c2"] = _make_span("c2", parent_call_id=None)

    def test_child_span_does_not_set_root(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None)
        trace["c2"] = _make_span("c2", parent_call_id="c1")
        assert trace._root_call_id == "c1"


class TestTraceGetLevel:
    def test_get_level_returns_matching_spans(self):
        trace = Trace()
        # root: "agent"  (level 1, depth 0)
        # child: "agent.step_a" (level 2, depth 1)
        trace["c1"] = _make_span("c1", parent_call_id=None, path="agent")
        trace["c2"] = _make_span("c2", parent_call_id="c1", path="agent.step_a")
        trace["c3"] = _make_span("c3", parent_call_id="c1", path="agent.step_b")

        # get_level("agent") returns spans one level deeper: path has 1 dot and starts with "agent"
        result = trace.get_level("agent")
        call_ids = {s.call_id for s in result}
        assert call_ids == {"c2", "c3"}

    def test_get_level_no_match(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None, path="agent")
        assert trace.get_level("agent") == []

    def test_get_path_returns_exact_matches(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None, path="agent")
        trace["c2"] = _make_span("c2", parent_call_id="c1", path="agent")

        result = trace.get_path("agent")
        assert len(result) == 2

    def test_get_path_no_match(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None, path="agent")
        assert trace.get_path("other") == []


class TestTraceModelDump:
    def test_model_dump_returns_list(self):
        trace = Trace()
        trace["c1"] = _make_span("c1", parent_call_id=None)
        trace["c2"] = _make_span("c2", parent_call_id="c1")
        result = trace.model_dump()
        assert isinstance(result, list)
        assert len(result) == 2

    def test_as_records_returns_spans(self):
        trace = Trace()
        span = _make_span("c1", parent_call_id=None)
        trace["c1"] = span
        records = trace.as_records()
        assert records == [span]
