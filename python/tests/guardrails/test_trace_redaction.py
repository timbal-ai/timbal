"""Trace-boundary redaction: trace_redactor + TracingProvider._trace_redactor."""

import json

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.guardrails import DetectPII, LLMJudge, trace_redactor
from timbal.state.tracing.providers import JsonlTracingProvider
from timbal.state.tracing.providers.base import Exporter

SSN_PROMPT = "my ssn is 123-45-6789 and my email is joe@example.com"


class TestTraceRedactorCallable:
    def test_walks_nested_structures(self):
        redact = trace_redactor("pii:redact")
        value = {
            "text": "ssn 123-45-6789",
            "nested": [{"email": "joe@x.com"}, ("tuple", "ip 10.0.0.1")],
            "number": 42,
            "none": None,
        }
        out = redact(value)
        assert out["text"] == "ssn [REDACTED_SSN]"
        assert out["nested"][0]["email"] == "[REDACTED_EMAIL]"
        assert out["nested"][1][1] == "ip [REDACTED_IP]"
        assert out["number"] == 42 and out["none"] is None
        # original untouched (walker rebuilds, never mutates)
        assert value["text"] == "ssn 123-45-6789"

    def test_default_battery_is_pii_plus_secrets(self):
        redact = trace_redactor()
        assert redact("key sk-abcdefghijklmnopqrstuvwx") == "key [REDACTED_OPENAI_KEY]"
        assert redact("ssn 123-45-6789") == "ssn [REDACTED_SSN]"

    def test_rejects_llm_rails(self):
        with pytest.raises(ValueError, match="deterministic"):
            trace_redactor(LLMJudge("no medical advice"))

    def test_accepts_configured_rail_instances(self):
        redact = trace_redactor(DetectPII(types=["ssn"], redaction="hash"))
        out = redact("ssn 123-45-6789 email joe@x.com")
        assert "<ssn_hash:" in out
        assert "joe@x.com" in out  # only ssn configured


class _CapturingExporter(Exporter):
    def __init__(self):
        self.traces = []

    async def export(self, run_context) -> None:
        self.traces.append(run_context._trace)


class TestProviderIntegration:
    def _provider(self, tmp_path, **kwargs):
        return JsonlTracingProvider.configured(
            _path=tmp_path / "traces.jsonl",
            _trace_redactor=trace_redactor(),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_stored_trace_is_redacted_including_inner_llm_span(self, tmp_path):
        provider = self._provider(tmp_path)
        agent = Agent(
            name="a",
            model=TestModel(handler=lambda msgs: "noted: " + msgs[-1].collect_text()),
            tools=[],
            tracing_provider=provider,
        )
        result = await agent(prompt=SSN_PROMPT).collect()
        assert result.status.code == "success"

        raw = (tmp_path / "traces.jsonl").read_text()
        assert "123-45-6789" not in raw
        assert "joe@example.com" not in raw
        assert "[REDACTED_SSN]" in raw

        # specifically: the inner LLM child span (path a.llm) is redacted — the edge
        # in-run guardrails cannot reach.
        record = json.loads(raw.splitlines()[-1])
        llm_spans = [s for s in record["spans"] if s["path"] == "a.llm"]
        assert llm_spans, "expected the inner llm span in the stored trace"
        assert "123-45-6789" not in json.dumps(llm_spans)

    @pytest.mark.asyncio
    async def test_live_run_is_never_mutated(self, tmp_path):
        provider = self._provider(tmp_path)
        agent = Agent(
            name="a",
            model=TestModel(handler=lambda msgs: "echo: " + msgs[-1].collect_text()),
            tools=[],
            tracing_provider=provider,
        )
        result = await agent(prompt=SSN_PROMPT).collect()
        # No agent-level guardrails: the live output keeps the raw text — redaction
        # applies only at the storage/export boundary.
        assert "123-45-6789" in result.output.collect_text()

    @pytest.mark.asyncio
    async def test_resumed_session_loads_redacted_history(self, tmp_path):
        provider = self._provider(tmp_path)
        model = TestModel(handler=lambda msgs: f"history: {msgs[0].collect_text()}")
        agent = Agent(name="a", model=model, tools=[], tracing_provider=provider)
        first = await agent(prompt=SSN_PROMPT).collect()

        second = await agent(prompt="follow up", parent_id=first.run_id).collect()
        seen = second.output.collect_text()
        assert "123-45-6789" not in seen
        assert "[REDACTED_SSN]" in seen

    @pytest.mark.asyncio
    async def test_exporters_receive_the_redacted_view(self, tmp_path):
        exporter = _CapturingExporter()
        provider = self._provider(tmp_path, _exporters=[exporter])
        agent = Agent(
            name="a",
            model=TestModel(responses=["fine"]),
            tools=[],
            tracing_provider=provider,
        )
        await agent(prompt=SSN_PROMPT).collect()

        assert exporter.traces
        exported = json.dumps(exporter.traces[-1].model_dump(), default=str)
        assert "123-45-6789" not in exported
        assert "[REDACTED_SSN]" in exported

    @pytest.mark.asyncio
    async def test_no_redactor_stores_raw(self, tmp_path):
        provider = JsonlTracingProvider.configured(_path=tmp_path / "raw.jsonl")
        agent = Agent(name="a", model=TestModel(responses=["ok"]), tools=[], tracing_provider=provider)
        await agent(prompt=SSN_PROMPT).collect()
        assert "123-45-6789" in (tmp_path / "raw.jsonl").read_text()
