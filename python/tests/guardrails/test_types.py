"""Guardrail core types: verdicts, coercion, stage/action resolution, callable wrapping."""

import pytest
from timbal.guardrails import Guardrail, GuardrailContext, GuardrailStage, Verdict, coerce_verdict, guardrail
from timbal.guardrails.types import GuardrailMatch


class TestVerdict:
    def test_helpers(self):
        assert Verdict.allow().action == "allow"
        assert not Verdict.allow().triggered
        b = Verdict.block("bad", blocked_message="Nope.")
        assert b.action == "block" and b.reason == "bad" and b.blocked_message == "Nope."
        r = Verdict.redact("clean")
        assert r.action == "replace" and r.replacement == "clean"
        rt = Verdict.retry("do better", reason="quality")
        assert rt.action == "retry" and rt.feedback == "do better"
        e = Verdict.escalate("approve this?")
        assert e.action == "escalate" and e.approval_prompt == "approve this?"

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError, match="Invalid verdict action"):
            Verdict(action="explode")


class TestCoerceVerdict:
    def test_true_and_none_allow(self):
        assert coerce_verdict(True).action == "allow"
        assert coerce_verdict(None).action == "allow"

    def test_false_blocks(self):
        assert coerce_verdict(False).action == "block"

    def test_str_replaces(self):
        v = coerce_verdict("replacement text")
        assert v.action == "replace" and v.replacement == "replacement text"

    def test_dict_replaces_tool_args(self):
        v = coerce_verdict({"q": "[REDACTED]"})
        assert v.action == "replace" and v.replacement == {"q": "[REDACTED]"}

    def test_verdict_passthrough(self):
        v = Verdict.block("x")
        assert coerce_verdict(v) is v

    def test_garbage_rejected_loudly(self):
        with pytest.raises(ValueError, match="expected bool, None, str, dict, or Verdict"):
            coerce_verdict(42)


class _StubRail(Guardrail):
    """Deterministic rail matching the literal word 'bad'."""

    name: str = "stub"

    def detect(self, text):
        out = []
        start = 0
        while (idx := text.find("bad", start)) != -1:
            out.append(GuardrailMatch(kind="bad_word", start=idx, end=idx + 3, text="bad"))
            start = idx + 3
        return out


class TestGuardrailBase:
    def test_name_defaults_to_snake_case(self):
        assert _StubRail().name == "stub"

        class MyCustomRail(_StubRail):
            name: str = ""

        assert MyCustomRail().name == "my_custom_rail"

    def test_invalid_action_rejected(self):
        with pytest.raises(ValueError, match="Invalid guardrail action"):
            _StubRail(action="obliterate")
        with pytest.raises(ValueError, match="Invalid guardrail action"):
            _StubRail(on_output="obliterate")

    def test_per_stage_override_wins(self):
        rail = _StubRail(action="redact", on_output="block")
        assert rail.action_for(GuardrailStage.INPUT) == "redact"
        assert rail.action_for(GuardrailStage.MODEL_OUTPUT) == "block"

    def test_stage_override_implicitly_opts_in(self):
        rail = _StubRail(stages={GuardrailStage.INPUT}, on_tool_result="redact")
        assert rail.runs_on(GuardrailStage.TOOL_RESULT)
        assert not rail.runs_on(GuardrailStage.TOOL_ARGS)

    def test_scrub_replaces_all_matches(self):
        assert _StubRail().scrub("bad things are bad") == "[REDACTED_BAD_WORD] things are [REDACTED_BAD_WORD]"

    @pytest.mark.asyncio
    async def test_default_check_maps_action_to_verdict(self):
        ctx = GuardrailContext(stage=GuardrailStage.MODEL_OUTPUT)
        assert (await _StubRail(action="block").check("bad", ctx)).action == "block"
        assert (await _StubRail(action="warn").check("bad", ctx)).action == "warn"
        redacted = await _StubRail(action="redact").check("bad", ctx)
        assert redacted.action == "replace" and redacted.replacement == "[REDACTED_BAD_WORD]"
        assert (await _StubRail(action="retry").check("bad", ctx)).action == "retry"
        assert (await _StubRail(action="escalate").check("bad", ctx)).action == "escalate"
        assert (await _StubRail().check("all good", ctx)).action == "allow"

    def test_streamable_only_for_detect_rails(self):
        assert _StubRail().streamable

        class JudgeLike(Guardrail):
            async def check(self, text, ctx):  # noqa: ARG002
                return True

        assert not JudgeLike().streamable


class TestGuardrailDecorator:
    @pytest.mark.asyncio
    async def test_wraps_sync_callable(self):
        rail = guardrail(lambda text: "bad" not in text, stages=["input"], name="no_bad")
        assert rail.name == "no_bad"
        assert rail.runs_on(GuardrailStage.INPUT) and not rail.runs_on(GuardrailStage.TOOL_ARGS)
        ctx = GuardrailContext(stage=GuardrailStage.INPUT)
        assert (await rail.check("bad", ctx)) is False
        assert (await rail.check("fine", ctx)) is True

    @pytest.mark.asyncio
    async def test_wraps_async_callable_with_ctx(self):
        async def check(text, ctx):
            assert ctx.stage == GuardrailStage.MODEL_OUTPUT
            return Verdict.warn("noted") if "hmm" in text else None

        rail = guardrail(check, stages=["model_output"])
        assert rail.name == "check"
        v = await rail.check("hmm", GuardrailContext(stage=GuardrailStage.MODEL_OUTPUT))
        assert v.action == "warn"

    def test_decorator_form(self):
        @guardrail(stages=["input"], action="warn")
        def screen(_text):
            return True

        assert screen.name == "screen"
        assert screen.action == "warn"

    def test_lambda_gets_stable_name(self):
        rail = guardrail(lambda _t: True)
        assert rail.name.startswith("guardrail_")
