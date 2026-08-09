"""GuardrailRunner: ordering, shadow mode, strict crashes, buffering decisions, scrubbing."""

import pytest
from timbal.guardrails import Guardrail, GuardrailContext, GuardrailRunner, GuardrailStage, Verdict, guardrail
from timbal.guardrails.runner import StreamScrubber
from timbal.guardrails.types import GuardrailMatch

INPUT = GuardrailStage.INPUT
OUTPUT = GuardrailStage.MODEL_OUTPUT


def _ctx(stage=INPUT):
    return GuardrailContext(stage=stage)


class _WordRail(Guardrail):
    """Deterministic rail matching a configured word."""

    word: str = "bad"

    def detect(self, text):
        out = []
        start = 0
        while (idx := text.find(self.word, start)) != -1:
            out.append(GuardrailMatch(kind=self.word, start=idx, end=idx + len(self.word), text=self.word))
            start = idx + len(self.word)
        return out


class TestRunnerBasics:
    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError, match="Duplicate guardrail name"):
            GuardrailRunner([_WordRail(name="x"), _WordRail(name="x")])

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="Invalid guardrail mode"):
            GuardrailRunner([], mode="audit")

    def test_stage_filtering(self):
        runner = GuardrailRunner([_WordRail(name="a", stages={INPUT}), _WordRail(name="b", stages={OUTPUT})])
        assert [r.name for r in runner.stage_rails(INPUT)] == ["a"]
        assert runner.has_stage(OUTPUT) and not runner.has_stage(GuardrailStage.TOOL_ARGS)

    def test_merged_with_combines_rails(self):
        base = GuardrailRunner([_WordRail(name="a")], mode="shadow", max_retries=5)
        merged = base.merged_with([_WordRail(name="b")])
        assert [r.name for r in merged.rails] == ["a", "b"]
        assert merged.mode == "shadow" and merged.max_retries == 5
        assert base.merged_with(None) is base
        assert base.merged_with([]) is base


class TestRunStage:
    @pytest.mark.asyncio
    async def test_allow_when_nothing_triggers(self):
        runner = GuardrailRunner([_WordRail(action="block")])
        outcome = await runner.run_stage(INPUT, "all fine", _ctx())
        assert outcome.verdict is None and not outcome.triggered and outcome.text == "all fine"

    @pytest.mark.asyncio
    async def test_block_wins_and_is_recorded(self):
        runner = GuardrailRunner([_WordRail(name="w", action="block")])
        outcome = await runner.run_stage(INPUT, "so bad", _ctx())
        assert outcome.verdict is not None and outcome.verdict.action == "block"
        assert outcome.rail.name == "w"
        assert [t.rail for t in outcome.triggered] == ["w"]
        assert outcome.triggered[0].action == "block" and not outcome.triggered[0].shadow

    @pytest.mark.asyncio
    async def test_mutating_rails_chain_in_list_order(self):
        runner = GuardrailRunner(
            [
                _WordRail(name="first", word="bad", action="redact"),
                # Second rail sees the first's replacement text.
                guardrail(lambda t: t.replace("[REDACTED_BAD]", "<clean>"), stages=["input"], name="second", action="redact"),
            ]
        )
        outcome = await runner.run_stage(INPUT, "bad stuff", _ctx())
        assert outcome.replaced
        assert outcome.text == "<clean> stuff"

    @pytest.mark.asyncio
    async def test_rail_after_a_redactor_sees_the_redacted_text(self):
        """The documented "normalize first, judge second" ordering must actually hold.

        A block rail used to be checked concurrently against the *original* text no
        matter where it sat in the list, so it both judged pre-redaction content and
        received the raw PII a redactor was placed in front of it to strip.
        """
        seen: list[str] = []

        def judge(text):
            seen.append(text)
            return Verdict.block("saw raw text") if "bad" in text else True

        runner = GuardrailRunner(
            [
                _WordRail(name="redactor", word="bad", action="redact"),
                guardrail(judge, stages=["input"], name="judge", action="block"),
            ]
        )
        outcome = await runner.run_stage(INPUT, "bad stuff", _ctx())
        assert seen == ["[REDACTED_BAD] stuff"], "the judge must be handed the redacted text"
        assert outcome.verdict is None, "nothing should block — the offending text was already scrubbed"

    @pytest.mark.asyncio
    async def test_rail_before_a_redactor_still_sees_the_original(self):
        seen: list[str] = []

        runner = GuardrailRunner(
            [
                guardrail(lambda t: seen.append(t) or True, stages=["input"], name="judge", action="block"),
                _WordRail(name="redactor", word="bad", action="redact"),
            ]
        )
        await runner.run_stage(INPUT, "bad stuff", _ctx())
        assert seen == ["bad stuff"]

    @pytest.mark.asyncio
    async def test_non_mutating_rails_still_run_concurrently(self):
        """The position fix must not serialize rails that cannot affect each other."""
        import asyncio

        running = 0
        peak = 0

        async def slow(_text):
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return True

        runner = GuardrailRunner(
            [guardrail(slow, stages=["input"], name=f"r{i}", action="block") for i in range(4)]
        )
        await runner.run_stage(INPUT, "text", _ctx())
        assert peak == 4, f"expected all 4 rails in flight together, saw {peak}"

    @pytest.mark.asyncio
    async def test_undeclared_text_rewrite_is_a_loud_error(self):
        """A plain callable returning a str coerces to `replace`, but `guardrail()`
        defaults to action="block". Batching is decided from the action, so allowing
        this would silently check later rails against pre-rewrite text — and would have
        already handed them the un-sanitized content. Fail instead of guessing."""
        runner = GuardrailRunner(
            [
                guardrail(lambda t: t.replace("bad", "ok"), stages=["input"], name="sanitize"),
                guardrail(lambda _t: True, stages=["input"], name="judge"),
            ]
        )
        with pytest.raises(ValueError, match="must declare action='redact'"):
            await runner.run_stage(INPUT, "bad stuff", _ctx())

    @pytest.mark.asyncio
    async def test_declared_redactor_feeds_later_rails(self):
        seen: list[str] = []

        runner = GuardrailRunner(
            [
                guardrail(
                    lambda t: t.replace("bad", "ok"), stages=["input"], name="sanitize", action="redact"
                ),
                guardrail(lambda t: seen.append(t) or True, stages=["input"], name="judge"),
            ]
        )
        outcome = await runner.run_stage(INPUT, "bad stuff", _ctx())
        assert outcome.text == "ok stuff"
        assert seen == ["ok stuff"]

    @pytest.mark.asyncio
    async def test_dict_replacement_needs_no_redact_action(self):
        """Tool-arg replacement does not touch the text, so it is not an ordering hazard
        and keeps working with the default action."""
        runner = GuardrailRunner(
            [guardrail(lambda _t: Verdict.replace({"env": "staging"}), stages=["tool_args"], name="downgrade")]
        )
        outcome = await runner.run_stage(GuardrailStage.TOOL_ARGS, '{"env": "prod"}', _ctx(GuardrailStage.TOOL_ARGS))
        assert outcome.replacement_args == {"env": "staging"}
        assert outcome.text == '{"env": "prod"}'

    @pytest.mark.asyncio
    async def test_shadowed_redactor_does_not_change_what_later_rails_see(self):
        seen: list[str] = []

        runner = GuardrailRunner(
            [
                _WordRail(name="redactor", word="bad", action="redact", shadow=True),
                guardrail(lambda t: seen.append(t) or True, stages=["input"], name="judge"),
            ]
        )
        await runner.run_stage(INPUT, "bad stuff", _ctx())
        assert seen == ["bad stuff"], "a shadowed rail must not affect the pipeline"

    @pytest.mark.asyncio
    async def test_first_blocking_rail_in_list_order_controls(self):
        runner = GuardrailRunner(
            [
                _WordRail(name="one", action="block"),
                _WordRail(name="two", action="block"),
            ]
        )
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.rail.name == "one"
        # both are still recorded
        assert {t.rail for t in outcome.triggered} == {"one", "two"}

    @pytest.mark.asyncio
    async def test_warn_never_controls(self):
        runner = GuardrailRunner([_WordRail(action="warn")])
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.verdict is None
        assert outcome.triggered[0].action == "warn"


class TestShadowMode:
    @pytest.mark.asyncio
    async def test_global_shadow_records_but_never_enforces(self):
        runner = GuardrailRunner([_WordRail(action="block"), _WordRail(name="r2", action="redact")], mode="shadow")
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.verdict is None
        assert outcome.text == "bad"  # no mutation either
        assert all(t.shadow for t in outcome.triggered)
        assert {t.action for t in outcome.triggered} == {"block", "replace"}

    @pytest.mark.asyncio
    async def test_per_rail_shadow(self):
        runner = GuardrailRunner([_WordRail(name="shadowed", action="block", shadow=True)])
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.verdict is None
        assert outcome.triggered[0].shadow


class TestCrashPolicy:
    @pytest.mark.asyncio
    async def test_fail_open_by_default(self):
        def boom(_text):
            raise RuntimeError("kaput")

        runner = GuardrailRunner([guardrail(boom, stages=["input"], name="boom")])
        outcome = await runner.run_stage(INPUT, "anything", _ctx())
        assert outcome.verdict is None
        assert outcome.triggered[0].action == "error"
        assert outcome.triggered[0].error == "RuntimeError"

    @pytest.mark.asyncio
    async def test_strict_fails_closed(self):
        def boom(_text):
            raise RuntimeError("kaput")

        runner = GuardrailRunner([guardrail(boom, stages=["input"], name="boom", strict=True)])
        outcome = await runner.run_stage(INPUT, "anything", _ctx())
        assert outcome.verdict is not None and outcome.verdict.action == "block"

    @pytest.mark.asyncio
    async def test_strict_shadow_still_fails_open(self):
        def boom(_text):
            raise RuntimeError("kaput")

        runner = GuardrailRunner(
            [guardrail(boom, stages=["input"], name="boom", strict=True, shadow=True)]
        )
        outcome = await runner.run_stage(INPUT, "anything", _ctx())
        assert outcome.verdict is None


class TestBufferingDecision:
    def test_redact_only_deterministic_rails_stream(self):
        runner = GuardrailRunner([_WordRail(action="redact")])
        assert not runner.needs_buffering(INPUT)
        assert runner.stream_scrubber(INPUT) is not None

    def test_block_rails_force_buffering(self):
        runner = GuardrailRunner([_WordRail(action="redact"), _WordRail(name="b", action="block")])
        assert runner.needs_buffering(INPUT)

    def test_non_streamable_redact_forces_buffering(self):
        class JudgeRedact(Guardrail):
            action: str = "redact"

            async def check(self, text, ctx):  # noqa: ARG002
                return Verdict.redact("x")

        runner = GuardrailRunner([JudgeRedact()])
        assert runner.needs_buffering(GuardrailStage.MODEL_OUTPUT) or runner.needs_buffering(INPUT)

    def test_shadow_rails_never_buffer(self):
        runner = GuardrailRunner([_WordRail(action="block")], mode="shadow")
        assert not runner.needs_buffering(INPUT)
        assert runner.stream_scrubber(INPUT) is None


class TestStreamScrubber:
    def test_pattern_spanning_chunk_boundary_is_caught(self):
        rail = _WordRail(word="secretword", action="redact", scrub_window=32)
        scrubber = StreamScrubber([rail])
        emitted = ""
        for chunk in ["talk about secr", "etword and other ", "things that pad the window far enough out"]:
            emitted += scrubber.feed(chunk)
        emitted += scrubber.flush()
        assert "secretword" not in emitted
        assert "[REDACTED_SECRETWORD]" in emitted

    def test_flush_releases_holdback(self):
        rail = _WordRail(word="zz", action="redact", scrub_window=64)
        scrubber = StreamScrubber([rail])
        assert scrubber.feed("short") == ""  # held back inside the window
        assert scrubber.flush() == "short"

    def test_shadow_rails_do_not_scrub(self):
        rail = _WordRail(word="bad", action="redact", shadow=True)
        scrubber = StreamScrubber([rail])
        assert scrubber.flush() == ""
        scrubber.feed("bad")
        assert scrubber.flush() == "bad"


class TestSampleRate:
    @pytest.mark.asyncio
    async def test_rate_zero_never_runs(self):
        runner = GuardrailRunner([_WordRail(action="block", shadow=True, sample_rate=0.0)])
        for _ in range(5):
            outcome = await runner.run_stage(INPUT, "bad", _ctx())
            assert not outcome.triggered

    @pytest.mark.asyncio
    async def test_rate_one_always_runs(self):
        runner = GuardrailRunner([_WordRail(action="warn", sample_rate=1.0)])
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.triggered

    @pytest.mark.asyncio
    async def test_fractional_rate_follows_the_roll(self, monkeypatch):
        import timbal.guardrails.runner as runner_module

        runner = GuardrailRunner([_WordRail(action="warn", shadow=True, sample_rate=0.5)])

        monkeypatch.setattr(runner_module.random, "random", lambda: 0.4)  # 0.4 < 0.5 → runs
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.triggered

        monkeypatch.setattr(runner_module.random, "random", lambda: 0.6)  # 0.6 >= 0.5 → sampled out
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert not outcome.triggered

    @pytest.mark.asyncio
    async def test_sampled_out_rail_neither_enforces_nor_mutates(self, monkeypatch):
        import timbal.guardrails.runner as runner_module

        monkeypatch.setattr(runner_module.random, "random", lambda: 0.99)
        runner = GuardrailRunner(
            [
                _WordRail(name="sampled", action="redact", sample_rate=0.5),
                _WordRail(name="always", action="warn"),
            ]
        )
        outcome = await runner.run_stage(INPUT, "bad", _ctx())
        assert outcome.text == "bad"  # sampled-out redactor never touched it
        assert [t.rail for t in outcome.triggered] == ["always"]

    def test_invalid_rate_rejected(self):
        with pytest.raises(ValueError):
            _WordRail(sample_rate=1.5)

    def test_describe_shows_fractional_rates_only(self):
        runner = GuardrailRunner(
            [_WordRail(name="sampled", shadow=True, sample_rate=0.1), _WordRail(name="full")]
        )
        rows = {r["name"]: r for r in runner.describe()}
        assert rows["sampled"]["sample_rate"] == 0.1
        assert "sample_rate" not in rows["full"]


class TestMultiStageHelpers:
    def test_needs_buffering_across_stages(self):
        runner = GuardrailRunner(
            [
                _WordRail(name="redactor", stages={OUTPUT}, action="redact"),
                _WordRail(name="blocker", stages={GuardrailStage.MODEL_STEP}, action="block"),
            ]
        )
        assert not runner.needs_buffering(OUTPUT)
        assert runner.needs_buffering(GuardrailStage.MODEL_STEP)
        assert runner.needs_buffering(OUTPUT, GuardrailStage.MODEL_STEP)

    def test_scrub_rails_dedupes_across_stages(self):
        rail = _WordRail(name="r", stages={OUTPUT, GuardrailStage.MODEL_STEP}, action="redact")
        runner = GuardrailRunner([rail])
        assert runner.scrub_rails(OUTPUT, GuardrailStage.MODEL_STEP) == [rail]

    def test_scrub_text_applies_all_redact_rails(self):
        runner = GuardrailRunner(
            [
                _WordRail(name="a", word="foo", stages={OUTPUT}, action="redact"),
                _WordRail(name="b", word="bar", stages={GuardrailStage.MODEL_STEP}, action="redact"),
                _WordRail(name="c", word="baz", stages={OUTPUT}, action="block"),  # not a scrub rail
            ]
        )
        out = runner.scrub_text("foo bar baz", OUTPUT, GuardrailStage.MODEL_STEP)
        assert out == "[REDACTED_FOO] [REDACTED_BAR] baz"

    def test_stream_scrubber_combines_stages(self):
        runner = GuardrailRunner(
            [
                _WordRail(name="a", word="foo", stages={OUTPUT}, action="redact"),
                _WordRail(name="b", word="bar", stages={GuardrailStage.MODEL_STEP}, action="redact"),
            ]
        )
        scrubber = runner.stream_scrubber(OUTPUT, GuardrailStage.MODEL_STEP)
        scrubber.feed("foo and bar")
        assert scrubber.flush() == "[REDACTED_FOO] and [REDACTED_BAR]"


class TestDescribe:
    def test_rows_include_stages_actions_flags(self):
        runner = GuardrailRunner([_WordRail(name="w", action="redact", on_output="block", strict=True)])
        [row] = runner.describe()
        assert row["name"] == "w"
        assert row["actions"]["model_output"] == "block"
        assert row["actions"]["input"] == "redact"
        assert row["strict"] and not row["shadow"]
