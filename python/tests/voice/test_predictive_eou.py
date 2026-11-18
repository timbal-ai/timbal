# ruff: noqa: ARG002
"""During-speech end-of-turn prediction.

The live path is opt-in. These tests cover the agreement rule, the endpointer
timing that rule produces, Flux predictions not becoming commits, and a replay
of the voice scenario scripts: decision latency versus false cuts.
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.voice import (
    SpeechEouPredictor,
    SpeechToText,
    TextToSpeech,
    TranscriptCommitted,
    TranscriptEvent,
    VadEndpointer,
    VoiceConfig,
    VoiceSession,
    endpointing_delay,
)
from timbal.voice.eou import AudioEouModel
from timbal.voice.predictive_eou import (
    AUDIO_COMPLETE,
    DEFAULT_MAX_DELAY_SECS,
    DEFAULT_MIN_DELAY_SECS,
    DEFAULT_STOP_SILENCE_SECS,
    EAGER_SILENCE_SECS,
    EouSignals,
    ScriptPause,
    baseline_decision_secs,
    endpoint_curve,
    evaluate_replay,
    flux_eager_threshold,
    prediction_is_complete,
    script_pauses,
    text_blocks_early,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeVad:
    async def start(self, *, sample_rate: int) -> None:
        return None

    def reset(self) -> None:
        return None

    def process(self, chunk: bytes) -> list[float]:
        return [1.0 if byte == ord("S") else 0.0 for byte in chunk]


class _Scores:
    def __init__(self, p_audio: float, *, delay: float = 0.0) -> None:
        self.p_audio = p_audio
        self.delay = delay
        self.audio_calls = 0

    async def audio(self) -> float:
        self.audio_calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.p_audio

    async def text(self, text: str) -> float:
        return 0.95


class _Gate:
    def __init__(self) -> None:
        self.commits = 0

    async def score(self) -> float:
        return 0.1

    async def commit(self) -> None:
        self.commits += 1

    def should_commit(self) -> bool:
        return True


def _predictor(scores: _Scores, **kwargs: float) -> SpeechEouPredictor:
    return SpeechEouPredictor(score_audio=scores.audio, score_text=scores.text, **kwargs)


async def _armed(ep: VadEndpointer, gate: _Gate) -> None:
    ep.bind(score=gate.score, commit=gate.commit, should_commit=gate.should_commit)
    await ep.start(sample_rate=16_000)


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------


def _signals(**overrides: object) -> EouSignals:
    base = dict(
        p_audio=0.9,
        audio_fresh=True,
        p_text=0.95,
        text_fresh=True,
        text_veto=False,
    )
    base.update(overrides)
    return EouSignals(**base)  # type: ignore[arg-type]


class TestAgreement:
    def test_all_signals_agree(self) -> None:
        assert prediction_is_complete(_signals())

    def test_rolling_score_is_not_a_vote(self) -> None:
        assert not prediction_is_complete(_signals(p_audio=None, audio_fresh=False, p_rolling=0.99))

    def test_audio_below_the_model_cut_keeps_the_floor(self) -> None:
        assert not prediction_is_complete(_signals(p_audio=AUDIO_COMPLETE - 0.01))

    def test_stale_audio_does_not_agree(self) -> None:
        assert not prediction_is_complete(_signals(audio_fresh=False))

    def test_incomplete_text_does_not_agree(self) -> None:
        assert not prediction_is_complete(_signals(p_text=0.2))

    def test_lexical_veto_blocks_a_confident_model(self) -> None:
        assert not prediction_is_complete(_signals(text_veto=True))

    def test_flux_resume_vetoes(self) -> None:
        assert not prediction_is_complete(_signals(flux_state="resumed", p_flux=0.99))

    def test_low_flux_eager_confidence_vetoes(self) -> None:
        assert not prediction_is_complete(_signals(flux_state="eager", p_flux=0.2))

    def test_high_flux_eager_still_needs_audio(self) -> None:
        assert prediction_is_complete(_signals(flux_state="eager", p_flux=0.8))
        assert not prediction_is_complete(_signals(flux_state="eager", p_flux=0.8, audio_fresh=False, p_audio=None))

    @pytest.mark.parametrize(
        "text",
        ["I need help with", "Uh...", "Send it to account 447. Sorry.", "and"],
    )
    async def test_unfinished_words_block_the_short_wait(self, text: str) -> None:
        assert await text_blocks_early(text)

    async def test_a_finished_sentence_does_not_block(self) -> None:
        assert not await text_blocks_early("Hello there.")

    def test_curve_matches_the_endpointer(self) -> None:
        assert DEFAULT_STOP_SILENCE_SECS == VadEndpointer.STOP_SILENCE_SECS
        assert DEFAULT_MIN_DELAY_SECS == VadEndpointer.MIN_DELAY_SECS
        assert DEFAULT_MAX_DELAY_SECS == VadEndpointer.MAX_DELAY_SECS
        for p in (0.0, 0.2, 0.5, 0.95, 1.0):
            assert endpoint_curve(p) == pytest.approx(endpointing_delay(p))


class TestPredictor:
    async def test_partial_is_scored_and_flux_resume_clears_eager(self) -> None:
        scores = _Scores(0.9)
        predictor = _predictor(scores)
        predictor.note_partial("Hello there.")
        predictor.note_flux("eager", "Hello there.", 0.8)
        ready = await predictor.signals()
        assert ready.text_fresh
        assert ready.p_text == pytest.approx(0.95)
        assert ready.flux_state == "eager"
        assert prediction_is_complete(ready) is False  # no pause-onset audio yet
        predictor.note_flux("resumed", "Hello there actually", 0.1)
        resumed = await predictor.signals()
        assert resumed.flux_state == "resumed"
        assert not prediction_is_complete(resumed)
        await predictor.close()

    async def test_rolling_audio_does_not_authorize_an_early_commit(self) -> None:
        scores = _Scores(0.99)
        predictor = _predictor(scores)
        predictor.note_partial("Hello there.")
        predictor.note_speech()
        await asyncio.sleep(0)
        ready = await predictor.signals()
        assert ready.p_rolling == pytest.approx(0.99)
        assert scores.audio_calls == 1
        assert not ready.audio_fresh
        assert not prediction_is_complete(ready)
        await predictor.close()

    async def test_pause_onset_score_is_the_audio_vote(self) -> None:
        scores = _Scores(0.91)
        predictor = _predictor(scores)
        predictor.note_partial("Hello there.")
        predictor.kick_audio()
        ready = await predictor.signals()
        assert ready.audio_fresh
        assert ready.p_audio == pytest.approx(0.91)
        assert prediction_is_complete(ready)
        await predictor.close()

    async def test_flux_resume_does_not_stick_to_the_next_pause(self) -> None:
        scores = _Scores(0.91)
        predictor = _predictor(scores)
        predictor.note_partial("Hello there.")
        predictor.note_flux("resumed", "Hello there", 0.2)
        blocked = await predictor.signals()
        assert blocked.flux_state == "resumed"
        assert not prediction_is_complete(blocked)
        predictor.on_speech_resumed()
        predictor.kick_audio()
        nxt = await predictor.signals()
        assert nxt.flux_state == "none"
        assert prediction_is_complete(nxt)
        await predictor.close()

    async def test_resumed_speech_drops_the_pause_score(self) -> None:
        scores = _Scores(0.91, delay=0.05)
        predictor = _predictor(scores)
        predictor.kick_audio()
        predictor.on_speech_resumed()
        ready = await predictor.signals()
        assert not ready.audio_fresh
        await predictor.close()


class TestEndpointerTiming:
    def _endpointer(self, predictor: SpeechEouPredictor) -> VadEndpointer:
        ep = VadEndpointer(
            _FakeVad(),
            stop_silence_secs=0.096,
            min_speech_secs=0.064,
            min_delay_secs=0.25,
            max_delay_secs=0.25,
            min_commit_interval_secs=0.0,
        )
        ep.attach_predictor(predictor)
        return ep

    async def test_agreement_commits_on_the_short_silence(self) -> None:
        scores = _Scores(0.95)
        predictor = _predictor(scores, eager_silence_secs=0.096, predict_silence_secs=0.032)
        gate = _Gate()
        ep = self._endpointer(predictor)
        await _armed(ep, gate)
        predictor.note_partial("Hello there.")
        await asyncio.sleep(0)
        # Two speech frames, then three silent ones: 96 ms, the eager confirmation.
        ep.push(b"SS" + b"\x00" * 3)
        await asyncio.sleep(0.05)
        assert gate.commits == 1
        assert ep.last_predictive is True
        # The silence floor would still be running.
        assert 0.05 < 0.25
        await ep.close()
        await predictor.close()

    async def test_disagreement_keeps_the_silence_floor(self) -> None:
        scores = _Scores(0.1)
        predictor = _predictor(scores, eager_silence_secs=0.096, predict_silence_secs=0.032)
        gate = _Gate()
        ep = self._endpointer(predictor)
        await _armed(ep, gate)
        predictor.note_partial("Hello there.")
        await asyncio.sleep(0)
        ep.push(b"SS" + b"\x00" * 3)
        await asyncio.sleep(0.05)
        assert gate.commits == 0
        await asyncio.sleep(0.35)
        assert gate.commits == 1
        assert ep.last_predictive is False
        await ep.close()
        await predictor.close()

    async def test_resumed_speech_cancels_the_short_commit(self) -> None:
        scores = _Scores(0.95, delay=0.2)
        predictor = _predictor(scores, eager_silence_secs=0.096, predict_silence_secs=0.032)
        gate = _Gate()
        ep = self._endpointer(predictor)
        await _armed(ep, gate)
        predictor.note_partial("Hello there.")
        await asyncio.sleep(0)
        ep.push(b"SS" + b"\x00" * 3)
        ep.push(b"SS")
        await asyncio.sleep(0.35)
        assert gate.commits == 0
        await ep.close()
        await predictor.close()


# ---------------------------------------------------------------------------
# Session: Flux predictions are not turns
# ---------------------------------------------------------------------------


class _ScriptSTT(SpeechToText):
    def __init__(self, script: list[TranscriptEvent]) -> None:
        self._script = script

    async def connect(self, config: object) -> None:
        return None

    async def push_audio(self, chunk: bytes) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        for event in self._script:
            yield event

    async def close(self) -> None:
        return None


class _QuietTTS(TextToSpeech):
    async def connect(self, config: object) -> None:
        return None

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        yield b"\x00\x01" * 8

    async def close(self) -> None:
        return None


class TestSession:
    async def _turns(self, *, predict: bool) -> tuple[list[str], list[str]]:
        stt = _ScriptSTT(
            [
                TranscriptEvent(type="eager", text="Hello there", confidence=0.9),
                TranscriptEvent(type="resumed", text="Hello there actually", confidence=0.2),
                TranscriptEvent(type="partial", text="Hello there."),
                TranscriptEvent(type="committed", text="Hello there."),
            ]
        )
        session = VoiceSession(
            agent=Agent(name="t", model=TestModel(responses=["Hi."]), tools=[]),
            stt=stt,
            tts=_QuietTTS(),
            turn_detector="heuristic",
            vad_endpointing=False,
            predict_during_speech=predict,
        )

        async def _silent() -> AsyncIterator[bytes]:
            return
            yield b""

        events = []
        async for event in session.run(_silent()):
            events.append(event)
        committed = [event.text for event in events if isinstance(event, TranscriptCommitted)]
        spoken = [entry.text for entry in session.transcript if entry.role == "user"]
        return committed, spoken

    async def test_predictions_do_not_start_a_turn_when_off(self) -> None:
        committed, spoken = await self._turns(predict=False)
        assert committed == ["Hello there."]
        assert spoken == ["Hello there."]

    async def test_eager_starts_a_turn_and_resume_cancels_it(self) -> None:
        committed, transcript = await self._turns(predict=True)
        assert "Hello there" in committed
        assert transcript == ["Hello there."]

    def test_config_defaults_off(self) -> None:
        assert VoiceConfig().predict_during_speech is False


# ---------------------------------------------------------------------------
# Replay corpus
# ---------------------------------------------------------------------------


def _load_scenarios() -> list:
    import sys

    path = Path(__file__).resolve().parents[3] / "benchmarks" / "voice" / "scenario.py"
    spec = importlib.util.spec_from_file_location("voice_replay_scenario", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the class module through sys.modules during import.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(module.SCENARIOS)


class TestReplay:
    async def test_script_pauses_split_a_fluent_gap_from_the_turn_end(self) -> None:
        scenarios = _load_scenarios()
        support = next(s for s in scenarios if s.id == "support_pause_short")
        pauses = script_pauses(support.script)
        intra = [p for p in pauses if p.continues]
        ends = [p for p in pauses if not p.continues]
        assert intra and intra[0].gap_secs == pytest.approx(0.6)
        assert any(not p.continues for p in pauses)
        assert ends[-1].text.startswith("I need help with")

    async def test_replay_false_cuts_and_decision_latency(self) -> None:
        pauses = [pause for scenario in _load_scenarios() for pause in script_pauses(scenario.script)]
        report = await evaluate_replay(pauses)
        # Fluent fragments in the scripts are lexically unfinished, so neither
        # the guarded policy nor a text-only policy cuts them. Finished turns
        # that read as sentences take the short confirmation.
        assert report.intra > 0
        assert report.turns > 0
        assert report.false_cuts == 0
        assert report.baseline_false_cuts == 0
        assert report.text_only_false_cuts == 0
        assert report.qualified_turn_ends > 0
        assert report.median_qualified_predictive_ms == pytest.approx(EAGER_SILENCE_SECS * 1000.0)
        assert report.median_qualified_baseline_ms == pytest.approx(baseline_decision_secs(0.95) * 1000.0)
        saved_ms = report.median_qualified_baseline_ms - report.median_qualified_predictive_ms
        assert saved_ms == pytest.approx(496.375)
        assert report.median_predictive_ms <= report.median_baseline_ms

    async def test_audio_prior_blocks_a_punctuated_mid_sentence_fragment(self) -> None:
        """The measured Smart Turn score on a fluent cut, once STT adds a period.

        ``benchmarks/voice/scenario.py`` records that cut at 0.019 complete.
        Text alone would confirm it 160 ms into a 1.5 s pause.
        """
        report = await evaluate_replay(
            [ScriptPause("I want to return an item I bought.", continues=True, gap_secs=1.5)]
        )
        assert report.false_cuts == 0
        assert report.text_only_false_cuts == 1


class _FixedAudio(AudioEouModel):
    def __init__(self, p: float) -> None:
        self.p = p

    async def predict_complete(self, pcm: bytes, *, sample_rate: int) -> float:
        return self.p


def _agent() -> Agent:
    return Agent(name="voice_test", model=TestModel(responses=["Hi."]), tools=[])


def _stub_tts(session: VoiceSession) -> None:
    """Keep a built session from opening a real speech socket."""

    async def connect(_config: object) -> None:
        return None

    async def close() -> None:
        return None

    async def synthesize(_text: str) -> AsyncIterator[bytes]:
        yield b"\x00\x01" * 4

    session.tts.connect = connect  # type: ignore[method-assign]
    session.tts.close = close  # type: ignore[method-assign]
    session.tts.synthesize = synthesize  # type: ignore[method-assign]
    session.tts.open_stream = lambda: None  # type: ignore[method-assign]


class TestBuildVoiceSession:
    """Prediction wired by the server, for Flux and for Scribe."""

    def test_flux_eager_threshold_and_vad_stays_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, meta = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        assert meta["stt_provider"] == "deepgram-flux"
        assert meta["turn_detector"] == "ProviderTurnDetector"
        assert session._vad_endpointing is False
        assert session._predict_during_speech is True
        assert session.eager_eot_threshold == pytest.approx(0.5)
        assert session.audio_input.extra["eager_eot_threshold"] == "0.5"

    def test_flux_keeps_a_caller_eager_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(
            _agent(),
            VoiceConfig(predict_during_speech=True, stt_extra={"eager_eot_threshold": 0.4}),
            {},
        )
        assert session.eager_eot_threshold == pytest.approx(0.4)
        assert session.audio_input.extra["eager_eot_threshold"] == "0.4"

    def test_threshold_stays_under_a_low_eot(self) -> None:
        assert flux_eager_threshold({}) == pytest.approx(0.5)
        assert flux_eager_threshold({"eot_threshold": 0.4}) == pytest.approx(0.3)
        assert flux_eager_threshold({"eot_threshold": 0.3}) == pytest.approx(0.2)
        assert flux_eager_threshold({"eot_threshold": 0.3}) < 0.3
        assert flux_eager_threshold({"eager_eot_threshold": "0.45"}) == pytest.approx(0.45)
        assert flux_eager_threshold({"eot_threshold": 0.4, "eager_eot_threshold": 0.4}) < 0.4

    async def test_flux_eager_commits_resume_cancels_end_confirms(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._maybe_start_endpointer()
        assert session._eou_predictor is not None
        assert session._endpointer is None

        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.2)
        )
        assert session._eager_unconfirmed is None

        session._predict_audio_model = _FixedAudio(0.1)
        session._predict_audio_model_loaded = True
        session._push_predict_pcm(b"\x01\x00" * 8000)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        assert session._eager_unconfirmed is None

        session._predict_audio_model = _FixedAudio(0.95)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        assert session._eager_unconfirmed == "Hello there."
        assert [entry.text for entry in session.transcript if entry.role == "user"] == ["Hello there."]

        await session._handle_provider_prediction(
            TranscriptEvent(type="resumed", text="Hello there actually", confidence=0.1)
        )
        assert session._eager_unconfirmed is None
        assert [entry.text for entry in session.transcript if entry.role == "user"] == []

        await session._handle_provider_commit("Hello there actually.")
        assert [entry.text for entry in session.transcript if entry.role == "user"] == [
            "Hello there actually.",
        ]

    async def test_flux_end_of_turn_confirms_without_a_second_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        await session._handle_provider_commit("Hello there?")
        assert session._eager_unconfirmed is None
        assert [entry.text for entry in session.transcript if entry.role == "user"] == ["Hello there."]

    async def test_punctuation_only_final_confirms_without_restarting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        cancelled: list[str] = []

        async def cancel() -> None:
            cancelled.append("yes")

        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there", confidence=0.9)
        )
        session._cancel_unconfirmed_eager = cancel  # type: ignore[method-assign]
        await session._handle_provider_commit("Hello there.")
        assert cancelled == []
        assert session._eager_unconfirmed is None
        assert [entry.text for entry in session.transcript if entry.role == "user"] == ["Hello there"]

    async def test_ignored_eager_does_not_swallow_end_of_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._handle_provider_prediction(TranscriptEvent(type="eager", text="Uh...", confidence=0.9))
        assert session._eager_unconfirmed is None
        await session._handle_provider_commit("I need the white one.")
        assert [entry.text for entry in session.transcript if entry.role == "user"] == ["I need the white one."]

    async def test_longer_end_of_turn_replaces_the_short_eager_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        await session._handle_provider_commit("Hello there. And fries.")
        assert [entry.text for entry in session.transcript if entry.role == "user"] == [
            "Hello there. And fries.",
        ]

    async def test_resume_retracts_the_bubble_and_the_memory(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session
        from timbal.voice import TranscriptCommitted

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        emitted: list = []
        original = session._emit

        async def capture(event: object) -> None:
            emitted.append(event)
            await original(event)

        session._emit = capture  # type: ignore[method-assign]

        aligned: list[str] = []

        async def align(*, fragment_user_text: str) -> None:
            aligned.append(fragment_user_text)

        session._align_continue_memory = align  # type: ignore[method-assign]
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        await session._handle_provider_prediction(
            TranscriptEvent(type="resumed", text="Hello there more", confidence=0.1)
        )
        assert [entry.text for entry in session.transcript if entry.role == "user"] == []
        assert aligned == ["Hello there."]
        retracted = [
            event
            for event in emitted
            if isinstance(event, TranscriptCommitted) and event.replace and event.text == ""
        ]
        assert retracted

    async def test_cancel_keeps_memory_when_the_reply_stays_on_screen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session
        from timbal.voice import TranscriptEntry

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        _stub_tts(session)
        await session._handle_provider_prediction(
            TranscriptEvent(type="eager", text="Hello there.", confidence=0.9)
        )
        session._transcript.append(TranscriptEntry(role="assistant", text="Sure."))
        session._active_turn_user_text = ""
        aligned: list[str] = []

        async def align(*, fragment_user_text: str) -> None:
            aligned.append(fragment_user_text)

        async def interrupt() -> None:
            return None

        session._align_continue_memory = align  # type: ignore[method-assign]
        session.interrupt = interrupt  # type: ignore[method-assign]
        await session._cancel_unconfirmed_eager()
        assert aligned == []
        assert [entry.role for entry in session.transcript] == ["user", "assistant"]

    async def test_audio_vote_does_not_wait_for_a_cold_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session

        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        session, _ = build_voice_session(_agent(), VoiceConfig(predict_during_speech=True), {})
        session._predict_audio_model_loaded = False
        session._push_predict_pcm(b"\x01\x00" * 8000)
        started = time.monotonic()
        assert await session._audio_vote() is None
        assert time.monotonic() - started < 0.05

    async def test_scribe_keeps_the_local_endpointer_and_confirms_early(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from timbal.server.voice import build_voice_session
        from timbal.voice import LocalAudioTurnDetector
        from timbal.voice.eou import PunctuationEouPredictor

        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        detector = LocalAudioTurnDetector(
            audio_eou=_FixedAudio(0.95),
            fallback_text_eou=PunctuationEouPredictor(),
        )
        session, meta = build_voice_session(
            _agent(),
            VoiceConfig(
                stt_provider="elevenlabs",
                predict_during_speech=True,
                turn_detector=detector,
            ),
            {},
        )
        assert meta["stt_provider"] == "elevenlabs"
        assert session._predict_during_speech is True
        assert session._vad_endpointing is None
        session._vad_endpointing = VadEndpointer(
            _FakeVad(),
            stop_silence_secs=0.096,
            min_speech_secs=0.064,
            min_delay_secs=0.25,
            max_delay_secs=0.25,
            min_commit_interval_secs=0.0,
        )
        commits: list[int] = []

        async def commit() -> None:
            commits.append(1)

        session.stt.commit = commit
        await session._maybe_start_endpointer()
        assert session._endpointer is not None
        assert session._endpointer._predictor is session._eou_predictor
        await session.turn_detector.start(session.audio_input)
        session.turn_detector.push_audio(b"\x01\x00" * 8000)
        assert session._eou_predictor is not None
        session._eou_predictor.note_partial("Hello there.")
        session._latest_partial_text = "Hello there."
        session._last_partial_at = time.monotonic()
        session._endpointer.push(b"SS" + bytes(3))
        # Armed at 96 ms of silence; the short confirmation is 160 ms.
        await asyncio.sleep(0.15)
        assert commits == [1]
        assert session._endpointer.last_predictive is True
