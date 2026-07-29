"""Tool-call fillers: LLM-generated latency-masking speech while tools run.

Unit tests drive ``VoiceSession`` internals directly (TestModel + mock TTS);
integration tests go through the ``/voice/ws`` WebSocket like a real client.
"""

# ruff: noqa: ARG001  (tool/handler signatures are fixed by their callers)
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server import voice as voice_routes
from timbal.server.http import create_app
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message
from timbal.voice import FillerConfig, VoiceConfig, VoiceSession
from timbal.voice.playback import map_played_bytes_to_text

from .test_voice_ws import _collect_ws_messages, _make_stt_class, _make_tts_class
from .voice_env import VOICE_ENV_KEYS

PHRASE = "One moment please."


def _tool_call_message(tool: str = "lookup") -> Message:
    return Message(
        role="assistant",
        content=[ToolUseContent(id="c1", name=tool, input={"q": "x"})],
        stop_reason="tool_use",
    )


def _make_session(
    *,
    tool_sleep: float = 0.2,
    filler: dict | FillerConfig | None = None,
    responses: list | None = None,
) -> VoiceSession:
    async def lookup(q: str) -> str:
        await asyncio.sleep(tool_sleep)
        return "42"

    agent = Agent(
        name="voice_test",
        model=TestModel(responses=responses or [_tool_call_message(), "The answer is 42."]),
        tools=[lookup],
    )
    return VoiceSession(
        agent,
        _make_stt_class()(),
        _make_tts_class()(),
        turn_detector="heuristic",
        filler=filler,
    )


def _default_filler(**overrides) -> dict:
    return {"model": TestModel(responses=[PHRASE]), "delay_secs": 0.0, **overrides}


class TestFillerConfig:
    def test_defaults(self) -> None:
        cfg = FillerConfig()
        assert "voice assistant" in cfg.system_prompt
        assert cfg.model is None
        assert cfg.delay_secs == 1.0
        assert cfg.repeat_secs is None
        assert cfg.max_per_turn == 3

    def test_unknown_key_fails(self) -> None:
        with pytest.raises(ValidationError):
            FillerConfig(phrases=["hi"])

    def test_negative_delay_fails(self) -> None:
        with pytest.raises(ValidationError):
            FillerConfig(delay_secs=-1)

    def test_voice_config_default_is_off(self) -> None:
        assert VoiceConfig().filler is None

    def test_empty_dict_enables_defaults(self) -> None:
        cfg = VoiceConfig(filler={})
        assert cfg.filler is not None
        assert cfg.filler.delay_secs == 1.0


class TestFillerTurnFlow:
    """Full turns through ``VoiceSession._run_turn`` — no transports."""

    async def test_filler_spoken_during_slow_tool(self) -> None:
        session = _make_session(filler=_default_filler())
        await session._run_turn("what is the answer?")

        fillers = [e for e in session.transcript if e.filler]
        assert [e.text for e in fillers] == [PHRASE]
        replies = [e for e in session.transcript if e.role == "assistant" and not e.filler]
        assert [e.text for e in replies] == ["The answer is 42."]
        # Transcript order: the filler was spoken before the reply.
        assert session.transcript.index(fillers[0]) < session.transcript.index(replies[0])
        assert session.metrics[-1].filler_spoken is True

    async def test_filler_never_enters_agent_memory(self) -> None:
        session = _make_session(filler=_default_filler())
        await session._run_turn("what is the answer?")

        root = session._last_run_context.root_span()
        memory_text = " ".join(m.collect_text() for m in root.memory)
        assert PHRASE not in memory_text
        assert "The answer is 42." in memory_text
        assert PHRASE not in session._turn_assistant_text

    async def test_fast_tool_inside_grace_delay_no_filler(self) -> None:
        session = _make_session(tool_sleep=0.0, filler=_default_filler(delay_secs=30.0))
        await session._run_turn("quick one")

        assert not any(e.filler for e in session.transcript)
        assert session.metrics[-1].filler_spoken is False
        # The pending flow was cancelled by the turn teardown, not left dangling.
        assert session._turn_filler_task is not None
        assert session._turn_filler_task.done()

    async def test_text_before_tool_call_suppresses_filler(self) -> None:
        first = Message(
            role="assistant",
            content=[TextContent(text="Let me look."), ToolUseContent(id="c1", name="lookup", input={"q": "x"})],
            stop_reason="tool_use",
        )
        session = _make_session(filler=_default_filler(), responses=[first, "The answer is 42."])
        await session._run_turn("what is the answer?")

        assert not any(e.filler for e in session.transcript)
        assert session.metrics[-1].filler_spoken is False

    async def test_short_unflushed_preamble_suppresses_filler(self) -> None:
        """A streamed preamble below the TTS flush threshold (nothing scheduled
        yet) still means a spoken reply is coming — no filler on top of it."""
        first = Message(
            role="assistant",
            content=[TextContent(text="Sure,"), ToolUseContent(id="c1", name="lookup", input={"q": "x"})],
            stop_reason="tool_use",
        )
        session = _make_session(filler=_default_filler(), responses=[first, "The answer is 42."])
        await session._run_turn("what is the answer?")

        assert not any(e.filler for e in session.transcript)
        assert session.metrics[-1].filler_spoken is False

    async def test_generation_failure_is_silent(self) -> None:
        def _boom(messages):
            raise RuntimeError("generator down")

        session = _make_session(filler={"model": TestModel(handler=_boom), "delay_secs": 0.0})
        await session._run_turn("what is the answer?")

        assert not any(e.filler for e in session.transcript)
        replies = [e for e in session.transcript if e.role == "assistant"]
        assert [e.text for e in replies] == ["The answer is 42."]

    async def test_reply_queues_behind_slow_filler(self) -> None:
        """Tool finishes while the filler is mid-speech → the reply chains
        behind it via ``_tts_tail``; audio never interleaves."""
        from timbal.voice import AudioOutput, TextToSpeech

        class SlowTTS(TextToSpeech):
            async def connect(self, config) -> None:
                pass

            async def close(self) -> None:
                pass

            async def synthesize(self, text: str):
                marker = b"F" if text == PHRASE else b"R"
                for _ in range(4):
                    await asyncio.sleep(0.08)  # filler speaks for ~0.32s
                    yield marker * 32

        async def lookup(q: str) -> str:
            await asyncio.sleep(0.1)  # tool finishes while the filler is speaking
            return "42"

        agent = Agent(
            name="voice_test",
            model=TestModel(responses=[_tool_call_message(), "The answer is 42."]),
            tools=[lookup],
        )
        session = VoiceSession(
            agent, _make_stt_class()(), SlowTTS(), turn_detector="heuristic", filler=_default_filler()
        )
        await session._run_turn("what is the answer?")

        markers = []
        while not session._event_queue.empty():
            ev = session._event_queue.get_nowait()
            if isinstance(ev, AudioOutput):
                markers.append(ev.data[:1])
        # Both spoke in full (4 chunks each), filler strictly before reply.
        assert markers == [b"F"] * 4 + [b"R"] * 4
        # Records in playback order: barge-in mapping attributes filler bytes
        # to no text and reply bytes to the reply.
        assert [(t, b) for t, b in session._turn_tts_segment_records] == [
            ("", 128),
            ("The answer is 42.", 128),
        ]
        assert session.metrics[-1].filler_spoken is True

    async def test_filler_disabled_by_default(self) -> None:
        session = _make_session()
        await session._run_turn("what is the answer?")
        assert session.filler is None
        assert not any(e.filler for e in session.transcript)


class TestFillerRepeat:
    """Long turns: re-arm on prolonged silence (``repeat_secs``)."""

    @staticmethod
    def _counting_model() -> TestModel:
        n = {"v": 0}

        def handler(messages):
            n["v"] += 1
            return f"Filler {n['v']}."

        return TestModel(handler=handler)

    async def test_repeats_on_prolonged_silence(self) -> None:
        session = _make_session(
            tool_sleep=0.6,
            filler={"model": self._counting_model(), "delay_secs": 0.0, "repeat_secs": 0.08},
        )
        await session._run_turn("what is the answer?")

        fillers = [e.text for e in session.transcript if e.filler]
        assert len(fillers) >= 2
        assert len(set(fillers)) == len(fillers)  # follow-ups are distinct phrases
        assert session.metrics[-1].filler_count == len(fillers)
        # Reply still arrives after all fillers, exactly once.
        replies = [e.text for e in session.transcript if e.role == "assistant" and not e.filler]
        assert replies == ["The answer is 42."]

    async def test_single_filler_without_repeat(self) -> None:
        session = _make_session(tool_sleep=0.4, filler=_default_filler())
        await session._run_turn("what is the answer?")
        assert session.metrics[-1].filler_count == 1

    async def test_max_per_turn_cap(self) -> None:
        session = _make_session(
            tool_sleep=0.9,
            filler={"model": self._counting_model(), "delay_secs": 0.0, "repeat_secs": 0.05, "max_per_turn": 2},
        )
        await session._run_turn("what is the answer?")
        assert session.metrics[-1].filler_count == 2

    async def test_no_followup_once_reply_arrives(self) -> None:
        session = _make_session(
            tool_sleep=0.05,
            filler={"model": self._counting_model(), "delay_secs": 0.0, "repeat_secs": 0.2},
        )
        await session._run_turn("what is the answer?")
        assert session.metrics[-1].filler_count == 1

    async def test_followup_generation_sees_previous_phrases(self) -> None:
        prompts: list[str] = []

        def handler(messages):
            prompts.append(messages[-1].collect_text())
            return f"Filler {len(prompts)}."

        session = _make_session(
            tool_sleep=0.5,
            filler={"model": TestModel(handler=handler), "delay_secs": 0.0, "repeat_secs": 0.08},
        )
        await session._run_turn("what is the answer?")

        assert len(prompts) >= 2
        assert "Filler 1." in prompts[1]  # follow-up knows what was already said
        assert "still" in prompts[1]


class TestFillerInternals:
    def test_once_per_turn_guard(self) -> None:
        session = _make_session(filler=_default_filler(delay_secs=60.0))

        async def run() -> None:
            session._maybe_schedule_filler("lookup")
            task = session._turn_filler_task
            assert task is not None
            session._maybe_schedule_filler("other_tool")
            assert session._turn_filler_task is task
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())

    def test_no_task_when_disabled(self) -> None:
        session = _make_session()
        session._maybe_schedule_filler("lookup")
        assert session._turn_filler_task is None

    async def test_filler_segment_records_empty_text(self) -> None:
        """Barge-in accounting: filler bytes count, filler words don't."""
        session = _make_session()
        await session._speak("Hello there friend", filler=True)

        [(text, num_bytes)] = [(t, b) for t, b in session._turn_tts_segment_records]
        assert text == ""
        assert num_bytes > 0
        assert map_played_bytes_to_text([(text, num_bytes)], num_bytes // 2) == ""

    async def test_generator_sees_user_text_and_custom_prompt(self) -> None:
        seen: list[list] = []

        def handler(messages):
            seen.append(messages)
            return "Un momento."

        session = _make_session(
            filler={"model": TestModel(handler=handler), "system_prompt": "CUSTOM PROMPT"},
        )
        session._active_turn_user_text = "que tiempo hace?"
        phrase = await session._generate_filler("lookup")

        assert phrase == "Un momento."
        assert "que tiempo hace?" in seen[0][-1].collect_text()
        assert session._filler_agent.system_prompt == "CUSTOM PROMPT"

    def test_buffered_assistant_text_blocks_filler(self) -> None:
        session = _make_session(filler=_default_filler())
        assert session._filler_ok_to_speak() is True
        session._turn_assistant_text = "Sure,"  # streamed delta, below flush threshold
        assert session._turn_tts_scheduled_text == ""
        assert session._filler_ok_to_speak() is False

    async def test_cancel_mid_speech_commits_nothing(self) -> None:
        """Cancelled during synthesis → no count / transcript entry / event;
        the phrase stays in ``_turn_filler_text`` for echo suppression."""
        from timbal.voice import FillerSpoken, TextToSpeech

        class BlockingTTS(TextToSpeech):
            async def connect(self, config) -> None:
                pass

            async def close(self) -> None:
                pass

            async def synthesize(self, text: str):  # noqa: ARG002
                await asyncio.sleep(30)
                yield b"\x00" * 32

        agent = Agent(name="voice_test", model=TestModel(responses=["hi"]), tools=[])
        session = VoiceSession(
            agent, _make_stt_class()(), BlockingTTS(), turn_detector="heuristic", filler=_default_filler()
        )
        task = asyncio.create_task(session._speak_filler_task(PHRASE, "lookup"))
        await asyncio.sleep(0.05)  # let it get into _speak
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert session._turn_filler_count == 0
        assert not any(e.filler for e in session.transcript)
        assert not any(
            isinstance(session._event_queue.get_nowait(), FillerSpoken)
            for _ in range(session._event_queue.qsize())
        )
        assert PHRASE in session._turn_filler_text

    async def test_echo_suppression_sees_filler_text(self) -> None:
        session = _make_session()
        session._turn_assistant_text = "The answer is 42."
        session._turn_filler_text = PHRASE
        assert PHRASE in session._spoken_assistant_text()
        assert "The answer is 42." in session._spoken_assistant_text()


@pytest.mark.usefixtures("clear_voice_env")
class TestFillerEnvDefaults:
    def test_unset_means_off(self) -> None:
        assert voice_routes.default_voice_config_from_env().ambient is None
        assert voice_routes.default_voice_config_from_env().filler is None

    def test_flag_enables_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_FILLER", "1")
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.filler is not None
        assert cfg.filler.delay_secs == 1.0

    def test_detail_var_implies_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_SYSTEM_PROMPT", "keep it short")
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_MODEL", "openai/gpt-4o-mini")
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_DELAY_SECS", "0.5")
        cfg = voice_routes.default_voice_config_from_env()
        assert cfg.filler is not None
        assert cfg.filler.system_prompt == "keep it short"
        assert cfg.filler.model == "openai/gpt-4o-mini"
        assert cfg.filler.delay_secs == 0.5

    def test_explicit_off_wins_over_detail_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_FILLER", "0")
        monkeypatch.setenv("TIMBAL_VOICE_FILLER_MODEL", "openai/gpt-4o-mini")
        assert voice_routes.default_voice_config_from_env().filler is None


class TestClientFillerOverrides:
    """The playground can enable/disable/tune the filler per session."""

    def test_filler_is_client_settable(self) -> None:
        assert "filler" in voice_routes.CLIENT_SETTABLE_VOICE_FIELDS

    def test_client_enables_when_server_off(self) -> None:
        out = voice_routes.merge_client_voice_overrides(
            VoiceConfig(), {"filler": {"enabled": True, "delay_secs": 0.5}}
        )
        assert isinstance(out.filler, FillerConfig)
        assert out.filler.enabled is True
        assert out.filler.delay_secs == 0.5

    def test_client_disables_server_default(self) -> None:
        base = VoiceConfig(filler={"delay_secs": 0.5})
        out = voice_routes.merge_client_voice_overrides(base, {"filler": {"enabled": False}})
        assert out.filler.enabled is False

    def test_partial_override_keeps_server_prompt(self) -> None:
        base = VoiceConfig(filler={"system_prompt": "SERVER PROMPT"})
        out = voice_routes.merge_client_voice_overrides(base, {"filler": {"delay_secs": 0.3}})
        assert out.filler.system_prompt == "SERVER PROMPT"
        assert out.filler.delay_secs == 0.3

    def test_invalid_client_filler_keeps_server_config(self) -> None:
        base = VoiceConfig(filler={"delay_secs": 0.5})
        out = voice_routes.merge_client_voice_overrides(base, {"filler": {"phrases": ["nope"]}})
        assert out.filler.delay_secs == 0.5
        assert out.filler.enabled is True

    async def test_disabled_config_never_schedules(self) -> None:
        session = _make_session(filler=_default_filler(enabled=False))
        await session._run_turn("what is the answer?")
        assert session._turn_filler_task is None
        assert not any(e.filler for e in session.transcript)


class TestFillerOverWs:
    """The wire format a real client sees."""

    def _run(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, module_body: str) -> list[dict]:
        mod = tmp_path / "voice_agent.py"
        mod.write_text(module_body)
        monkeypatch.setenv("TIMBAL_RUNNABLE", f"{mod.resolve()}::agent")
        for k in VOICE_ENV_KEYS:
            monkeypatch.delenv(k, raising=False)
        from timbal.voice import TranscriptEvent

        stt_cls = _make_stt_class([TranscriptEvent(type="committed", text="what is the answer?")])
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsRealtimeSTT", stt_cls)
        monkeypatch.setattr("timbal.voice.elevenlabs.ElevenLabsStreamTTS", _make_tts_class())
        app = create_app()
        with TestClient(app) as client:
            with client.websocket_connect("/voice/ws") as ws:
                ws.send_json({"turn_detector": "heuristic"})
                return _collect_ws_messages(ws)

    def test_filler_payload_and_transcript_flag(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module_body = (
            "import asyncio\n"
            "from timbal import Agent\n"
            "from timbal.core.test_model import TestModel\n"
            "from timbal.types.message import Message\n"
            "from timbal.types.content import ToolUseContent\n"
            "async def lookup(q: str) -> str:\n"
            "    await asyncio.sleep(0.2)\n"
            "    return '42'\n"
            "agent = Agent(name='voice_test', model=TestModel(responses=[\n"
            "    Message(role='assistant', content=[ToolUseContent(id='c1', name='lookup', input={'q': 'x'})],\n"
            "            stop_reason='tool_use'),\n"
            "    'The answer is 42.',\n"
            "]), tools=[lookup])\n"
            f"agent.voice_config = {{'filler': {{'model': TestModel(responses=[{PHRASE!r}]), 'delay_secs': 0}}}}\n"
        )
        messages = self._run(monkeypatch, tmp_path, module_body=module_body)

        types = [m["type"] for m in messages]
        assert "filler" in types
        filler_msg = next(m for m in messages if m["type"] == "filler")
        assert filler_msg["text"] == PHRASE
        # The filler is spoken before the reply text arrives.
        assert types.index("filler") < types.index("agent_text_done")
        done = next(m for m in messages if m["type"] == "agent_text_done")
        assert done["text"] == "The answer is 42."
        transcript = next(m for m in messages if m["type"] == "session_transcript")["entries"]
        assert [e["text"] for e in transcript if e.get("filler")] == [PHRASE]

    def test_no_filler_when_not_configured(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        module_body = (
            "import asyncio\n"
            "from timbal import Agent\n"
            "from timbal.core.test_model import TestModel\n"
            "from timbal.types.message import Message\n"
            "from timbal.types.content import ToolUseContent\n"
            "async def lookup(q: str) -> str:\n"
            "    await asyncio.sleep(0.05)\n"
            "    return '42'\n"
            "agent = Agent(name='voice_test', model=TestModel(responses=[\n"
            "    Message(role='assistant', content=[ToolUseContent(id='c1', name='lookup', input={'q': 'x'})],\n"
            "            stop_reason='tool_use'),\n"
            "    'The answer is 42.',\n"
            "]), tools=[lookup])\n"
        )
        messages = self._run(monkeypatch, tmp_path, module_body=module_body)
        assert "filler" not in [m["type"] for m in messages]
