"""``pipeline="live"``: the voice server builds a GPT-Live ``LiveSession`` instead of the cascaded one."""

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.server.voice import CLIENT_SETTABLE_VOICE_FIELDS, build_voice_session, event_to_payloads
from timbal.voice import DelegationCreated, DelegationResult, SessionStarted, VoiceConfig, VoiceSession
from timbal.voice.openai_live import LiveSession, OpenAILiveClient


def _agent() -> Agent:
    return Agent(name="backend", model=TestModel(responses=["ok"]), tools=[], system_prompt="Backend rules.")


class TestBuild:
    def test_live_pipeline_builds_live_session(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        defaults = VoiceConfig(pipeline="live", greeting="Welcome to Acme.")
        session, meta = build_voice_session(_agent(), defaults, {"sample_rate": 16000}, parent_run_id="run_1")
        assert isinstance(session, LiveSession)
        assert isinstance(session.transport, OpenAILiveClient)
        cfg = session.transport.session_config()
        assert cfg["model"] == "gpt-live-1"
        assert cfg["audio"] == {"format": {"type": "audio/pcm", "rate": 16000}, "output": {"voice": "marin"}}
        assert cfg["delegation"] == {"type": "client"}
        assert "delegate to the backend" in cfg["instructions"]
        assert session.drop_silence is True
        assert session.parent_run_id == "run_1"
        assert session.greeting and "Welcome to Acme." in session.greeting
        assert meta["pipeline"] == "live" and meta["live_model"] == "gpt-live-1" and meta["live_voice"] == "marin"
        assert meta["stt_provider"] is None and meta["tts_provider"] is None
        assert meta["parent_run_id"] == "run_1"

    def test_client_hello_selects_live_and_voice(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert {"pipeline", "live_voice"} <= CLIENT_SETTABLE_VOICE_FIELDS
        session, meta = build_voice_session(_agent(), VoiceConfig(), {"pipeline": "live", "live_voice": "vesper"})
        assert isinstance(session, LiveSession)
        assert session.transport.session_config()["audio"]["output"] == {"voice": "vesper"}
        assert meta["live_voice"] == "vesper"

    def test_default_pipeline_is_cascaded(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
        session, meta = build_voice_session(_agent(), VoiceConfig(), {})
        assert isinstance(session, VoiceSession)
        assert "pipeline" not in meta

    def test_unsupported_rate_falls_back_to_cascaded(self, monkeypatch):
        monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
        session, _ = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {"sample_rate": 44100})
        assert isinstance(session, VoiceSession)

    def test_live_instructions_and_model_override(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        defaults = VoiceConfig(pipeline="live", live_instructions="Speak like a pirate.", live_model="gpt-live-1")
        session, meta = build_voice_session(_agent(), defaults, {"model": "openai/gpt-5.4-nano"})
        assert session.transport.session_config()["instructions"] == "Speak like a pirate."
        assert session.model == "openai/gpt-5.4-nano" and meta["model"] == "openai/gpt-5.4-nano"

    def test_default_instructions_list_the_agents_tools(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")

        def book_appointment(day: str) -> str:
            """Book a dental appointment on the given day."""
            return day

        agent = Agent(name="backend", model=TestModel(responses=["ok"]), tools=[book_appointment])
        session, _ = build_voice_session(agent, VoiceConfig(pipeline="live"), {})
        text = session.transport.session_config()["instructions"]
        assert "Backchannel policy:" in text and "Interruption policy:" in text
        assert "Backend tools:\n- book appointment: Book a dental appointment on the given day." in text
        # No tools → a generic capability line, never an empty list.
        session, _ = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {})
        assert (
            "- Look up information and take actions on the caller's behalf."
            in (session.transport.session_config()["instructions"])
        )

    def test_language_becomes_prompt_rule_and_opener_language(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        assert "live_instructions" in CLIENT_SETTABLE_VOICE_FIELDS
        defaults = VoiceConfig(pipeline="live", greeting="Welcome.")
        session, meta = build_voice_session(
            _agent(), defaults, {"language": "es-ES", "live_instructions": "Eres Sam, de Northwind Dental."}
        )
        text = session.transport.session_config()["instructions"]
        assert text.startswith("Eres Sam, de Northwind Dental.")
        assert "Speak Spanish unless the user asks to switch." in text
        assert session.greeting.startswith("Speak Spanish. ")
        assert meta["language"] == "Spanish"
        # Unknown code is passed through verbatim; auto means no rule.
        session, _ = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {"language": "xx"})
        assert "Speak xx unless" in session.transport.session_config()["instructions"]
        session, meta = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {"language": "auto"})
        assert "unless the user asks to switch" not in session.transport.session_config()["instructions"]
        assert meta["language"] is None

    def test_no_greeting_config_means_no_opener(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        session, _ = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {})
        assert session.greeting is None
        defaults = VoiceConfig(pipeline="live", greeting={"instructions": "Greet in Catalan, then listen."})
        session, _ = build_voice_session(_agent(), defaults, {})
        assert session.greeting == "Greet in Catalan, then listen."

    def test_live_session_matches_livekit_host_contract(self, monkeypatch):
        # livekit_session.py reads session.closed / recording_meta and may
        # await prepare() before publishing. Missing any of these used to
        # ClientInitiated-disconnect the room ~200ms after pipeline=live.
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        session, _ = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {})
        assert session.closed is False
        assert session.recording_meta is None
        session.recording_meta = {"transport": "livekit"}
        assert session.recording_meta["transport"] == "livekit"
        assert callable(session.prepare)

    def test_paced_transport_pins_24k_and_keeps_silence(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        session, meta = build_voice_session(
            _agent(), VoiceConfig(pipeline="live"), {"sample_rate": 16000}, playback_tracker=object()
        )
        assert session.transport.session_config()["audio"]["format"]["rate"] == 24000
        assert session.drop_silence is False
        assert session.audio_input.sample_rate == 24000


class TestPayloads:
    def _session(self, monkeypatch) -> tuple[LiveSession, dict]:
        monkeypatch.setenv("OPENAI_API_KEY", "k")
        session, meta = build_voice_session(_agent(), VoiceConfig(pipeline="live"), {})
        return session, {"transport": "websocket", "playback_acks": "ignored", **meta}

    def test_session_started_has_no_endpointer_and_live_id(self, monkeypatch):
        session, meta = self._session(monkeypatch)
        session.live_session_id = "live_abc"
        (payload,) = event_to_payloads(SessionStarted(), session, meta)
        assert payload["type"] == "session_started"
        assert payload["vad_endpointing"] is False
        assert payload["pipeline"] == "live" and payload["live_session_id"] == "live_abc"
        # Ours: the recording / manifest / platform-session key, known before connect.
        assert payload["session_id"] == session.session_id and len(session.session_id) == 32

    def test_delegation_events_are_forwarded(self, monkeypatch):
        session, meta = self._session(monkeypatch)
        (created,) = event_to_payloads(DelegationCreated(delegation_id="d1", prompt="user: hi"), session, meta)
        assert created == {"type": "delegation_created", "delegation_id": "d1", "prompt": "user: hi"}
        (result,) = event_to_payloads(
            DelegationResult(delegation_id="d1", text="24C", run_id="r", stale=True, spoken=False), session, meta
        )
        assert result["type"] == "delegation_result"
        assert result["stale"] is True and result["spoken"] is False and result["run_id"] == "r"


@pytest.mark.parametrize("field", ["pipeline", "live_model", "live_voice", "live_instructions"])
def test_voice_config_has_live_fields(field):
    assert field in VoiceConfig.model_fields
