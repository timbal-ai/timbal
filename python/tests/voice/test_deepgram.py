"""Deepgram STT providers: message → TranscriptEvent mapping, URI building, resolve_stt.

No live API calls — messages are fed straight into ``_handle_message`` and the
event queue is drained, mirroring how the receive loop feeds ``events()``.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import pytest
from timbal.voice import AudioInputConfig
from timbal.voice.deepgram import (
    _AUDIO_FLUSH_BYTES,
    DEFAULT_FLUX_MODEL,
    DEFAULT_NOVA_MODEL,
    DeepgramFluxSTT,
    DeepgramNovaSTT,
    effective_stt_model,
    is_flux_model,
    resolve_stt,
    stt_provider_id,
)
from timbal.voice.elevenlabs import ElevenLabsRealtimeSTT
from timbal.voice.munsit import MunsitStreamSTT


def _drain(stt) -> list:
    events = []
    while not stt._queue.empty():
        events.append(stt._queue.get_nowait())
    return events


def _flux_turn_info(event: str, transcript: str = "hello there", **kw) -> dict:
    return {
        "type": "TurnInfo",
        "event": event,
        "transcript": transcript,
        "turn_index": 0,
        "end_of_turn_confidence": 0.9,
        **kw,
    }


class _FakeWs:
    def __init__(self) -> None:
        self.sent: list = []

    async def send(self, payload) -> None:
        self.sent.append(payload)


class _SlowFakeWs:
    """WS that yields during send so concurrent callers can race without a lock."""

    def __init__(self) -> None:
        self.sent: list = []
        self.in_flight = 0
        self.max_in_flight = 0

    async def send(self, payload) -> None:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await asyncio.sleep(0.02)
        self.sent.append(payload)
        self.in_flight -= 1


class TestWireSendSerialization:
    async def test_threshold_push_and_flush_serialize_sends(self) -> None:
        """push_audio and _flush_audio must not await ``_ws.send`` concurrently.

        Regression: buffer lock used to release before send, so the flush loop
        and a threshold push could interleave PCM frames on one socket.
        """
        stt = DeepgramNovaSTT()
        ws = _SlowFakeWs()
        stt._ws = ws
        frame = b"\xab" * _AUDIO_FLUSH_BYTES

        async def _pushes() -> None:
            await stt.push_audio(frame)
            await stt.push_audio(frame)

        async def _flushes() -> None:
            for _ in range(6):
                await stt._flush_audio()
                await asyncio.sleep(0.005)

        await asyncio.gather(_pushes(), _flushes())
        assert ws.max_in_flight == 1
        assert ws.sent == [frame, frame]

    async def test_flush_drains_remainder_after_threshold_push(self) -> None:
        stt = DeepgramFluxSTT()
        ws = _FakeWs()
        stt._ws = ws
        frame = b"\x11" * _AUDIO_FLUSH_BYTES
        remainder = b"\x22" * 100
        await stt.push_audio(frame + remainder)
        assert ws.sent == [frame + remainder]  # one threshold send of all buffered
        # Small chunk under threshold sits until flush.
        await stt.push_audio(b"\x33" * 50)
        assert len(ws.sent) == 1
        await stt._flush_audio()
        assert ws.sent[-1] == b"\x33" * 50


class TestFluxEventMapping:
    async def test_update_and_start_of_turn_are_partials(self) -> None:
        stt = DeepgramFluxSTT()
        await stt._handle_message(_flux_turn_info("StartOfTurn", "hi"))
        await stt._handle_message(_flux_turn_info("Update", "hi there"))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("partial", "hi"), ("partial", "hi there")]

    async def test_end_of_turn_is_committed(self) -> None:
        stt = DeepgramFluxSTT()
        await stt._handle_message(_flux_turn_info("EndOfTurn", "tell me a story."))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "tell me a story.")]

    async def test_eager_events_emit_nothing(self) -> None:
        stt = DeepgramFluxSTT()
        await stt._handle_message(_flux_turn_info("EagerEndOfTurn"))
        await stt._handle_message(_flux_turn_info("TurnResumed"))
        assert _drain(stt) == []

    async def test_empty_transcripts_skipped(self) -> None:
        stt = DeepgramFluxSTT()
        await stt._handle_message(_flux_turn_info("Update", ""))
        await stt._handle_message(_flux_turn_info("EndOfTurn", "  "))
        assert _drain(stt) == []

    async def test_connected_ignored_fatal_error_surfaces(self) -> None:
        stt = DeepgramFluxSTT()
        await stt._handle_message({"type": "Connected", "request_id": "r1", "sequence_id": 0})
        await stt._handle_message({"type": "FatalError", "description": "bad auth", "code": "AUTH"})
        events = _drain(stt)
        assert len(events) == 1
        assert events[0].type == "error"
        assert "bad auth" in events[0].text

    async def test_commit_is_noop(self) -> None:
        stt = DeepgramFluxSTT()
        stt._ws = _FakeWs()
        await stt.commit()
        assert stt._ws.sent == []


class TestFluxUri:
    def _parse(self, stt: DeepgramFluxSTT, config: AudioInputConfig) -> dict:
        uri = stt._build_uri(config)
        parsed = urlparse(uri)
        assert parsed.scheme == "wss"
        assert parsed.path == "/v2/listen"
        return parse_qs(parsed.query)

    def test_defaults(self) -> None:
        q = self._parse(DeepgramFluxSTT(), AudioInputConfig())
        assert q["model"] == [DEFAULT_FLUX_MODEL]
        assert q["encoding"] == ["linear16"]
        assert q["sample_rate"] == ["16000"]
        assert q["eot_timeout_ms"] == ["2000"]
        assert "language" not in q

    def test_language_hint_only_for_multi(self) -> None:
        q = self._parse(DeepgramFluxSTT(), AudioInputConfig(language="es"))
        assert q["language_hint"] == ["es"]
        q = self._parse(DeepgramFluxSTT(), AudioInputConfig(model="flux-general-en", language="es"))
        assert "language_hint" not in q

    def test_foreign_model_replaced(self) -> None:
        # Only TIMBAL_STT_PROVIDER switched; env stt_model still Scribe's.
        q = self._parse(DeepgramFluxSTT(), AudioInputConfig(model="scribe_v2_realtime"))
        assert q["model"] == [DEFAULT_FLUX_MODEL]

    def test_extra_passthrough_filtered(self) -> None:
        config = AudioInputConfig(
            extra={
                "eot_threshold": 0.8,
                "eot_timeout_ms": 3000,
                "keyterm": ["Timbal", "Scribe"],
                # Scribe-only knob must not leak into the Flux query.
                "commit_strategy": "vad",
            }
        )
        q = self._parse(DeepgramFluxSTT(), config)
        assert q["eot_threshold"] == ["0.8"]
        assert q["eot_timeout_ms"] == ["3000"]
        assert q["keyterm"] == ["Timbal", "Scribe"]
        assert "commit_strategy" not in q


class TestNovaEventMapping:
    def _results(self, text: str, *, is_final: bool, speech_final: bool = False, **kw) -> dict:
        return {
            "type": "Results",
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": text}]},
            **kw,
        }

    async def test_interim_is_partial(self) -> None:
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("hello", is_final=False))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("partial", "hello")]

    async def test_is_final_buffers_until_speech_final(self) -> None:
        """Deepgram's documented recipe: concat is_final segments, flush on speech_final."""
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("yeah so my credit card number is two two", is_final=True))
        assert _drain(stt) == []
        await stt._handle_message(self._results("two two three three three three", is_final=True, speech_final=True))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [
            ("committed", "yeah so my credit card number is two two two two three three three three")
        ]
        assert stt._segments == []

    async def test_partial_includes_buffered_segments(self) -> None:
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("hello there", is_final=True))
        await stt._handle_message(self._results("how are", is_final=False))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("partial", "hello there how are")]

    async def test_from_finalize_flushes(self) -> None:
        """commit() → Finalize → is_final response with from_finalize must commit."""
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("stop the music", is_final=True, from_finalize=True))
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "stop the music")]

    async def test_utterance_end_flushes_buffer(self) -> None:
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("hello there", is_final=True))
        await stt._handle_message({"type": "UtteranceEnd", "last_word_end": 1.2})
        events = _drain(stt)
        assert [(e.type, e.text) for e in events] == [("committed", "hello there")]

    async def test_empty_speech_final_no_commit(self) -> None:
        stt = DeepgramNovaSTT()
        await stt._handle_message(self._results("", is_final=True, speech_final=True))
        assert _drain(stt) == []

    async def test_error_surfaces(self) -> None:
        stt = DeepgramNovaSTT()
        await stt._handle_message({"type": "Error", "description": "NET-0001"})
        events = _drain(stt)
        assert events[0].type == "error"
        assert "NET-0001" in events[0].text

    async def test_commit_sends_finalize(self) -> None:
        stt = DeepgramNovaSTT()
        ws = _FakeWs()
        stt._ws = ws
        await stt.commit()
        assert [json.loads(m) for m in ws.sent if isinstance(m, str)] == [{"type": "Finalize"}]


class TestNovaUri:
    def _parse(self, config: AudioInputConfig) -> dict:
        uri = DeepgramNovaSTT()._build_uri(config)
        parsed = urlparse(uri)
        assert parsed.path == "/v1/listen"
        return parse_qs(parsed.query)

    def test_defaults(self) -> None:
        q = self._parse(AudioInputConfig())
        assert q["model"] == [DEFAULT_NOVA_MODEL]
        assert q["encoding"] == ["linear16"]
        assert q["interim_results"] == ["true"]
        assert q["smart_format"] == ["true"]
        assert q["punctuate"] == ["true"]
        assert q["endpointing"] == ["300"]
        assert q["channels"] == ["1"]

    def test_language_and_extra_override(self) -> None:
        q = self._parse(AudioInputConfig(language="es", extra={"endpointing": 500, "utterance_end_ms": 1000}))
        assert q["language"] == ["es"]
        assert q["endpointing"] == ["500"]
        assert q["utterance_end_ms"] == ["1000"]

    def test_foreign_model_replaced(self) -> None:
        q = self._parse(AudioInputConfig(model="scribe_v2_realtime"))
        assert q["model"] == [DEFAULT_NOVA_MODEL]
        q = self._parse(AudioInputConfig(model="flux-general-en"))
        assert q["model"] == [DEFAULT_NOVA_MODEL]

    def test_nova_variant_kept(self) -> None:
        q = self._parse(AudioInputConfig(model="nova-3-medical"))
        assert q["model"] == ["nova-3-medical"]


class TestResolveStt:
    def test_explicit_providers(self) -> None:
        assert isinstance(resolve_stt("elevenlabs"), ElevenLabsRealtimeSTT)
        assert isinstance(resolve_stt("deepgram"), DeepgramFluxSTT)
        assert isinstance(resolve_stt("deepgram", model="nova-3"), DeepgramNovaSTT)
        assert isinstance(resolve_stt("deepgram", model="flux-general-en"), DeepgramFluxSTT)
        # Leftover Scribe model + bare deepgram must NOT silently become Nova.
        assert isinstance(resolve_stt("deepgram", model="scribe_v2_realtime"), DeepgramFluxSTT)
        assert isinstance(resolve_stt("munsit"), MunsitStreamSTT)
        assert isinstance(resolve_stt("faseeh"), MunsitStreamSTT)
        # Label wins over a stale model id from env defaults.
        assert isinstance(resolve_stt("munsit", model="scribe_v2_realtime"), MunsitStreamSTT)

    def test_ui_labels(self) -> None:
        assert isinstance(resolve_stt("deepgram-flux"), DeepgramFluxSTT)
        assert isinstance(resolve_stt("deepgram-nova"), DeepgramNovaSTT)
        # Label wins over a stale model id from env defaults.
        assert isinstance(resolve_stt("deepgram-flux", model="scribe_v2_realtime"), DeepgramFluxSTT)
        assert isinstance(resolve_stt("deepgram-nova", model="flux-general-multi"), DeepgramNovaSTT)

    def test_inference_from_model(self) -> None:
        assert isinstance(resolve_stt(None, model="flux-general-multi"), DeepgramFluxSTT)
        assert isinstance(resolve_stt(None, model="nova-3"), DeepgramNovaSTT)
        assert isinstance(resolve_stt(None, model="munsit-en-ar"), MunsitStreamSTT)
        assert isinstance(resolve_stt(None, model="munsit"), MunsitStreamSTT)
        assert isinstance(resolve_stt(None, model="scribe_v2_realtime"), ElevenLabsRealtimeSTT)
        assert isinstance(resolve_stt(None, model=None), ElevenLabsRealtimeSTT)

    def test_effective_stt_model(self) -> None:
        assert effective_stt_model(DeepgramFluxSTT(), "scribe_v2_realtime") == DEFAULT_FLUX_MODEL
        assert effective_stt_model(DeepgramFluxSTT(), "flux-general-en") == "flux-general-en"
        assert effective_stt_model(DeepgramNovaSTT(), "scribe_v2_realtime") == DEFAULT_NOVA_MODEL
        assert effective_stt_model(DeepgramNovaSTT(), "nova-3-general") == "nova-3-general"
        assert effective_stt_model(DeepgramNovaSTT(), "munsit-en-ar") == DEFAULT_NOVA_MODEL
        assert effective_stt_model(MunsitStreamSTT(), "scribe_v2_realtime") == "munsit-en-ar"
        assert effective_stt_model(MunsitStreamSTT(), "flux-general-multi") == "munsit-en-ar"
        assert effective_stt_model(MunsitStreamSTT(), "munsit") == "munsit"
        # Unknown-provider fallback → ElevenLabs must not keep a foreign id.
        assert effective_stt_model(ElevenLabsRealtimeSTT(), "flux-general-multi") is None
        assert effective_stt_model(ElevenLabsRealtimeSTT(), "nova-3") is None
        assert effective_stt_model(ElevenLabsRealtimeSTT(), "munsit-en-ar") is None
        assert effective_stt_model(ElevenLabsRealtimeSTT(), "scribe_v2_realtime") == "scribe_v2_realtime"
        assert effective_stt_model(ElevenLabsRealtimeSTT(), None) is None

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown STT provider"):
            resolve_stt("whisper-cpp")

    def test_stt_provider_id(self) -> None:
        assert stt_provider_id(DeepgramFluxSTT()) == "deepgram-flux"
        assert stt_provider_id(DeepgramNovaSTT()) == "deepgram-nova"
        assert stt_provider_id(MunsitStreamSTT()) == "munsit"
        assert stt_provider_id(ElevenLabsRealtimeSTT()) == "elevenlabs"

    def test_is_flux_model(self) -> None:
        assert is_flux_model("flux-general-multi")
        assert is_flux_model("FLUX-GENERAL-EN")
        assert not is_flux_model("nova-3")
        assert not is_flux_model(None)
        assert not is_flux_model("")
