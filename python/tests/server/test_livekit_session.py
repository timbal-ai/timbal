"""LiveKit session driver: env gating and reliable-data chunking.

Does not talk to a real SFU — the FFI extra is not required.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace

from timbal.server.livekit_session import (
    _is_caller,
    chunk_data_payloads,
    is_config_hello,
    maybe_start_livekit_session,
    merge_client_config,
)


class TestMaybeStart:
    def test_off_when_transport_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("TIMBAL_VOICE_TRANSPORT", raising=False)
        assert maybe_start_livekit_session(SimpleNamespace()) is None

    def test_off_when_transport_is_webrtc(self, monkeypatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_TRANSPORT", "webrtc")
        assert maybe_start_livekit_session(SimpleNamespace()) is None

    async def test_starts_a_task_when_livekit(self, monkeypatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_TRANSPORT", "livekit")
        task = maybe_start_livekit_session(SimpleNamespace(state=SimpleNamespace(runnable=None)))
        assert task is not None
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


class TestCallerIdentity:
    def test_empty_prefix_matches_anyone(self) -> None:
        assert _is_caller("laptop", "") is True

    def test_prefix_match(self) -> None:
        assert _is_caller("user-abc", "user") is True
        assert _is_caller("agent", "user") is False


class TestChunkDataPayloads:
    def test_small_payloads_pass_through(self) -> None:
        payloads = [{"type": "interrupted", "heard_text": "hi"}]
        assert chunk_data_payloads(payloads) == payloads

    def test_oversized_transcript_is_split_with_seq_total(self) -> None:
        # Each entry is ~200 bytes; 100 of them blow past 12 KiB.
        entries = [
            {"role": "assistant", "text": "x" * 180, "timestamp": 1.0 + i}
            for i in range(100)
        ]
        payloads = [{"type": "session_transcript", "entries": entries, "started_at": 1.0}]
        chunks = chunk_data_payloads(payloads)
        assert len(chunks) > 1
        assert all(c["type"] == "session_transcript" for c in chunks)
        assert [c["seq"] for c in chunks] == list(range(len(chunks)))
        assert all(c["total"] == len(chunks) for c in chunks)
        assert all(c["started_at"] == 1.0 for c in chunks)
        rebuilt = [e for c in chunks for e in c["entries"]]
        assert rebuilt == entries

    def test_non_transcript_oversized_is_left_intact(self) -> None:
        payload = {"type": "agent_text_done", "text": "y" * 20_000}
        assert chunk_data_payloads([payload]) == [payload]


class TestMergeClientConfig:
    def test_hello_overlays_env(self) -> None:
        merged = merge_client_config(
            '{"stt_provider": "elevenlabs", "model": "env/model"}',
            {"stt_provider": "deepgram-flux", "turn_detector": "provider"},
        )
        assert merged["stt_provider"] == "deepgram-flux"
        assert merged["turn_detector"] == "provider"
        assert merged["model"] == "env/model"

    def test_bad_env_json_is_empty_base(self) -> None:
        assert merge_client_config("not-json", {"tts_provider": "elevenlabs"}) == {
            "tts_provider": "elevenlabs"
        }

    def test_no_hello_keeps_env(self) -> None:
        assert merge_client_config('{"voice": "abc"}', None) == {"voice": "abc"}


class TestConfigHello:
    def test_untyped_object_is_hello(self) -> None:
        assert is_config_hello({"sample_rate": 16000, "stt_provider": "elevenlabs"})

    def test_typed_frame_is_not_hello(self) -> None:
        assert not is_config_hello({"type": "playback", "played_ms": 12})

    def test_null_type_is_hello(self) -> None:
        assert is_config_hello({"type": None, "sample_rate": 16000})
