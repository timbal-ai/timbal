"""CallRecorder: timeline placement, mixing, truncation, and decode round-trips.

Every assertion decodes the actual MP3 back and checks *where* energy sits on
the timeline — that's the recorder's whole job. Tolerances are generous
(100ms guard bands) because lame adds encoder delay/padding.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")
np = pytest.importorskip("numpy", reason="timbal[voice] extra (numpy) not installed")

import av  # noqa: E402
from timbal.voice.recording import CallRecorder, build_manifest  # noqa: E402

SR = 16_000


def _tone(secs: float, freq: float = 440.0, amp: int = 12000) -> bytes:
    n = int(secs * SR)
    buf = bytearray()
    for i in range(n):
        v = int(amp * math.sin(2 * math.pi * freq * i / SR))
        buf += v.to_bytes(2, "little", signed=True)
    return bytes(buf)


def _silence(secs: float) -> bytes:
    return b"\x00" * (int(secs * SR) * 2)


def _decode(path: Path) -> np.ndarray:
    """Decode to float32, shape (channels, samples)."""
    chunks = []
    with av.open(str(path)) as container:
        for frame in container.decode(audio=0):
            arr = frame.to_ndarray()
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            if not frame.format.is_planar and frame.layout.nb_channels > 1:
                arr = arr.reshape(-1, frame.layout.nb_channels).T
            chunks.append(arr.astype(np.float32))
    return np.concatenate(chunks, axis=1)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def _span(audio: np.ndarray, ch: int, t0: float, t1: float) -> np.ndarray:
    return audio[ch, int(t0 * SR) : int(t1 * SR)]


class TestTimeline:
    def test_agent_audio_lands_where_the_mic_clock_said(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        # 1s of quiet call, then the agent speaks 0.5s while the caller is silent.
        rec.add_mic(_silence(1.0))
        rec.add_agent(_tone(0.5))
        rec.add_mic(_silence(1.0))
        assert rec.close(manifest=None) is not None

        audio = _decode(tmp_path / "call.mp3")
        assert audio.shape[0] == 1  # mixed => mono
        assert abs(audio.shape[1] / SR - 2.0) < 0.15
        assert _rms(_span(audio, 0, 0.1, 0.9)) < 0.01  # before: silence
        assert _rms(_span(audio, 0, 1.1, 1.4)) > 0.1  # agent speaking
        assert _rms(_span(audio, 0, 1.6, 1.9)) < 0.01  # after: silence again

    def test_mixed_layout_sums_both_voices(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_agent(_tone(1.0, freq=300))
        rec.add_mic(_tone(1.0, freq=800))  # both talk over the same second
        rec.close()

        audio = _decode(tmp_path / "call.mp3")
        assert audio.shape[0] == 1
        assert _rms(audio[0]) > 0.3  # both voices present, no channel lost

    def test_split_layout_keeps_sides_separate(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR, layout="split")
        rec.add_agent(_tone(1.0))
        rec.add_mic(_silence(1.0))  # caller silent while the agent talks
        rec.close()

        audio = _decode(tmp_path / "call.mp3")
        assert audio.shape[0] == 2
        assert _rms(audio[0]) < 0.02  # left = caller: silent
        assert _rms(audio[1]) > 0.1  # right = agent: tone

    def test_close_drains_agent_audio_past_the_mic_timeline(self, tmp_path: Path) -> None:
        """Call ends while TTS is still playing: the tail belongs in the file."""
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_mic(_silence(0.2))
        rec.add_agent(_tone(1.0))
        rec.close()

        audio = _decode(tmp_path / "call.mp3")
        assert abs(audio.shape[1] / SR - 1.2) < 0.15


class TestBargeInTruncation:
    def test_dropped_tail_never_reaches_the_file(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_agent(_tone(2.0))  # a long reply, synthesized instantly
        rec.add_mic(_silence(0.5))  # 0.5s heard...
        emitted = int(2.0 * SR) * 2
        heard = int(0.5 * SR) * 2
        dropped = rec.drop_agent_tail(emitted - heard)  # ...then barge-in
        assert dropped == emitted - heard
        rec.add_mic(_silence(1.0))  # call continues
        rec.close()

        audio = _decode(tmp_path / "call.mp3")
        assert abs(audio.shape[1] / SR - 1.5) < 0.15
        assert _rms(_span(audio, 0, 0.1, 0.4)) > 0.1  # heard prefix present
        assert _rms(_span(audio, 0, 0.7, 1.4)) < 0.01  # unheard tail gone

    def test_drop_clamps_to_the_queue(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_agent(_tone(0.1))
        assert rec.drop_agent_tail(10**9) == int(0.1 * SR) * 2
        assert rec.agent_pending_bytes == 0
        rec.close()


class TestRobustness:
    def test_odd_sized_chunks_stay_sample_aligned(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        tone = _tone(0.5)
        rec.add_mic(tone[:333])
        rec.add_mic(tone[333:])
        rec.close()
        audio = _decode(tmp_path / "call.mp3")
        assert abs(audio.shape[1] / SR - 0.5) < 0.1

    def test_close_is_idempotent_and_feeds_after_close_are_noops(self, tmp_path: Path) -> None:
        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_mic(_silence(0.2))
        first = rec.close()
        rec.add_mic(_silence(1.0))
        rec.add_agent(_tone(1.0))
        assert rec.close() is first

    def test_empty_close_still_leaves_a_playable_file(self, tmp_path: Path) -> None:
        """Zero-packet MP3 close deletes the file on Windows; pad silence."""
        path = tmp_path / "empty.mp3"
        rec = CallRecorder(path, sample_rate=SR)
        result = rec.close()
        assert result is not None
        assert path.exists()
        assert path.stat().st_size > 0
        assert result.duration_secs > 0

    def test_rejects_bad_layout(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="layout"):
            CallRecorder(tmp_path / "call.mp3", layout="both")  # type: ignore[arg-type]


class TestSessionIntegration:
    async def test_session_writes_recording_manifest_and_fires_on_saved(self, tmp_path: Path) -> None:
        import asyncio
        from contextlib import aclosing

        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.voice.session import (
            AgentTextDone,
            SessionStarted,
            TranscriptEvent,
            VoiceSession,
            VoiceSessionEvent,
        )

        from .test_session import DelayedMockSTT, MockTTS

        saved: list = []

        async def _on_saved(result) -> None:
            saved.append(result)

        recorder = CallRecorder(tmp_path / "call.mp3", sample_rate=SR, on_saved=_on_saved)
        stt = DelayedMockSTT()
        session = VoiceSession(
            agent=Agent(name="rec", model=TestModel(responses=["Hi there!"]), tools=[]),
            stt=stt,
            tts=MockTTS(chunk=b"\x01\x02" * 800, num_chunks=2),
            recorder=recorder,
            session_id="fixed-session-id",
        )
        session.recording_meta = {"transport": "test", "model": "test/model"}

        events: list[VoiceSessionEvent] = []

        async def _mic() -> object:
            # A short burst of real mic PCM so the caller side has timeline.
            for _ in range(5):
                yield b"\x00" * 640
                await asyncio.sleep(0.01)
            while not any(isinstance(e, AgentTextDone) for e in events):
                await asyncio.sleep(0.01)
            await stt.finish()

        async def _drive() -> None:
            while not any(isinstance(e, SessionStarted) for e in events):
                await asyncio.sleep(0.01)
            await stt.inject(TranscriptEvent(type="committed", text="Hello"))

        async def _run() -> None:
            async with aclosing(session.run(_mic())) as stream:
                driver = asyncio.create_task(_drive())
                async for ev in stream:
                    events.append(ev)
                await driver

        await asyncio.wait_for(_run(), timeout=10)

        audio_path = tmp_path / "call.mp3"
        manifest_path = tmp_path / "call.json"
        assert audio_path.exists() and manifest_path.exists()
        assert saved and saved[0].audio_path == audio_path

        manifest = json.loads(manifest_path.read_text())
        assert manifest["session_id"] == "fixed-session-id"
        assert manifest["meta"] == {"transport": "test", "model": "test/model"}
        roles = [e["role"] for e in manifest["transcript"]]
        assert roles == ["user", "assistant"]
        assert all(e["offset_ms"] >= 0 for e in manifest["transcript"])
        assert manifest["audio"]["duration_secs"] > 0
        assert manifest["turns"]  # per-turn latency metrics for the UI chips

        decoded = _decode(audio_path)
        assert decoded.shape[1] > 0


class TestSessionBargeIn:
    async def test_interrupt_drops_the_unheard_tail_from_the_recording(self, tmp_path: Path) -> None:
        import asyncio
        from contextlib import aclosing

        from timbal import Agent
        from timbal.core.test_model import TestModel
        from timbal.voice.session import SessionStarted, TranscriptEvent, VoiceSession

        from .test_session import DelayedMockSTT, MockTTS

        recorder = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        stt = DelayedMockSTT()
        # 4s of agent audio emitted near-instantly (TTS is faster than real time).
        emitted_secs = 4.0
        session = VoiceSession(
            agent=Agent(name="rec", model=TestModel(responses=["A very long reply"]), tools=[]),
            stt=stt,
            tts=MockTTS(chunk=b"\x01\x02" * (SR // 2), num_chunks=int(emitted_secs)),
            recorder=recorder,
        )

        started = asyncio.Event()

        async def _mic() -> object:
            await started.wait()
            yield b"\x00" * 640

        async def _drive() -> None:
            while recorder.agent_pending_bytes == 0:
                await asyncio.sleep(0.01)
            # Barge-in almost immediately: nearly nothing was heard yet.
            await session.interrupt()
            assert recorder.agent_pending_bytes < int(1.0 * SR) * 2
            await stt.finish()

        async def _run() -> None:
            async with aclosing(session.run(_mic())) as stream:
                driver: asyncio.Task | None = None
                async for ev in stream:
                    if isinstance(ev, SessionStarted):
                        started.set()
                        await stt.inject(TranscriptEvent(type="committed", text="Hello"))
                        driver = asyncio.create_task(_drive())
                if driver is not None:
                    await driver

        await asyncio.wait_for(_run(), timeout=10)

        audio = _decode(tmp_path / "call.mp3")
        duration = audio.shape[1] / SR
        assert 0 < duration < emitted_secs / 2  # the unheard tail never landed


class TestManifest:
    def test_manifest_written_next_to_the_audio_with_refreshed_duration(self, tmp_path: Path) -> None:
        from timbal.voice.session import TranscriptEntry

        rec = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
        rec.add_mic(_silence(0.5))
        rec.add_agent(_tone(0.5))  # queued; drained by close() after manifest build
        t0 = 1_000_000.0
        manifest = build_manifest(
            session_id="abc123",
            started_at=t0,
            meta={"transport": "webrtc", "model": "test/model"},
            transcript=[
                TranscriptEntry(role="user", text="Hello", timestamp=t0 + 1.2),
                TranscriptEntry(role="assistant", text="Hi!", timestamp=t0 + 2.5),
            ],
            turns=[],
            recorder=rec,
        )
        result = rec.close(manifest=manifest)
        assert result is not None and result.manifest_path is not None

        saved = json.loads(result.manifest_path.read_text())
        assert saved["session_id"] == "abc123"
        assert saved["meta"]["transport"] == "webrtc"
        assert [e["offset_ms"] for e in saved["transcript"]] == [1200, 2500]
        # 0.5s mic + 0.5s tail drained at close — not the pre-drain 0.5s.
        assert abs(saved["audio"]["duration_secs"] - 1.0) < 0.1
        assert abs(result.duration_secs - 1.0) < 0.1
