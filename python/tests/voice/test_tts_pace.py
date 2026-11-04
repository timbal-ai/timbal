"""Post-processed TTS bytes are what interruption accounting counts.

``heard_bytes`` is the played prefix of the audio that was emitted. The
segment record, ``_turn_audio_bytes``, playback and the call recorder must
all use that same length, or ``heard_text`` is cut against audio the caller
never heard.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="timbal[voice] extra (numpy) not installed")
pytest.importorskip("av", reason="timbal[voice] extra (av) not installed")

from timbal import Agent  # noqa: E402
from timbal.core.test_model import TestModel  # noqa: E402
from timbal.voice.events import AudioOutput  # noqa: E402
from timbal.voice.playback import map_played_bytes_to_text  # noqa: E402
from timbal.voice.providers import (  # noqa: E402
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    TTSStream,
)
from timbal.voice.recording import CallRecorder  # noqa: E402
from timbal.voice.session import VoiceSession  # noqa: E402

SR = 16_000
TEXT = "alpha beta gamma delta epsilon"


def _const(samples: int, amp: int) -> bytes:
    return np.full(samples, amp, dtype="<i2").tobytes()


def _sine(samples: int, freq: float = 440.0, amp: int = 8000) -> bytes:
    t = np.arange(samples)
    return (amp * np.sin(2 * np.pi * freq * t / SR)).astype("<i2").tobytes()


def _gap_phrase() -> bytes:
    speech = _const(int(0.12 * SR), 12_000)
    gap = _const(int(0.80 * SR), 0)
    return speech + gap + speech


class _SilentSTT(SpeechToText):
    async def connect(self, _config: AudioInputConfig) -> None:
        return None

    async def push_audio(self, _chunk: bytes) -> None:
        return None

    async def commit(self) -> None:
        return None

    async def events(self) -> AsyncIterator[TranscriptEvent]:
        if False:
            yield TranscriptEvent(type="committed", text="")

    async def close(self) -> None:
        return None


class _PcmTTS(TextToSpeech):
    """Yields one buffer in odd-sized pieces so the processors see chunk edges."""

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm

    async def connect(self, _config: AudioOutputConfig) -> None:
        return None

    async def synthesize(self, _text: str) -> AsyncIterator[bytes]:
        step = 701
        for i in range(0, len(self.pcm), step):
            yield self.pcm[i : i + step]

    async def close(self) -> None:
        return None


class _PieceStream(TTSStream):
    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm

    async def feed(self, _text: str) -> None:
        return None

    async def end(self) -> None:
        return None

    async def abort(self) -> None:
        return None

    async def audio(self) -> AsyncIterator[bytes]:
        step = 701
        for i in range(0, len(self._pcm), step):
            yield self._pcm[i : i + step]


class _GatedTTS(TextToSpeech):
    def __init__(self, first: bytes, second: bytes) -> None:
        self._first = first
        self._second = second
        self.entered_second = asyncio.Event()
        self.release = asyncio.Event()

    async def connect(self, _config: AudioOutputConfig) -> None:
        return None

    async def synthesize(self, _text: str) -> AsyncIterator[bytes]:
        yield self._first
        self.entered_second.set()
        await self.release.wait()
        yield self._second

    async def close(self) -> None:
        return None


def _session(tts: TextToSpeech, tmp_path: Path, **kwargs) -> tuple[VoiceSession, CallRecorder]:
    recorder = CallRecorder(tmp_path / "call.mp3", sample_rate=SR)
    session = VoiceSession(
        agent=Agent(name="t", model=TestModel(responses=["ok"]), tools=[]),
        stt=_SilentSTT(),
        tts=tts,
        recorder=recorder,
        turn_detector="heuristic",
        **kwargs,
    )
    return session, recorder


def _emitted(session: VoiceSession) -> bytes:
    parts: list[bytes] = []
    while not session._event_queue.empty():
        event = session._event_queue.get_nowait()
        if isinstance(event, AudioOutput):
            parts.append(event.data)
    return b"".join(parts)


def _assert_one_ledger(session: VoiceSession, recorder: CallRecorder, audio: bytes, raw: bytes) -> None:
    assert len(audio) < len(raw)
    record_bytes = sum(int(record[1]) for record in session._turn_tts_segment_records)
    assert record_bytes == len(audio)
    assert session._turn_audio_bytes == len(audio)
    assert recorder._agent_bytes == len(audio)
    assert session.playback._scheduled_bytes == len(audio)
    segments = [(str(text), int(nbytes)) for text, nbytes in session._turn_tts_segment_records]
    assert map_played_bytes_to_text(segments, len(audio)) == TEXT
    # Counting the provider's raw bytes would report the fully played reply as unheard.
    assert map_played_bytes_to_text([(TEXT, len(raw))], len(audio)) != TEXT
    half = len(audio) // 2
    heard = map_played_bytes_to_text(segments, half)
    assert heard
    assert heard != map_played_bytes_to_text([(TEXT, len(raw))], half)
    session._transcript.clear()
    session._turn_heard_bytes = len(audio)
    session._apply_interruption_truncation(ctx=None)
    assert session._last_interruption_heard_text == TEXT
    session._transcript.clear()
    session._last_interruption_heard_text = None
    session._turn_heard_bytes = half
    session._apply_interruption_truncation(ctx=None)
    assert session._last_interruption_heard_text == heard


class TestByteAccounting:
    async def test_defaults_leave_pcm_unchanged(self, tmp_path: Path) -> None:
        raw = _gap_phrase()
        session, recorder = _session(_PcmTTS(raw), tmp_path)
        try:
            await session._speak(TEXT)
            assert _emitted(session) == raw
            assert session._turn_audio_bytes == len(raw)
            assert recorder._agent_bytes == len(raw)
        finally:
            recorder.close()

    async def test_pause_trim_accounts_the_shortened_audio(self, tmp_path: Path) -> None:
        raw = _gap_phrase()
        session, recorder = _session(_PcmTTS(raw), tmp_path, max_tts_pause_secs=0.10)
        try:
            await session._speak(TEXT)
            _assert_one_ledger(session, recorder, _emitted(session), raw)
        finally:
            recorder.close()

    async def test_tempo_accounts_the_shorter_audio(self, tmp_path: Path) -> None:
        raw = _sine(int(0.50 * SR))
        session, recorder = _session(_PcmTTS(raw), tmp_path, tts_tempo=2.0)
        try:
            await session._speak(TEXT)
            _assert_one_ledger(session, recorder, _emitted(session), raw)
        finally:
            recorder.close()

    async def test_pause_then_tempo_share_one_ledger(self, tmp_path: Path) -> None:
        raw = _gap_phrase()
        session, recorder = _session(
            _PcmTTS(raw), tmp_path, max_tts_pause_secs=0.10, tts_tempo=1.5, tts_quiet_dbfs=-45.0
        )
        try:
            await session._speak(TEXT)
            _assert_one_ledger(session, recorder, _emitted(session), raw)
        finally:
            recorder.close()

    async def test_streaming_pump_counts_postprocessed_bytes(self, tmp_path: Path) -> None:
        raw = _gap_phrase()
        session, recorder = _session(_PcmTTS(raw), tmp_path, max_tts_pause_secs=0.10, tts_tempo=1.25)
        stream = _PieceStream(raw)
        record: list = [TEXT, 0]
        session._turn_tts_stream = stream
        session._turn_stream_record = record
        session._turn_tts_segment_records.append(record)
        try:
            await session._pump_tts_stream(stream, record)
            _assert_one_ledger(session, recorder, _emitted(session), raw)
        finally:
            recorder.close()

    async def test_non_pcm_encoding_is_left_unchanged(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        raw = _gap_phrase()
        session, recorder = _session(_PcmTTS(raw), tmp_path, max_tts_pause_secs=0.10, tts_tempo=2.0)
        session.audio_output.encoding = "mulaw"
        try:
            await session._speak(TEXT)
            await session._speak(TEXT)
            assert _emitted(session) == raw + raw
            assert capsys.readouterr().out.count("tts_pace_skipped") == 1
        finally:
            recorder.close()

    async def test_linear16_alias_is_still_processed(self, tmp_path: Path) -> None:
        raw = _gap_phrase()
        session, recorder = _session(_PcmTTS(raw), tmp_path, max_tts_pause_secs=0.10)
        session.audio_output.encoding = "linear16"
        try:
            await session._speak(TEXT)
            audio = _emitted(session)
            assert len(audio) < len(raw)
        finally:
            recorder.close()

    async def test_first_audio_is_not_held_for_a_later_pause(self, tmp_path: Path) -> None:
        speech = _const(int(0.05 * SR), 10_000)
        pause = _const(int(0.60 * SR), 0)
        tts = _GatedTTS(speech, pause)
        session, recorder = _session(tts, tmp_path, max_tts_pause_secs=0.10)
        task = asyncio.create_task(session._speak("hello there"))
        try:
            await asyncio.wait_for(tts.entered_second.wait(), timeout=1.0)
            assert session._turn_first_audio_at is not None
            assert _emitted(session) == speech
            tts.release.set()
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            tts.release.set()
            recorder.close()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
