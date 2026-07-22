"""Speech-to-speech (realtime) session seam.

Cascaded :class:`~timbal.voice.VoiceSession` assumes text at every joint:
STT commits text, the :class:`~timbal.core.agent.Agent` produces text, TTS
synthesizes text. Speech-to-speech models (OpenAI Realtime, Gemini Live,
Amazon Nova Sonic, ...) collapse STT+LLM+TTS into one bidirectional audio
session and own turn-taking server-side. They are **not** a drop-in
:class:`~timbal.voice.SpeechToText` — they need a sibling session type.

This module locks that contract:

* :class:`RealtimeModel` — the provider ABC. One instance == one live
  provider session (socket). It receives mic PCM and yields
  :class:`RealtimeEvent`s (transcripts, assistant text/audio, turn
  boundaries, provider-side barge-in).
* :class:`RealtimeSession` — adapts a :class:`RealtimeModel` onto the exact
  :class:`~timbal.voice.VoiceSessionEvent` vocabulary the cascaded session
  emits (``SessionStarted``, ``TranscriptPartial/Committed``,
  ``AgentTextDelta/Done``, ``AudioOutput``, ``SessionInterrupted``,
  ``TurnMetricsEvent``, ``SessionError``, ``SessionEnded``) so WebSocket
  protocol, playground, transcript recording, and metrics consumers work
  unchanged regardless of which pipeline produced the events.

What maps where (vs the cascaded session):

* **Turn detection** — none. The provider owns endpointing and barge-in
  (``interrupted`` events). There is no :class:`~timbal.voice.TurnDetector`
  seam here; server-side semantic VAD configuration is provider ``extra``.
* **Playback truth** — identical. The same
  :class:`~timbal.voice.PlaybackTracker` tracks the heard position, and on
  barge-in the session truncates transcript entries to the heard prefix and
  reports the turn-relative heard position to the provider via
  :meth:`RealtimeModel.truncate` (OpenAI ``conversation.item.truncate``
  shape, LiveKit-style).
* **Memory / tracing** — provider-held. The model keeps conversation state
  inside its own session; there is no :class:`~timbal.core.agent.Agent`, no
  ``RunContext`` chaining, and no trace spans. The session-level
  ``transcript`` is still recorded.
* **Metrics** — subset. ``eou_to_first_audio_ms`` (committed user transcript
  → first audio byte of the reply) and ``turn_total_ms`` are measured;
  LLM/TTS-split fields stay ``None`` because the provider does not expose
  the internal boundary.

No concrete provider ships yet — the contract is exercised by a scripted
fake in the test suite. Provider adapters (OpenAI Realtime, Gemini Live)
land against this frozen interface.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING, Literal

import structlog
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .metrics import TurnMetrics

from .playback import BufferedPlaybackTracker, PlaybackTracker, map_played_bytes_to_text
from .session import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SessionEnded,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    TranscriptCommitted,
    TranscriptEntry,
    TranscriptPartial,
    VoiceSessionEvent,
)

logger = structlog.get_logger("timbal.voice.realtime")


class RealtimeEvent(BaseModel):
    """Single event from a speech-to-speech provider session.

    ``text`` carries transcript / assistant text (or the error message for
    ``error``); ``data`` carries PCM for ``output_audio``. ``turn_done`` may
    set ``text`` to the provider's canonical full reply — when empty, the
    session uses the accumulated ``output_text_delta`` stream.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal[
        "input_transcript_partial",
        "input_transcript_committed",
        "turn_started",
        "output_text_delta",
        "output_audio",
        "turn_done",
        "interrupted",
        "error",
    ]
    text: str = ""
    data: bytes = b""


class RealtimeModel(ABC):
    """Bidirectional speech-to-speech provider session (STT+LLM+TTS+turns).

    Lifecycle: ``connect`` → ``send_audio`` / consume ``events`` → ``close``.
    One instance == one live provider session; instances are not reused
    across :class:`RealtimeSession`s.

    Providers must emit ``output_audio`` in the ``audio_output`` encoding they
    were connected with (PCM16LE mono at ``sample_rate``) so playback tracking
    and the WS protocol stay uniform with the cascaded pipeline.
    """

    @abstractmethod
    async def connect(self, audio_input: AudioInputConfig, audio_output: AudioOutputConfig) -> None:
        """Open the provider session. Called once, before any audio."""

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None:
        """Push a chunk of mic PCM (``audio_input`` encoding) to the provider."""

    @abstractmethod
    def events(self) -> AsyncIterator[RealtimeEvent]:
        """Provider event stream. Ending the stream ends the session."""

    async def truncate(self, played_ms: float) -> None:  # noqa: B027
        """The user heard only the first ``played_ms`` of the current reply.

        Called on barge-in with the turn-relative heard position so the
        provider can align its conversation state with reality (OpenAI
        ``conversation.item.truncate`` ``audio_end_ms`` shape). No-op default
        for providers that handle truncation fully server-side.
        """

    @abstractmethod
    async def close(self) -> None:
        """Tear down the provider session."""


class RealtimeSession:
    """Voice session backed by a speech-to-speech :class:`RealtimeModel`.

    Same calling convention as :class:`~timbal.voice.VoiceSession`: feed an
    async iterable of mic PCM to :meth:`run` and consume
    :class:`~timbal.voice.VoiceSessionEvent`s. After the session closes,
    :attr:`transcript` holds the committed conversation and (with
    ``record_audio=True``) :attr:`input_audio` / :attr:`output_audio` hold raw
    PCM.
    """

    def __init__(
        self,
        model: RealtimeModel,
        audio_input: AudioInputConfig | None = None,
        audio_output: AudioOutputConfig | None = None,
        *,
        playback_tracker: PlaybackTracker | None = None,
        record_audio: bool = False,
    ):
        self.model = model
        self.audio_input = audio_input or AudioInputConfig()
        self.audio_output = audio_output or AudioOutputConfig()
        # PCM16 mono: 2 bytes per sample.
        self._output_bps = self.audio_output.sample_rate * 2
        self.playback = playback_tracker or BufferedPlaybackTracker(bytes_per_second=self._output_bps)

        self._event_queue: asyncio.Queue[VoiceSessionEvent | None] = asyncio.Queue()
        self._closed = False

        # -- Per-turn state ---------------------------------------------------
        self._turn_index = 0
        self._turn_active = False
        # True between a turn's normal completion (turn_done) and the next turn
        # start / interruption: buffered audio may still be playing, and a
        # barge-in in that window must truncate the already-committed entry
        # (same contract as VoiceSession._turn_finalized_ok).
        self._turn_finalized_ok = False
        self._turn_text = ""
        self._turn_audio_bytes = 0
        self._turn_played_baseline = 0
        self._turn_started_at = 0.0
        self._turn_first_audio_at: float | None = None
        self._turn_eou_at: float | None = None
        self._turn_user_text = ""
        # Cumulative bytes emitted on the *played axis* this session (audio
        # discarded on interruption collapses out). The gapless client queue
        # plays everything already on the axis before a new turn's audio, so
        # this — not ``playback.played_bytes`` at turn start — is the correct
        # per-turn baseline when the previous reply's tail is still draining.
        self._axis_emitted_bytes = 0

        # -- Session recording --------------------------------------------------
        self._transcript: list[TranscriptEntry] = []
        self._record_audio = record_audio
        self._input_audio_chunks: list[bytes] = []
        self._output_audio_chunks: list[bytes] = []
        self._metrics: list[TurnMetrics] = []

    # -- Public: session recording ------------------------------------------

    @property
    def transcript(self) -> list[TranscriptEntry]:
        """Ordered transcript of committed user/assistant text for this session."""
        return list(self._transcript)

    @property
    def metrics(self) -> list[TurnMetrics]:
        """Per-turn latency metrics accumulated this session."""
        return list(self._metrics)

    @property
    def input_audio(self) -> bytes:
        """Raw PCM of mic input (empty when ``record_audio=False``)."""
        return b"".join(self._input_audio_chunks)

    @property
    def output_audio(self) -> bytes:
        """Raw PCM of provider audio output (empty when ``record_audio=False``)."""
        return b"".join(self._output_audio_chunks)

    # -- Public API ---------------------------------------------------------

    async def run(self, audio_in: AsyncIterable[bytes]) -> AsyncIterator[VoiceSessionEvent]:
        """Main loop. Yields events until the session is closed or errors out."""
        try:
            await self.model.connect(self.audio_input, self.audio_output)
            await self._emit(SessionStarted())

            audio_task = asyncio.create_task(self._forward_audio(audio_in))
            events_task = asyncio.create_task(self._process_model_events())

            try:
                while True:
                    event = await self._event_queue.get()
                    if event is None:  # sentinel → stop
                        break
                    yield event
            finally:
                for task in (audio_task, events_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(audio_task, events_task, return_exceptions=True)

        except Exception as e:
            logger.error("realtime_session_error", error=str(e), exc_info=True)
            yield SessionError(message=str(e))
        finally:
            await self._cleanup()
            yield SessionEnded()

    async def interrupt(self) -> None:
        """Client-side barge-in: truncate the in-flight reply to the heard prefix.

        Mirrors provider-side ``interrupted`` events; both funnel into the same
        truncation path. Also reports the turn-relative heard position to the
        provider via :meth:`RealtimeModel.truncate`.
        """
        await self._handle_interruption(notify_model=True)

    async def close(self) -> None:
        """Gracefully shut down the session."""
        if self._closed:
            return
        self._closed = True
        await self._emit(None)  # sentinel stops the run() iterator

    # -- Internal: audio uplink ----------------------------------------------

    async def _forward_audio(self, audio_in: AsyncIterable[bytes]) -> None:
        try:
            async for chunk in audio_in:
                if self._record_audio:
                    self._input_audio_chunks.append(chunk)
                await self.model.send_audio(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("realtime_audio_forward_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Audio input error: {e}"))

    # -- Internal: provider events → session events ---------------------------

    async def _process_model_events(self) -> None:
        try:
            async for event in self.model.events():
                await self._dispatch(event)
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("realtime_event_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Realtime model error: {e}"))
        await self.close()

    async def _dispatch(self, event: RealtimeEvent) -> None:
        if event.type == "input_transcript_partial":
            await self._emit(TranscriptPartial(text=event.text))
        elif event.type == "input_transcript_committed":
            text = event.text.strip()
            if not text:
                return
            self._turn_eou_at = time.monotonic()
            self._turn_user_text = text
            self._transcript.append(TranscriptEntry(role="user", text=text))
            await self._emit(TranscriptCommitted(text=text))
        elif event.type == "turn_started":
            self._begin_turn()
        elif event.type == "output_text_delta":
            if not self._turn_active:
                self._begin_turn()
            self._turn_text += event.text
            await self._emit(AgentTextDelta(text=event.text))
        elif event.type == "output_audio":
            if not self._turn_active:
                self._begin_turn()
            if not event.data:
                return
            if self._turn_first_audio_at is None:
                self._turn_first_audio_at = time.monotonic()
            self._turn_audio_bytes += len(event.data)
            if self._record_audio:
                self._output_audio_chunks.append(event.data)
            await self._emit(AudioOutput(data=event.data))
            self.playback.on_audio_emitted(len(event.data))
            self._axis_emitted_bytes += len(event.data)
        elif event.type == "turn_done":
            await self._finish_turn(final_text=event.text)
        elif event.type == "interrupted":
            # Provider already stopped generating; align local state and let it
            # know what was actually heard.
            await self._handle_interruption(notify_model=True)
        elif event.type == "error":
            await self._emit(SessionError(message=event.text or "Realtime model error"))

    # -- Internal: turn bookkeeping -------------------------------------------

    def _begin_turn(self) -> None:
        if self._turn_active:
            return
        self._turn_active = True
        self._turn_finalized_ok = False
        self._turn_index += 1
        self._turn_text = ""
        self._turn_audio_bytes = 0
        # Baseline on the played axis where *this turn's* audio begins. The
        # previous reply's tail may still be draining client-side, so
        # ``playback.played_bytes`` (the playhead) would land inside the old
        # turn's audio and count its tail as heard bytes of this turn.
        self._turn_played_baseline = self._axis_emitted_bytes
        self._turn_started_at = time.monotonic()
        self._turn_first_audio_at = None

    async def _finish_turn(self, *, final_text: str = "") -> None:
        if not self._turn_active:
            return
        # Providers may put the canonical full reply on turn_done; trust it when
        # it extends what was streamed.
        text = final_text if len(final_text) > len(self._turn_text) else self._turn_text
        self._turn_text = text
        if text.strip():
            self._transcript.append(TranscriptEntry(role="assistant", text=text))
        await self._emit(AgentTextDone(text=text))
        await self._emit_turn_metrics(interrupted=False)
        self._turn_active = False
        # Buffered audio may still be draining client-side; keep the turn's
        # text/bytes/baseline intact so a barge-in in that window can still
        # truncate the committed entry to the heard prefix.
        self._turn_finalized_ok = True
        self._turn_eou_at = None

    async def _handle_interruption(self, *, notify_model: bool) -> None:
        if not self._turn_active and not self.playback.is_playing:
            return
        heard_bytes = max(0, self.playback.played_bytes - self._turn_played_baseline)
        self.playback.on_interrupted()
        heard_text: str | None = None
        if self._turn_text or self._turn_audio_bytes:
            # S2S interleaves text and audio without segment boundaries: map the
            # heard position proportionally over the whole turn.
            heard_text = map_played_bytes_to_text(
                [(self._turn_text, self._turn_audio_bytes)], heard_bytes
            )
        if self._turn_active:
            if heard_text:
                self._transcript.append(TranscriptEntry(role="assistant", text=heard_text))
            await self._emit_turn_metrics(interrupted=True, heard_bytes=heard_bytes)
        elif self._turn_finalized_ok:
            # The turn completed (turn_done: full reply committed, metrics
            # emitted) but buffered audio was still playing: rewrite the
            # committed entry in place to the heard prefix. Metrics for the
            # turn already went out with interrupted=False — same contract as
            # the cascaded session's post-completion truncation.
            if self._transcript and self._transcript[-1].role == "assistant":
                if heard_text:
                    self._transcript[-1] = TranscriptEntry(role="assistant", text=heard_text)
                else:
                    self._transcript.pop()
        if notify_model:
            played_ms = heard_bytes / self._output_bps * 1000
            try:
                await self.model.truncate(played_ms)
            except Exception as e:
                logger.warning("realtime_truncate_failed", error=str(e))
        self._turn_active = False
        self._turn_finalized_ok = False
        self._turn_eou_at = None
        await self._emit(SessionInterrupted(heard_text=heard_text))

    async def _emit_turn_metrics(self, *, interrupted: bool, heard_bytes: int | None = None) -> None:
        from .metrics import TurnMetrics, TurnMetricsEvent

        def _ms(t0: float | None, t1: float | None) -> float | None:
            if t0 is None or t1 is None:
                return None
            return round((t1 - t0) * 1000, 1)

        now = time.monotonic()
        eou_to_first_audio = _ms(self._turn_eou_at, self._turn_first_audio_at)
        metrics = TurnMetrics(
            turn_index=self._turn_index,
            user_text_chars=len(self._turn_user_text),
            eou_to_llm_first_token_ms=None,
            eou_to_tts_first_byte_ms=eou_to_first_audio,
            eou_to_first_audio_ms=eou_to_first_audio,
            llm_total_ms=None,
            tts_total_ms=None,
            turn_total_ms=_ms(self._turn_started_at or None, now) or 0.0,
            interrupted=interrupted,
            tts_segments=0,
            audio_bytes=self._turn_audio_bytes,
            playback_acks_received=self.playback.ack_received,
            heard_bytes=heard_bytes,
        )
        self._metrics.append(metrics)
        await self._emit(TurnMetricsEvent(metrics=metrics))

    # -- Internal: helpers -----------------------------------------------------

    async def _emit(self, event: VoiceSessionEvent | None) -> None:
        await self._event_queue.put(event)

    async def _cleanup(self) -> None:
        try:
            await self.model.close()
        except Exception as e:
            logger.debug("realtime_model_close_failed", error=str(e))
