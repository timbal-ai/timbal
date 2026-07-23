"""
VoiceSession — voice-enabled agent session.

Orchestrates a real-time voice conversation:
    Audio input → STT → Agent → TTS → Audio output

All provider-specific logic (STT/TTS) is injected via abstract classes,
keeping VoiceSession provider-agnostic.

Interruption policy:
    User barge-in cancels both TTS playback and the current agent turn.
    Background tasks (run_in_background) are NOT cancelled — the agent
    decides their lifecycle via get_background_task or explicit cancellation.

TTS is scheduled on a chained background task so the agent stream can drain
(LLM ``OutputEvent`` + trace ``OUTPUT``) without waiting for audio synthesis.
"""

from __future__ import annotations

import asyncio
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ..core.agent import Agent
from ..state import get_run_context, set_run_context
from ..state.context import RunContext
from ..types.content import TextContent
from ..types.events import OutputEvent
from ..types.events.delta import DeltaEvent, Text, TextDelta
from ..types.message import Message
from .playback import BufferedPlaybackTracker, PlaybackTracker, map_played_bytes_to_text
from .turn_detection import (
    CommitAction,
    PartialDecision,
    TurnDetector,
    TurnState,
    resolve_turn_detector,
)

if TYPE_CHECKING:
    from .endpointing import VadEndpointer
    from .metrics import TurnMetrics

logger = structlog.get_logger("timbal.voice.session")


def _trace_debug_fields() -> dict[str, Any]:
    """Best-effort tracing ids for debug logs (safe when no RunContext)."""
    ctx = get_run_context()
    if ctx is None:
        return {}
    out: dict[str, Any] = {"run_id": ctx.id}
    try:
        sp = ctx.current_span()
        out["span_path"] = sp.path
        out["span_call_id"] = sp.call_id
    except Exception:
        pass
    return out


# Strip markdown formatting before TTS so bold/headers/lists don't get read aloud.
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_NUMBERED_LIST = re.compile(r"^\d+\.\s+", re.MULTILINE)
_MD_BULLET = re.compile(r"^[-*]\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_NUMBERED_LIST.sub("", text)
    text = _MD_BULLET.sub("", text)
    return text


# TTS flush: send text to ElevenLabs when we have a clause boundary so audio tracks the LLM
# without waiting for huge buffers. ``first_segment`` uses a low threshold so the first
# sentence (e.g. "Hello!") reaches TTS quickly even if the model omits a space after "!".
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n;:])\s+")
# Minimum chars before flushing to TTS.  Bigger segments = better prosody (ElevenLabs
# multi-context has no cross-context continuity, so each segment's intonation is
# independent).  Too small → choppy "final" intonation at every boundary.
# When audio is already playing, _flush_segment skips these thresholds entirely and
# buffers up to MAX_TTS_BUFFER_CHARS for maximum prosody quality.
MIN_FLUSH_CHARS = 24
FIRST_SEGMENT_MIN_CHARS = 6
MAX_TTS_BUFFER_CHARS = 200

# Clause-ending chars for flush heuristics (ASCII + common Spanish + fullwidth variants).
_CLAUSE_END_CHARS = frozenset(".!?;\n:\uff1f\uff01")


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _nfc_aligned_prefix_end(final_assistant: str, scheduled: str) -> int | None:
    """Byte index in ``final_assistant`` after a prefix whose NFC equals NFC(``scheduled``)."""
    sn = _nfc(scheduled)
    if not sn:
        return 0
    acc = ""
    for i, ch in enumerate(final_assistant):
        acc += ch
        na = _nfc(acc)
        if na == sn:
            return i + 1
        if not sn.startswith(na):
            return None
    return None


def _pending_tts_after_scheduled(scheduled: str, final_assistant: str) -> str:
    """Substring of ``final_assistant`` not yet passed to TTS this turn.

    We concatenate every ``_schedule_tts`` argument into ``scheduled``; if streaming
    flush rules skip a tail (or a delta is dropped), this catches it at LLM OUTPUT.

    Uses NFC-aligned prefix matching because the streamed deltas and the terminal
    ``Message`` text can differ in Unicode normalization (Gemini / OpenAI-compatible).
    """
    if not final_assistant:
        return ""
    if not scheduled:
        return final_assistant
    if final_assistant.startswith(scheduled):
        return final_assistant[len(scheduled) :]
    end = _nfc_aligned_prefix_end(final_assistant, scheduled)
    if end is not None:
        return final_assistant[end:]
    return ""


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class VoiceSessionEvent(BaseModel):
    """Base for all events emitted by a :class:`VoiceSession`."""

    type: str


class SessionStarted(VoiceSessionEvent):
    type: Literal["session_started"] = "session_started"


class SessionEnded(VoiceSessionEvent):
    type: Literal["session_ended"] = "session_ended"


class TranscriptPartial(VoiceSessionEvent):
    type: Literal["transcript_partial"] = "transcript_partial"
    text: str


class TranscriptCommitted(VoiceSessionEvent):
    type: Literal["transcript_committed"] = "transcript_committed"
    text: str


class AgentTextDelta(VoiceSessionEvent):
    type: Literal["agent_text_delta"] = "agent_text_delta"
    text: str


class AgentTextDone(VoiceSessionEvent):
    type: Literal["agent_text_done"] = "agent_text_done"
    text: str


class AudioOutput(VoiceSessionEvent):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: Literal["audio_output"] = "audio_output"
    data: bytes


class SessionInterrupted(VoiceSessionEvent):
    type: Literal["interrupted"] = "interrupted"
    heard_text: str | None = None
    """Assistant text the user actually heard before the interruption (None if unknown/none)."""


class SessionError(VoiceSessionEvent):
    type: Literal["error"] = "error"
    message: str


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class AudioInputConfig(BaseModel):
    """Cross-provider STT configuration.

    Generic fields cover the common surface across providers.
    Provider-specific knobs go in ``extra``.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    language: str | None = None
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    extra: dict[str, Any] = Field(default_factory=dict)


class AudioOutputConfig(BaseModel):
    """Cross-provider TTS configuration.

    Generic fields cover the common surface across providers.
    Provider-specific knobs go in ``extra``.
    """

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    voice: str | None = None
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    extra: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Provider ABCs
# ---------------------------------------------------------------------------


class TranscriptEvent(BaseModel):
    """Single event from an STT provider."""

    type: Literal["partial", "committed", "error"]
    text: str


class SpeechToText(ABC):
    """Abstract STT provider.

    Lifecycle: ``connect`` → push audio / consume ``events`` → ``close``.
    """

    @abstractmethod
    async def connect(self, config: AudioInputConfig) -> None: ...

    @abstractmethod
    async def push_audio(self, chunk: bytes) -> None: ...

    @abstractmethod
    async def commit(self) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[TranscriptEvent]: ...

    @abstractmethod
    async def close(self) -> None: ...


class TTSStream(ABC):
    """One streaming synthesis unit (e.g. an ElevenLabs *context*): text is fed
    incrementally while audio is read concurrently from :meth:`audio`.

    Contract:

    * ``feed`` may be called any number of times (in order).
    * ``end`` signals no more text; ``audio()`` finishes once the provider
      drains the remaining synthesis.
    * ``abort`` (barge-in) stops generation ASAP and unblocks ``audio()``.
    * All methods are idempotent-safe after ``end``/``abort``.
    """

    @abstractmethod
    async def feed(self, text: str) -> None: ...

    @abstractmethod
    async def end(self) -> None: ...

    @abstractmethod
    async def abort(self) -> None: ...

    @abstractmethod
    def audio(self) -> AsyncIterator[bytes]: ...


class TextToSpeech(ABC):
    """Abstract TTS provider.

    Lifecycle: ``connect`` → ``synthesize`` (repeatable) → ``close``.
    """

    @abstractmethod
    async def connect(self, config: AudioOutputConfig) -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

    def open_stream(self) -> TTSStream | None:
        """Optional capability: a per-reply streaming context.

        Providers that support incremental text input with prosody continuity
        (ElevenLabs multi-context WS) return a fresh :class:`TTSStream` per
        call; the session then feeds each flush into ONE context instead of
        synthesizing independent segments — independent segments each get
        "final sentence" intonation and an audible seam between them.
        Default ``None`` → the session falls back to per-segment
        ``synthesize``.
        """
        return None

    @abstractmethod
    async def close(self) -> None: ...


# ---------------------------------------------------------------------------
# VoiceSession
# ---------------------------------------------------------------------------


class TranscriptEntry(BaseModel):
    """Single entry in the session transcript."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str
    timestamp: float = Field(default_factory=time.time)
    # TODO(tool-filler): re-add `filler: bool = False` when tool-call filler
    # speech returns, so transcript entries for spoken latency-masking phrases
    # ("let me check that") are distinguishable from real reply text.


class VoiceSession:
    """Voice-enabled agent session.

    Composes a Timbal :class:`Agent` with pluggable STT / TTS providers to
    run a real-time voice conversation.  Consumes an async stream of audio
    bytes and yields :class:`VoiceSessionEvent` instances.

    Interruption policy
    -------------------
    A user barge-in (new committed transcript while the agent is speaking)
    cancels **both** TTS playback **and** the in-flight agent turn.
    Background tasks spawned via ``run_in_background`` are **not** cancelled;
    the agent decides their lifecycle.

    After the session closes, :attr:`transcript` contains the ordered list of
    committed user/assistant text, and (when ``record_audio=True``)
    :attr:`input_audio` / :attr:`output_audio` hold raw PCM bytes.
    """

    def __init__(
        self,
        agent: Agent,
        stt: SpeechToText,
        tts: TextToSpeech,
        audio_input: AudioInputConfig | None = None,
        audio_output: AudioOutputConfig | None = None,
        *,
        turn_detector: TurnDetector | str | None = None,
        playback_tracker: PlaybackTracker | None = None,
        record_audio: bool = False,
        hold_timeout_secs: float = 1.5,
        vad_endpointing: bool | VadEndpointer | None = None,
    ):
        self.agent = agent
        self.stt = stt
        self.tts = tts
        # Always clone: the session owns the detector's start/push_audio/close
        # lifecycle, and the spec may be a shared instance (server voice_config)
        # or a factory returning a singleton. Inspect ``session.turn_detector``,
        # not the object you passed in.
        self.turn_detector = resolve_turn_detector(turn_detector).clone()
        self.audio_input = audio_input or AudioInputConfig()
        self.audio_output = audio_output or AudioOutputConfig()
        # PCM16 mono: 2 bytes per sample.
        self.playback = playback_tracker or BufferedPlaybackTracker(
            bytes_per_second=self.audio_output.sample_rate * 2,
        )
        # Default HOLD timeout when a detector returns CommitAction.HOLD without
        # a per-decision override. Heuristic/provider never HOLD, so this is inert
        # unless an opt-in detector (local / lexical) is used.
        self.hold_timeout_secs = hold_timeout_secs
        # VAD endpointing fast path (Silero speech-stop → audio EOU → stt.commit()).
        # None = auto: on when the detector exposes an audio EOU model and the
        # timbal[voice] extra is installed; False = off; True = on (warn when
        # unavailable); a VadEndpointer instance = use as-is (custom knobs).
        self._vad_endpointing = vad_endpointing
        self._endpointer: VadEndpointer | None = None
        # time.monotonic() of the last endpointer-forced stt.commit(); the next
        # committed transcript within a short window is attributed to it.
        self._endpoint_commit_sent_at: float | None = None
        self._turn_vad_endpointed = False
        self._llm_warmup_task: asyncio.Task[None] | None = None
        # TODO(tool-filler): re-add the `filler` constructor param (phrases
        # rotated across turns, or a `(tool_name) -> str | None` callable) to
        # mask tool-call dead air with spoken filler. Removed for now to keep
        # the session surface small while turn detection stabilizes.

        self._event_queue: asyncio.Queue[VoiceSessionEvent | None] = asyncio.Queue()
        self._cancel_turn = asyncio.Event()
        self._current_turn_task: asyncio.Task | None = None
        self._is_speaking = False
        self._closed = False
        self._held_user_text: str | None = None
        self._hold_task: asyncio.Task | None = None
        self._hold_armed_timeout_secs: float | None = None

        # Tracks the RunContext from the last completed turn so the agent's
        # __call__ auto-chains parent_id for multi-turn memory.
        self._last_run_context: RunContext | None = None

        # Assistant text accumulated during the in-flight turn (for STT echo suppression).
        self._turn_assistant_text: str = ""

        # User text for the in-flight turn.  Streaming STT (e.g. ElevenLabs VAD) often
        # emits a second ``committed_transcript`` that extends the first; without this,
        # we treat it as barge-in and cancel the agent mid-reply.
        self._active_turn_user_text: str = ""
        self._turn_started_at: float = 0.0
        self._last_commit_at: float = 0.0
        self._partials_since_last_commit: int = 0
        self._last_partial_at: float = 0.0
        # Latest non-empty STT partial text — fed to the VAD endpointer's
        # optional text_score so a mid-thought hedge can bump the delay even
        # before the provider commits.
        self._latest_partial_text: str = ""
        # A pending HOLD must not expire while the user is audibly mid-utterance
        # (recent STT partial): the upcoming commit merges with / supersedes the
        # hold. Must exceed the STT VAD silence threshold (~1.2s default) so the
        # commit always lands before the extended expiry re-fires.
        self._hold_partial_grace_secs = 2.0
        # When the last STT commit event arrived. Anchors the hold-expiry
        # extension: only partials *newer* than the commit mean the user
        # resumed speaking — the committed fragment's own trailing partial
        # refinements must not stretch the hold.
        self._commit_event_at: float = 0.0
        # Watchdog for transcripts the provider never commits: STT can emit a
        # partial (e.g. quiet speech ducked by AEC during assistant playback)
        # whose VAD never registers an utterance — no commit ever fires and the
        # words hang as a "…" caption forever. After this much silence past the
        # last partial (comfortably beyond the provider's ~1.2s debounce, so it
        # only fires when the provider clearly won't), force ``stt.commit()``.
        self._stale_partial_commit_secs = 2.5
        self._stale_partial_poll_secs = 0.5
        self._stale_commit_sent_at = 0.0

        # Serial TTS runs off the agent ``async for`` critical path so we keep pulling
        # LLM/Agent events (and emit trace OUTPUT) while audio still synthesizes.
        self._tts_tail: asyncio.Task | None = None
        self._tts_tasks: set[asyncio.Task] = set()
        # Concatenation of all strings passed to ``_schedule_tts`` this turn (OUTPUT tail catch-up).
        self._turn_tts_scheduled_text: str = ""
        # Streaming TTS (providers with ``open_stream``): one context per reply,
        # fed incrementally, drained by one pump task. ``None`` → per-segment
        # ``synthesize`` fallback.
        self._turn_tts_stream: TTSStream | None = None
        self._turn_tts_pump: asyncio.Task | None = None
        self._turn_stream_record: list | None = None

        # Per-turn playback accounting for interruption truncation: played bytes
        # at turn start, spoken (text, bytes) records per TTS segment, and the
        # heard-bytes snapshot captured by interrupt() before the buffer clears.
        self._turn_played_baseline = 0
        self._turn_tts_segment_records: list[list] = []
        self._turn_heard_bytes: int | None = None
        self._last_interruption_heard_text: str | None = None
        # True between a turn's normal completion and the next turn start; a
        # barge-in in that window must truncate the already-committed entry.
        self._turn_finalized_ok = False

        # -- Session recording --------------------------------------------------
        self._transcript: list[TranscriptEntry] = []
        self._record_audio = record_audio
        self._input_audio_chunks: list[bytes] = []
        self._output_audio_chunks: list[bytes] = []

        # -- Per-turn latency metrics (time.monotonic stamps) --------------------
        self._metrics: list[TurnMetrics] = []
        self._turn_index = 0
        self._turn_eou_at = 0.0
        self._turn_first_token_at: float | None = None
        self._turn_first_audio_at: float | None = None
        self._turn_llm_done_at: float | None = None
        self._turn_tts_started_at: float | None = None
        self._turn_tts_ended_at: float | None = None
        self._turn_tts_segments = 0
        self._turn_audio_bytes = 0

    # -- Public: session recording ------------------------------------------

    @property
    def transcript(self) -> list[TranscriptEntry]:
        """Ordered transcript of committed user/assistant text for this session."""
        return list(self._transcript)

    @property
    def metrics(self) -> list[TurnMetrics]:
        """Per-turn latency metrics accumulated this session (one entry per turn attempt)."""
        return list(self._metrics)

    @property
    def input_audio(self) -> bytes:
        """Raw PCM of mic input (empty when ``record_audio=False``)."""
        return b"".join(self._input_audio_chunks)

    @property
    def output_audio(self) -> bytes:
        """Raw PCM of TTS output (empty when ``record_audio=False``)."""
        return b"".join(self._output_audio_chunks)

    # -- Playback tracking --------------------------------------------------

    @property
    def _assistant_audio_playing(self) -> bool:
        """True if the client likely still has queued audio to play."""
        return self.playback.is_playing

    @property
    def _assistant_active(self) -> bool:
        """True if the agent is generating OR audio is still playing in the browser."""
        return self._is_speaking or self._assistant_audio_playing

    # -- Public API ---------------------------------------------------------

    async def run(self, audio_in: AsyncIterable[bytes]) -> AsyncIterator[VoiceSessionEvent]:
        """Main loop.  Yields events until the session is closed or errors out."""
        try:
            self._start_llm_warmup()
            await self.stt.connect(self.audio_input)
            await self.tts.connect(self.audio_output)
            await self.turn_detector.start(self.audio_input)
            await self._maybe_start_endpointer()
            await self._emit(SessionStarted())

            audio_task = asyncio.create_task(self._forward_audio(audio_in))
            stt_task = asyncio.create_task(self._process_stt_events())
            sweep_task = asyncio.create_task(self._sweep_stale_partials())

            try:
                while True:
                    event = await self._event_queue.get()
                    if event is None:  # sentinel → stop
                        break
                    yield event
            finally:
                for task in (audio_task, stt_task, sweep_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(audio_task, stt_task, sweep_task, return_exceptions=True)

        except Exception as e:
            logger.error("voice_session_error", error=str(e), exc_info=True)
            yield SessionError(message=str(e))
        finally:
            await self._cleanup()
            yield SessionEnded()

    async def interrupt(self, *, truncate_completed: bool = True) -> None:
        """Cancel TTS playback and the current agent turn.

        ``truncate_completed`` controls the post-completion truncation path: on
        a real barge-in a finished-but-still-playing reply is rewritten to what
        was heard; on session close (``close()``) the committed transcript is
        left untouched.

        Always cancels an in-flight ``_current_turn_task`` even before
        ``_is_speaking`` flips true — HOLD expiry / commit races can leave an
        orphan task that must not double-run with the next turn.
        """
        has_turn = self._current_turn_task is not None and not self._current_turn_task.done()
        was_active = self._assistant_active
        if not was_active and not has_turn:
            logger.debug(
                "session_interrupt_skipped",
                reason="not_active",
                cancel_turn_set=self._cancel_turn.is_set(),
                audio_playing=self._assistant_audio_playing,
                **_trace_debug_fields(),
            )
            return
        logger.debug(
            "session_interrupt_begin",
            turn_task_done=(self._current_turn_task.done() if self._current_turn_task is not None else None),
            assistant_chars=len(self._turn_assistant_text),
            audio_playing=self._assistant_audio_playing,
            was_active=was_active,
            has_turn=has_turn,
            **_trace_debug_fields(),
        )
        # Snapshot how much of this turn's audio was heard *before* the playback
        # buffer is cleared — the truncation in _run_turn's finally needs it.
        if was_active and self._turn_heard_bytes is None:
            self._turn_heard_bytes = max(0, self.playback.played_bytes - self._turn_played_baseline)
        self._is_speaking = False
        if was_active:
            self.playback.on_interrupted()
        self._cancel_turn.set()
        for t in list(self._tts_tasks):
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks.clear()
        self._tts_tail = None
        # Streaming TTS: closing the context stops generation server-side and
        # unblocks the pump (which the turn task may be draining right now).
        await self._abort_tts_stream()
        if has_turn:
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        if (
            was_active
            and self._turn_finalized_ok
            and truncate_completed
            and self._last_interruption_heard_text is None
        ):
            # The turn completed (AgentTextDone emitted, full reply committed to
            # transcript/memory) but buffered audio was still playing: rewrite the
            # committed entry in place to the heard prefix. Checked *after* the
            # cancel/await above because the barge-in can land while the finished
            # turn's ``finally`` is still persisting the trace — the task is not
            # ``done()`` yet, the cancel is swallowed there, and no truncation has
            # run (``_last_interruption_heard_text`` would be set otherwise).
            #
            # Re-save after truncation: the turn's finally already persisted the
            # *pre*-truncation snapshot for serializing providers (jsonl/platform).
            # Live session/memory are mutated in place; without this re-save the
            # stored trace still shows the full unheard reply.
            try:
                self._apply_interruption_truncation(ctx=self._last_run_context, replace_last=True)
            except Exception as e:
                logger.warning("turn_truncation_failed", error=str(e), exc_info=True)
            ctx = self._last_run_context
            if ctx is not None:
                try:
                    await asyncio.shield(ctx._save_trace())
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug("interrupt_trace_resave_failed", error=str(e))
        self._turn_finalized_ok = False
        # Defensive: if the cancelled turn's finally aborted mid-way, don't leave
        # a zombie active_user_text for the next detector snapshot.
        if not (self._current_turn_task and not self._current_turn_task.done()):
            self._active_turn_user_text = ""
        if was_active:
            # The cancelled turn's finally (or the in-place path above) computed
            # what the user actually heard.
            heard = self._last_interruption_heard_text
            await self._emit(SessionInterrupted(heard_text=heard))
            self._last_interruption_heard_text = None
            # INFO on purpose: this event makes the client rewrite (or remove,
            # when heard_text is empty) the displayed reply — essential context
            # when a reply "disappears" in the UI.
            logger.info(
                "session_interrupt_emitted",
                heard_text_preview=(heard[:120] if heard else heard),
                heard_bytes=self._turn_heard_bytes,
                **_trace_debug_fields(),
            )

    async def close(self) -> None:
        """Gracefully shut down the session."""
        if self._closed:
            return
        self._closed = True
        self._cancel_hold()
        self._held_user_text = None
        await self.interrupt(truncate_completed=False)
        await self._emit(None)  # sentinel stops the run() iterator

    # -- Internal: first-reply warmup -----------------------------------------

    def _start_llm_warmup(self) -> None:
        """Fire-and-forget: pre-establish the LLM provider's connection pool.

        The first turn of a session otherwise pays the provider's TCP+TLS
        handshake inside its TTFT (measured: 1.84s cold vs 0.54s warm against
        OpenAI). Only applies to string model specs — a ``TestModel`` (or any
        custom model object) has no provider connection to warm.
        """
        model = getattr(self.agent, "model", None)
        if not (isinstance(model, str) and "/" in model):
            return

        from ..core.llm_router import warmup_llm_connection

        self._llm_warmup_task = asyncio.create_task(warmup_llm_connection(model))

    # -- Internal: VAD endpointing fast path ---------------------------------

    async def _maybe_start_endpointer(self) -> None:
        """Arm the local VAD endpointing loop when the pieces exist.

        Requires a turn detector exposing an audio EOU (``score_recent_audio``
        with a non-None ``audio_eou``) and the ``timbal[voice]`` extra for
        Silero. Silently stays off otherwise (warning when explicitly
        requested). See :mod:`timbal.voice.endpointing`.
        """
        spec = self._vad_endpointing
        if spec is False:
            return
        explicit = spec is not None
        score_fn = getattr(self.turn_detector, "score_recent_audio", None)
        audio_eou = getattr(self.turn_detector, "audio_eou", None)
        if score_fn is None or audio_eou is None:
            if explicit:
                logger.warning(
                    "vad_endpointing_unavailable",
                    reason="turn detector has no audio EOU model",
                    detector=type(self.turn_detector).__name__,
                )
            return
        if spec is None or spec is True:
            from .endpointing import VadEndpointer

            endpointer = VadEndpointer()
        else:
            endpointer = spec
        endpointer.bind(
            score=score_fn,
            commit=self._endpoint_commit,
            should_commit=self._endpoint_should_commit,
            text_score=self._endpoint_text_score,
        )
        try:
            await endpointer.start(sample_rate=self.audio_input.sample_rate)
        except ImportError as e:
            log = logger.warning if explicit else logger.debug
            log(
                "vad_endpointing_unavailable",
                reason="timbal[voice] extra not installed",
                error=str(e),
            )
            return
        except Exception as e:
            logger.warning("vad_endpointer_start_failed", error=str(e))
            return
        self._endpointer = endpointer
        logger.info(
            "vad_endpointing_active",
            stop_silence_secs=endpointer.stop_silence_secs,
            max_delay_secs=endpointer.max_delay_secs,
        )

    def _endpoint_should_commit(self) -> bool:
        """Gate for the endpointer: only force-commit real transcribed speech.

        VAD alone can trigger on echo/noise the STT never heard; requiring a
        partial newer than the last commit means ElevenLabs actually has words
        in its buffer.
        """
        if self._closed:
            return False
        return self._last_partial_at > self._last_commit_at

    async def _endpoint_text_score(self) -> float | None:
        """``P(complete)`` for the latest STT partial, for VAD delay bumping.

        Uses the turn detector's text EOU when present (``fallback_text_eou``
        on :class:`~timbal.voice.LocalAudioTurnDetector`); otherwise the
        shared :class:`~timbal.voice.PunctuationEouPredictor` baseline. Returns
        ``None`` when there is no fresh partial — endpointer keeps the
        audio-only delay.
        """
        if self._last_partial_at <= self._last_commit_at or not self._latest_partial_text:
            return None
        text_eou = getattr(self.turn_detector, "fallback_text_eou", None)
        if text_eou is None:
            from .eou import PunctuationEouPredictor

            text_eou = PunctuationEouPredictor()
        return await text_eou.predict_eou(self._latest_partial_text)

    async def _endpoint_commit(self) -> None:
        self._endpoint_commit_sent_at = time.monotonic()
        await self.stt.commit()

    # Corroboration window for partial barge-ins: a real >=3-word interruption
    # means >=~1s of actual speech shortly before the partial arrives, so the
    # local VAD must have seen at least this much energy recently. STT can
    # hallucinate plausible multi-word phrases from silence/room noise (Whisper
    # -family behaviour) and those pass every *text* gate — but they carry no
    # mic energy. Same idea as LiveKit's min_interruption_duration / Pipecat's
    # volume-based interruption strategy.
    MIN_BARGE_IN_VAD_SPEECH_SECS = 0.25
    BARGE_IN_VAD_WINDOW_SECS = 2.0

    def _vad_vetoes_barge_in(self, text: str) -> bool:
        """True when the local VAD saw no real speech energy recently.

        Only active when the endpointer (Silero on the mic feed) is armed and
        healthy; otherwise returns False — never make the assistant
        uninterruptible on missing evidence. Speaker echo DOES carry energy, so
        this gate never vetoes echo; the text-similarity check handles that.
        """
        if self._endpointer is None:
            return False
        speech_secs = self._endpointer.speech_secs_in_window(self.BARGE_IN_VAD_WINDOW_SECS)
        if speech_secs is None or speech_secs >= self.MIN_BARGE_IN_VAD_SPEECH_SECS:
            return False
        # INFO on purpose: a vetoed barge-in is invisible otherwise, and "why
        # didn't it interrupt" / "why did it interrupt" are the two sides of
        # the same debugging session.
        logger.info(
            "stt_partial_barge_in_vetoed",
            reason="no_recent_vad_speech",
            vad_speech_secs=round(speech_secs, 3),
            text_preview=text[:80],
            audio_playing=self._assistant_audio_playing,
        )
        return True

    def _vad_contradicts_recent_partial(self) -> bool:
        """True when the local VAD is healthy and saw no real speech energy
        recently — a fresh STT partial is then a hallucination, not the user
        mid-utterance. Conservative on missing evidence (no endpointer /
        starved VAD → ``False``), mirroring :meth:`_vad_vetoes_barge_in`.
        """
        if self._endpointer is None:
            return False
        speech_secs = self._endpointer.speech_secs_in_window(self.BARGE_IN_VAD_WINDOW_SECS)
        return speech_secs is not None and speech_secs < self.MIN_BARGE_IN_VAD_SPEECH_SECS

    # -- Internal: audio → STT ---------------------------------------------

    async def _forward_audio(self, audio_in: AsyncIterable[bytes]) -> None:
        try:
            async for chunk in audio_in:
                if self._record_audio:
                    self._input_audio_chunks.append(chunk)
                self.turn_detector.push_audio(chunk)
                if self._endpointer is not None:
                    self._endpointer.push(chunk)
                await self.stt.push_audio(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("audio_forward_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Audio input error: {e}"))

    async def _sweep_stale_partials(self) -> None:
        """Watchdog: force-commit transcripts the provider never commits.

        Failure mode (seen live): the user speaks quietly while the assistant
        is playing — AEC ducks the near-end audio, the STT still transcribes a
        partial, but neither the provider VAD nor Silero registers an
        utterance. No commit ever fires and the words hang as a "…" caption
        until the session dies. When a partial has gone
        ``_stale_partial_commit_secs`` with no commit and no newer partial,
        force ``stt.commit()`` so the transcript flows through the normal
        ``_handle_committed`` path (providers without manual commit no-op).
        """
        try:
            while not self._closed:
                await asyncio.sleep(self._stale_partial_poll_secs)
                if self._closed:
                    return
                if self._last_partial_at <= self._last_commit_at:
                    continue
                # One forced commit per stranded partial: an IGNOREd commit
                # does not bump _last_commit_at, so without this guard a noise
                # partial would retrigger a commit every poll.
                if self._last_partial_at <= self._stale_commit_sent_at:
                    continue
                stale_secs = time.monotonic() - self._last_partial_at
                if stale_secs < self._stale_partial_commit_secs:
                    continue
                self._stale_commit_sent_at = time.monotonic()
                # INFO on purpose: this is the only trace that a transcript was
                # rescued from a provider that silently refused to commit.
                logger.info("stt_stale_partial_commit", stale_secs=round(stale_secs, 1))
                try:
                    await self.stt.commit()
                except Exception as e:
                    logger.warning("stt_stale_partial_commit_failed", error=str(e))
        except asyncio.CancelledError:
            return

    # -- Internal: STT → turns ---------------------------------------------

    async def _process_stt_events(self) -> None:
        try:
            async for event in self.stt.events():
                if event.type == "partial":
                    text = event.text.strip()
                    self._partials_since_last_commit += 1
                    if text:
                        self._last_partial_at = time.monotonic()
                        self._latest_partial_text = text
                    await self._emit(TranscriptPartial(text=text))
                    decision = await self.turn_detector.on_partial(text, self._turn_state())
                    if decision is PartialDecision.BARGE_IN and self._vad_vetoes_barge_in(text):
                        decision = PartialDecision.IGNORE
                    if decision is PartialDecision.BARGE_IN:
                        # INFO on purpose: a barge-in cancels TTS and truncates the
                        # committed reply — when debugging "the agent went silent /
                        # the reply vanished", this is the first thing to look for.
                        logger.info(
                            "stt_partial_barge_in",
                            text_preview=text[:80],
                            assistant_chars=len(self._turn_assistant_text),
                            audio_playing=self._assistant_audio_playing,
                        )
                        await self.interrupt()
                elif event.type == "committed" and event.text.strip():
                    await self._handle_committed(event.text.strip())
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("stt_event_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"STT error: {e}"))
        # STT stream ended (exhausted, error, or connection lost).
        # Wait for the last turn to finish before closing so AgentTextDone is emitted.
        if self._current_turn_task is not None and not self._current_turn_task.done():
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.close()

    def _turn_state(self) -> TurnState:
        """Snapshot of session state for :class:`TurnDetector` decisions."""
        now = time.monotonic()
        holding = self._held_user_text is not None
        # While HOLDing an incomplete commit, expose it as the active user text
        # so the next commit can refine/merge against it — but do NOT fold HOLD
        # into assistant_active (that breaks hallucination filters + refinements).
        active = self._held_user_text if holding else self._active_turn_user_text
        return TurnState(
            assistant_active=self._assistant_active,
            audio_playing=self._assistant_audio_playing,
            assistant_text=self._turn_assistant_text,
            active_user_text=active,
            seconds_since_turn_start=now - self._turn_started_at,
            seconds_since_last_commit=now - self._last_commit_at,
            partials_since_last_commit=self._partials_since_last_commit,
            holding=holding,
        )

    def _cancel_hold(self) -> None:
        if self._hold_task is not None and not self._hold_task.done():
            self._hold_task.cancel()
        self._hold_task = None

    async def _arm_hold(self, text: str, timeout_secs: float) -> None:
        """Defer starting a turn until more speech arrives or ``timeout_secs`` elapses."""
        self._cancel_hold()
        self._held_user_text = text
        self._hold_armed_timeout_secs = timeout_secs
        self._last_commit_at = time.monotonic()

        # Anchor for "user resumed speaking": partials older than the commit
        # event are the held fragment's own trailing refinements and must not
        # stretch the hold (they otherwise floor every expiry at the grace
        # window instead of ``timeout_secs``).
        anchor = self._commit_event_at if self._commit_event_at > 0 else time.monotonic()

        async def _expire() -> None:
            me = asyncio.current_task()
            try:
                remaining = timeout_secs
                while True:
                    await asyncio.sleep(remaining)
                    # Never fire mid-utterance: a *new* STT partial since the
                    # commit means the user resumed speaking, and their commit
                    # is about to merge with / supersede this hold. But require
                    # mic energy to corroborate: STT partials can hallucinate
                    # from silence (same failure as barge-ins) and would extend
                    # the hold into dead air.
                    since_partial = time.monotonic() - self._last_partial_at
                    if (
                        self._last_partial_at > anchor
                        and since_partial < self._hold_partial_grace_secs
                        and not self._vad_contradicts_recent_partial()
                    ):
                        remaining = self._hold_partial_grace_secs - since_partial
                        continue
                    break
            except asyncio.CancelledError:
                return
            # A refine/re-arm may have replaced us — do not wipe the new hold.
            if self._hold_task is not me:
                return
            held = self._held_user_text
            self._held_user_text = None
            self._hold_armed_timeout_secs = None
            if self._hold_task is me:
                self._hold_task = None
            if held and not self._closed:
                logger.info(
                    "stt_hold_expired",
                    text_preview=held[:120],
                    timeout_secs=timeout_secs,
                    **_trace_debug_fields(),
                )
                await self.interrupt()
                if self._closed:
                    return
                self._cancel_turn.clear()
                await self._begin_user_turn(held, replace_user_entry=False)

        self._hold_task = asyncio.create_task(_expire())
        logger.info(
            "stt_hold_armed",
            text_preview=text[:120],
            timeout_secs=timeout_secs,
            **_trace_debug_fields(),
        )

    async def _begin_user_turn(
        self, final_text: str, *, replace_user_entry: bool, vad_endpointed: bool = False
    ) -> None:
        """Record ``final_text`` and start an agent turn.

        Caller is responsible for ``interrupt()`` / clearing ``_cancel_turn``
        when an in-flight reply must be stopped first (HOLD expiry is usually
        idle; CONTINUE/NEW_TURN paths interrupt before calling this).
        """
        if self._closed:
            # Final gate: callers await (detector, interrupt) between their own
            # _closed checks and here; never start an agent turn after close().
            logger.debug("stt_turn_dropped_session_closed", text_preview=final_text[:80])
            return
        self._turn_vad_endpointed = vad_endpointed
        self._last_commit_at = time.monotonic()
        if replace_user_entry and self._transcript and self._transcript[-1].role == "user":
            self._transcript[-1] = TranscriptEntry(role="user", text=final_text)
        else:
            self._transcript.append(TranscriptEntry(role="user", text=final_text))
        await self._emit(TranscriptCommitted(text=final_text))
        self._active_turn_user_text = final_text
        self._turn_eou_at = time.monotonic()
        self._current_turn_task = asyncio.create_task(self._run_turn(final_text))
        logger.debug(
            "stt_turn_task_created",
            text_preview=final_text[:120],
            **_trace_debug_fields(),
        )

    async def _handle_committed(self, text: str) -> None:
        if self._closed:
            return
        self._commit_event_at = time.monotonic()
        # Any commit (endpointer-forced or provider debounce) makes a pending
        # VAD endpoint stale — the STT segment it targeted is already closed.
        if self._endpointer is not None:
            self._endpointer.notify_committed()
        vad_endpointed = (
            self._endpoint_commit_sent_at is not None
            and time.monotonic() - self._endpoint_commit_sent_at < 2.0
        )
        if vad_endpointed:
            # INFO: the observable payoff of the fast path — how much sooner
            # this commit landed than the provider's own debounce would have.
            logger.info(
                "vad_endpointed_commit",
                commit_latency_ms=round((time.monotonic() - self._endpoint_commit_sent_at) * 1000, 1),
                text_preview=text[:80],
            )
        self._endpoint_commit_sent_at = None
        state = self._turn_state()
        self._partials_since_last_commit = 0
        # Cancel the hold *timer* before awaiting the detector. Local audio EOU
        # yields to the event loop (executor), and a racing hold-expiry task can
        # otherwise start a turn on the old fragment while we are mid-decision
        # — then this path starts a second turn for the merge/continuation.
        # Keep ``_held_user_text`` so ``state.holding`` / active text stay correct;
        # HOLD re-arms below, NEW_TURN/CONTINUE clear it.
        self._cancel_hold()
        logger.debug(
            "stt_committed_received",
            text_preview=text[:160],
            text_len=len(text),
            partials_before=state.partials_since_last_commit,
            is_speaking=self._is_speaking,
            audio_playing=state.audio_playing,
            active_user_preview=(state.active_user_text[:100] if state.active_user_text else ""),
            assistant_so_far_chars=len(state.assistant_text),
            holding=self._held_user_text is not None,
            **_trace_debug_fields(),
        )
        decision = await self.turn_detector.on_committed(text, state)
        # The detector await yields to the event loop (local audio EOU runs on
        # the executor; a cold model load can take seconds) and close() may land
        # in that window — the server calls it the moment the client
        # disconnects. Starting a turn now would fire an LLM call against
        # closing providers and emit events nobody consumes; the hold-expiry
        # task makes the same check before beginning a turn.
        if self._closed:
            logger.debug("stt_commit_dropped_session_closed", text_preview=text[:80])
            return
        # INFO: one line per user utterance describing what the detector did —
        # the minimum needed to debug turn-taking without DEBUG-level firehose.
        if decision.action is CommitAction.IGNORE:
            logger.info(
                "stt_commit_ignored",
                reason=decision.reason,
                text_preview=text[:160],
            )
            # Timer was cancelled at the top of this method; re-arm so a noise/
            # hesitation commit mid-hold does not freeze the fragment forever.
            if self._held_user_text is not None and (self._hold_task is None or self._hold_task.done()):
                await self._arm_hold(
                    self._held_user_text,
                    self._hold_armed_timeout_secs
                    if self._hold_armed_timeout_secs is not None
                    else self.hold_timeout_secs,
                )
            return

        final_text = decision.text or text
        logger.info(
            "stt_committed_accepted",
            action=decision.action.value,
            reason=decision.reason,
            had_active_speech=self._is_speaking,
            text_preview=final_text[:160],
            **_trace_debug_fields(),
        )

        if decision.action is CommitAction.HOLD:
            # Stop any still-playing TTS (common right after agent_text_done).
            # NEW_TURN / CONTINUE_TURN always interrupt first; HOLD must too or
            # assistant audio keeps talking over the deferred user fragment.
            await self.interrupt()
            self._cancel_turn.clear()
            # Detector returns the full utterance to hold (refine/merge already applied).
            timeout = (
                decision.hold_timeout_secs if decision.hold_timeout_secs is not None else self.hold_timeout_secs
            )
            await self._arm_hold(final_text, timeout)
            return

        # A real accept cancels any pending HOLD, then interrupts so truncation
        # can append a heard assistant fragment *before* we rewrite the user
        # entry (CONTINUE_TURN must see that fragment to pop it).
        self._cancel_hold()
        self._held_user_text = None
        self._hold_armed_timeout_secs = None
        await self.interrupt()
        self._cancel_turn.clear()

        replace = False
        if decision.action is CommitAction.CONTINUE_TURN:
            # interrupt() may have recorded the heard fragment of the aborted
            # reply right after the fragment's user entry (interruption
            # truncation). A continuation merges the fragment into a single
            # utterance, so that reply is superseded — drop it so the merge
            # below updates the fragment user entry instead of appending a
            # duplicate user line around a stray assistant fragment.
            if (
                len(self._transcript) >= 2
                and self._transcript[-1].role == "assistant"
                and self._transcript[-2].role == "user"
                and self._transcript[-2].text == state.active_user_text
            ):
                self._transcript.pop()
            replace = bool(self._transcript and self._transcript[-1].role == "user")
            # Transcript cleanup alone is not enough — parent-chain memory still
            # has user:fragment + assistant:heard unless we mirror the rewrite.
            await self._align_continue_memory(fragment_user_text=state.active_user_text)

        await self._begin_user_turn(final_text, replace_user_entry=replace, vad_endpointed=vad_endpointed)

    # -- Internal: agent turn → TTS ----------------------------------------

    async def _run_turn(self, user_text: str) -> None:
        self._is_speaking = True
        self._turn_assistant_text = ""
        self._turn_started_at = time.monotonic()
        self._turn_index += 1
        self._turn_first_token_at = None
        self._turn_first_audio_at = None
        self._turn_llm_done_at = None
        self._turn_tts_started_at = None
        self._turn_tts_ended_at = None
        self._turn_tts_segments = 0
        self._turn_audio_bytes = 0
        self._turn_played_baseline = self.playback.played_bytes
        self._turn_tts_segment_records = []
        self._turn_heard_bytes = None
        self._turn_finalized_ok = False
        self._turn_tts_stream = None
        self._turn_tts_pump = None
        self._turn_stream_record = None
        agen = None
        # Where we were when cancel / finally ran (loop_exit is only set on break/else/exception;
        # CancelledError inside ``await _speak`` left the old code stuck at "not_started").
        turn_phase = "init"
        full_response = ""
        try:
            if self._last_run_context is not None:
                set_run_context(self._last_run_context)

            # ``RunContext.parent_id`` is the session parent chain, not "previous run for trace".
            # Runnable chains the next voice_live run using ``parent_id = prior_ctx.id``.
            logger.debug(
                "turn_begin",
                user_preview=user_text[:160],
                resume_from_saved_run_id=(self._last_run_context.id if self._last_run_context else None),
                saved_run_parent_id=(self._last_run_context.parent_id if self._last_run_context else None),
                **_trace_debug_fields(),
            )
            text_buffer = ""
            self._turn_tts_scheduled_text = ""

            turn_phase = "creating_agent_generator"
            msg = Message(role="user", content=[TextContent(text=user_text)])
            agen = self.agent(prompt=msg)
            async for event in agen:
                turn_phase = "awaiting_agent_event"
                if self._cancel_turn.is_set():
                    turn_phase = "cancel_turn_flag_at_iter_start"
                    logger.debug(
                        "turn_agent_loop_break",
                        reason="cancel_turn_set",
                        response_chars=len(full_response),
                        **_trace_debug_fields(),
                    )
                    break

                if isinstance(event, DeltaEvent) and isinstance(event.item, TextDelta | Text):
                    # Google (and others) often emit a full ``Text`` block first, then ``TextDelta`` tails.
                    chunk = event.item.text if isinstance(event.item, Text) else event.item.text_delta
                    if not chunk:
                        continue
                    if self._turn_first_token_at is None:
                        self._turn_first_token_at = time.monotonic()
                    full_response += chunk
                    text_buffer += chunk
                    self._turn_assistant_text += chunk
                    turn_phase = "emit_agent_text_delta"
                    await self._emit(AgentTextDelta(text=chunk))

                    # Flush from ``full_response`` vs ``_turn_tts_scheduled_text`` so we never
                    # depend on ``text_buffer`` drifting from the true unscheduled suffix (Gemini
                    # splits ``Text`` + ``text_delta`` unpredictably).
                    if full_response.startswith(self._turn_tts_scheduled_text):
                        tts_tail = full_response[len(self._turn_tts_scheduled_text) :]
                    else:
                        logger.warning(
                            "tts_stream_scheduled_mismatch",
                            scheduled_len=len(self._turn_tts_scheduled_text),
                            full_len=len(full_response),
                            **_trace_debug_fields(),
                        )
                        tts_tail = text_buffer
                    if tts_tail != text_buffer:
                        text_buffer = tts_tail
                    flush_text = _flush_segment(
                        tts_tail,
                        first_segment=len(self._turn_tts_scheduled_text) == 0,
                        audio_playing=self._assistant_audio_playing,
                    )
                    if flush_text is not None:
                        turn_phase = "tts_synthesize_flush"
                        logger.debug(
                            "turn_tts_flush_decision",
                            flush_chars=len(flush_text),
                            remainder_chars=len(tts_tail) - len(flush_text),
                            is_partial=flush_text != tts_tail,
                            **_trace_debug_fields(),
                        )
                        self._schedule_tts(flush_text)
                        text_buffer = tts_tail[len(flush_text) :]

                # TODO(tool-filler): hook here on `DeltaEvent` with a `ToolUse`
                # item — the earliest moment we know a tool call (dead air) is
                # coming for streaming providers — to speak a filler phrase.

                elif isinstance(event, OutputEvent):
                    # Outer Agent OUTPUT mirrors the LLM message; handling both would
                    # double-run reconcile / optional suffix TTS.
                    if not str(event.path).endswith(".llm"):
                        logger.debug(
                            "turn_skip_non_llm_output_event",
                            event_path=event.path,
                            **_trace_debug_fields(),
                        )
                        continue
                    turn_phase = "process_agent_output_event"
                    logger.debug(
                        "turn_agent_runnable_output_event",
                        event_path=event.path,
                        status_code=event.status.code,
                        status_reason=getattr(event.status, "reason", None),
                        output_is_none=event.output is None,
                        **_trace_debug_fields(),
                    )
                    if event.status.code == "success":
                        out = event.output
                        if isinstance(out, Message):
                            msg_text = out.collect_text()
                            # Models that skip streaming deliver all text here — this is
                            # the first moment LLM text exists, so stamp first-token now.
                            if msg_text and self._turn_first_token_at is None:
                                self._turn_first_token_at = time.monotonic()
                            # LLM generation for this iteration is complete; stamp before
                            # the TTS drain below so llm_total_ms excludes synthesis time.
                            # Multi-iteration agents overwrite — the last iteration wins.
                            self._turn_llm_done_at = time.monotonic()
                            # Anything still buffered from deltas must hit TTS before we ``anext``
                            # again: the next event pull runs the outer Runnable's post-hook,
                            # ``dump(output)``, and trace save — and the loop would not flush
                            # ``text_buffer`` until after the outer Agent OUTPUT is consumed.
                            if text_buffer and not self._cancel_turn.is_set():
                                turn_phase = "tts_flush_buffer_on_llm_output"
                                self._schedule_tts(text_buffer)
                                text_buffer = ""
                            if msg_text:
                                merged, suffix = _reconcile_final_assistant_text(full_response, msg_text)
                                if suffix is not None and merged != full_response:
                                    full_response = merged
                                    self._turn_assistant_text = full_response
                                    turn_phase = "emit_agent_text_delta_from_output"
                                    await self._emit(AgentTextDelta(text=suffix))
                                    if not self._cancel_turn.is_set():
                                        turn_phase = "tts_synthesize_from_output"
                                        self._schedule_tts(suffix)
                            # TODO(tool-filler): non-streaming providers (and
                            # TestModel) surface tool calls only here, via
                            # `out.stop_reason == "tool_use"` — second filler
                            # hook point, after the text handling above so any
                            # spoken text this turn suppresses the filler.
                            if not self._cancel_turn.is_set():
                                # Prefer API ``Message`` text, then streamed ``full_response``, so a
                                # Unicode/stream mismatch does not drop ``_pending_tts_after_scheduled``.
                                pending_sources: list[str] = []
                                if msg_text and msg_text.strip():
                                    pending_sources.append(msg_text)
                                pending_sources.append(full_response)
                                for assistant_final in pending_sources:
                                    if not str(assistant_final).strip():
                                        continue
                                    pending = _pending_tts_after_scheduled(
                                        self._turn_tts_scheduled_text,
                                        assistant_final,
                                    )
                                    if pending.strip():
                                        logger.debug(
                                            "tts_pending_final_tail",
                                            pending_chars=len(pending),
                                            scheduled_chars=len(self._turn_tts_scheduled_text),
                                            **_trace_debug_fields(),
                                        )
                                        turn_phase = "tts_synthesize_pending_final_tail"
                                        self._schedule_tts(pending)
                                        break
                            # Drain streaming TTS before the outer Agent OUTPUT (``anext``),
                            # so all PCM is queued before post-hook / trace save / finally.
                            if not self._cancel_turn.is_set():
                                turn_phase = "await_tts_after_llm_output"
                                await self._await_tts_chain()
                                # Attach provisional metrics: the outer Agent's trace is first
                                # persisted in its generator ``finally``, before this turn's own
                                # ``finally`` builds the final numbers. The finally re-saves the
                                # trace with final metrics; this keeps the intermediate snapshot
                                # meaningful if the process dies before then.
                                self._attach_metrics_to_trace(self._build_turn_metrics(user_text, interrupted=False))
            else:
                turn_phase = "agent_generator_exhausted"
                logger.debug(
                    "turn_agent_loop_complete",
                    reason="generator_exhausted",
                    response_chars=len(full_response),
                    **_trace_debug_fields(),
                )

            # Normally stamped at the .llm OUTPUT event (before the TTS drain);
            # fall back here for runs that never produced one.
            if self._turn_llm_done_at is None:
                self._turn_llm_done_at = time.monotonic()

            if text_buffer and not self._cancel_turn.is_set():
                turn_phase = "tts_synthesize_tail_buffer"
                self._schedule_tts(text_buffer)

            if not self._cancel_turn.is_set():
                turn_phase = "await_tts_before_done"
                await self._await_tts_chain()
                turn_phase = "emit_agent_text_done"
                if full_response.strip():
                    self._transcript.append(TranscriptEntry(role="assistant", text=full_response))
                await self._emit(AgentTextDone(text=full_response))
                self._turn_finalized_ok = True
                logger.debug(
                    "turn_agent_text_done_emitted",
                    response_chars=len(full_response),
                    **_trace_debug_fields(),
                )

        except asyncio.CancelledError:
            logger.debug(
                "turn_cancelled_error",
                turn_phase_at_cancel=turn_phase,
                response_chars=len(full_response),
                **_trace_debug_fields(),
            )
            turn_phase = f"cancelled_during_{turn_phase}"
            raise
        except Exception as e:
            turn_phase = "exception"
            logger.error("turn_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Turn failed: {e}"))
        finally:
            # Cancel-hard: ``interrupt()`` may cancel this task while finally is
            # awaiting TTS gather / metrics emit. Those awaits must not skip
            # truncation, ``_last_run_context`` update, or trace re-save.
            logger.debug(
                "turn_finally",
                turn_phase=turn_phase,
                will_aclose=agen is not None,
                response_chars=len(full_response),
                **_trace_debug_fields(),
            )
            try:
                # Normal turn already awaited the chain; cancel only if we bailed before
                # ``AgentTextDone`` (interrupt / error) and tasks may still be running.
                self._tts_tail = None
                if turn_phase != "emit_agent_text_done":
                    for t in list(self._tts_tasks):
                        if not t.done():
                            t.cancel()
                    if self._tts_tasks:
                        try:
                            await asyncio.shield(
                                asyncio.gather(*self._tts_tasks, return_exceptions=True)
                            )
                        except asyncio.CancelledError:
                            pass
                    try:
                        await asyncio.shield(self._abort_tts_stream())
                    except asyncio.CancelledError:
                        pass
                self._tts_tasks.clear()
                # Metrics: finalize and emit exactly once per turn attempt (interrupted included).
                # Runs before ``agen.aclose()`` so interrupted turns still get final metrics
                # attached to the trace before it is persisted.
                interrupted = self._cancel_turn.is_set() or turn_phase.startswith("cancelled_during")
                try:
                    from .metrics import TurnMetricsEvent

                    turn_metrics = self._build_turn_metrics(user_text, interrupted=interrupted)
                    self._metrics.append(turn_metrics)
                    self._attach_metrics_to_trace(turn_metrics)
                    await self._emit(TurnMetricsEvent(metrics=turn_metrics))
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug("turn_metrics_failed", error=str(e))
                # Close the Agent async generator *before* mutating session state so nested
                # ``voice_live.llm`` spans flush OUTPUT while the run context is still valid.
                # ``task.cancel()`` does not trigger ``async for``'s implicit ``aclose()``.
                #
                # STT often commits again (segment end / refinement) while we're still in
                # ``await _speak`` — ``interrupt()`` cancels this task.  ``CancelledError`` is
                # not a subclass of ``Exception``, so ``suppress(Exception)`` alone lets
                # ``aclose()`` abort before Runnable ``finally`` runs → no OUTPUT in logs.
                if agen is not None:
                    logger.debug("turn_agent_aclose_begin", **_trace_debug_fields())
                    try:
                        await asyncio.shield(agen.aclose())
                    except asyncio.CancelledError:
                        logger.debug(
                            "turn_agent_aclose_swallowed_cancelled_error",
                            **_trace_debug_fields(),
                        )
                    except Exception as e:
                        logger.debug(
                            "turn_agent_aclose_error",
                            error=str(e),
                            **_trace_debug_fields(),
                        )
                    else:
                        logger.debug("turn_agent_aclose_ok", **_trace_debug_fields())

                # Interruption truncation runs after aclose so the salvaged partial
                # assistant message is already in the root span's memory.
                #
                # ``replace_last``: the barge-in can land while this turn awaits the
                # final TTS chain — ``_await_tts_chain`` swallows the cancellation of
                # the TTS tasks, so the turn resumes normally and commits the *full*
                # reply (transcript + AgentTextDone) with ``_cancel_turn`` already
                # set. In that case the committed entry must be rewritten in place;
                # appending would leave the full entry plus a duplicate heard-prefix
                # entry (and ``interrupt()`` skips its own rewrite once
                # ``_last_interruption_heard_text`` is set here).
                if interrupted:
                    try:
                        self._apply_interruption_truncation(
                            ctx=get_run_context(),
                            replace_last=self._turn_finalized_ok,
                        )
                    except Exception as e:
                        logger.warning("turn_truncation_failed", error=str(e), exc_info=True)
                ctx = get_run_context()
                if ctx is not None:
                    self._last_run_context = ctx
                    # Re-persist the trace: the agent's own generator saved it on
                    # exhaustion, *before* this finally attached the final metrics.
                    # Live-object providers (in-memory) see the mutation anyway, but
                    # serializing providers (jsonl, platform) captured the provisional
                    # snapshot — without this re-save, stored voice_turn_metrics stay
                    # incomplete (e.g. null llm_total_ms). Providers already receive
                    # one put() per span completion and keep the latest snapshot per
                    # run id, so an extra save is consistent with their contract.
                    try:
                        await asyncio.shield(ctx._save_trace())
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        logger.debug("turn_trace_resave_failed", error=str(e))
            finally:
                self._is_speaking = False
                self._active_turn_user_text = ""

    # TODO(tool-filler): reintroduce tool-call filler speech here. Design that
    # worked (see git history / test_voice_filler.py): a `_maybe_speak_filler`
    # that fires at most once per turn, only when nothing has been spoken yet
    # (no TTS scheduled this turn, no audio still playing); the phrase is
    # transcript-recorded with `filler=True` but never enters `full_response` /
    # `_turn_assistant_text` (so no memory leak), and its TTS segment is
    # recorded with empty text so barge-in truncation counts its bytes without
    # attributing words the agent never said. Needs a `filler=True` synthesis
    # path that bypasses `_turn_tts_scheduled_text` (not reply text — must not
    # affect delta/OUTPUT reconciliation).

    # -- Internal: TTS ------------------------------------------------------

    def _schedule_tts(self, text: str) -> None:
        """Queue ``text`` for serial synthesis without blocking the agent event loop.

        If we ``await`` synthesis inside ``async for event in agen``, the LLM cannot
        yield its final ``OutputEvent`` until TTS finishes — trace ``OUTPUT`` lines and
        nested span teardown are then delayed until interrupt or turn end.
        """
        if not text:
            return
        self._turn_tts_scheduled_text += text
        # First flush of the turn: providers with a streaming context (ElevenLabs
        # multi-context WS) get ONE context per reply — every subsequent flush is
        # *fed* into it instead of synthesized as an independent segment. Separate
        # segments each get final-sentence intonation and an audible seam between
        # them ("choppy" speech); one context keeps prosody continuous.
        if self._turn_tts_stream is None and self._turn_tts_pump is None and not self._cancel_turn.is_set():
            stream = self.tts.open_stream()
            if stream is not None:
                self._turn_tts_stream = stream
                record: list = ["", 0]
                self._turn_stream_record = record
                self._turn_tts_segment_records.append(record)
                if self._turn_tts_started_at is None:
                    self._turn_tts_started_at = time.monotonic()
                self._turn_tts_pump = asyncio.create_task(self._pump_tts_stream(stream, record))
        stream = self._turn_tts_stream
        record = self._turn_stream_record
        prev = self._tts_tail

        async def chain() -> None:
            if prev is not None:
                try:
                    await prev
                except (asyncio.CancelledError, Exception):
                    pass
            if self._cancel_turn.is_set():
                return
            if stream is not None:
                clean = _strip_markdown(text)
                if not clean.strip():
                    return
                self._turn_tts_segments += 1
                if record is not None:
                    record[0] += clean
                try:
                    await stream.feed(clean)
                except Exception as e:
                    logger.error("tts_feed_error", error=str(e), text_preview=clean[:80], exc_info=True)
                    await self._emit(SessionError(message=f"TTS failed: {e}"))
                return
            await self._speak(text)

        t = asyncio.create_task(chain())
        self._tts_tasks.add(t)
        t.add_done_callback(lambda _t: self._tts_tasks.discard(_t))
        self._tts_tail = t

    async def _await_tts_chain(self) -> None:
        """Wait for all segments scheduled with :meth:`_schedule_tts` for this turn.

        Streaming mode: after the last feed, close the context (``end``) and
        drain the pump until the provider signals the reply is fully
        synthesized.
        """
        if self._tts_tail is not None and not self._tts_tail.done():
            try:
                await self._tts_tail
            except (asyncio.CancelledError, Exception):
                pass
        self._tts_tail = None
        stream = self._turn_tts_stream
        if stream is not None:
            try:
                await stream.end()
            except Exception as e:
                logger.debug("tts_stream_end_failed", error=str(e))
        pump = self._turn_tts_pump
        if pump is not None and not pump.done():
            # Shielded: a barge-in cancels this turn task while we drain — the
            # pump itself is stopped via interrupt() → _abort_tts_stream(),
            # and this method (like the chain await above) swallows the
            # cancellation so the turn's normal finalize path runs.
            try:
                await asyncio.wait_for(asyncio.shield(pump), timeout=30.0)
            except TimeoutError:
                logger.warning("tts_stream_drain_timeout")
                await self._abort_tts_stream()
            except (asyncio.CancelledError, Exception):
                pass
        self._turn_tts_stream = None
        self._turn_tts_pump = None

    async def _pump_tts_stream(self, stream: TTSStream, record: list) -> None:
        """Drain one streaming TTS context; mirrors ``_speak``'s accounting."""
        chunk_count = 0
        total_bytes = 0
        try:
            async for chunk in stream.audio():
                if self._cancel_turn.is_set():
                    # INFO on purpose: explains truncated audio_bytes in metrics.
                    logger.info(
                        "turn_tts_stream_break",
                        chunks_sent=chunk_count,
                        audio_bytes=total_bytes,
                        cancel_turn_set=True,
                        **_trace_debug_fields(),
                    )
                    break
                chunk_count += 1
                total_bytes += len(chunk)
                if self._turn_first_audio_at is None:
                    self._turn_first_audio_at = time.monotonic()
                self._turn_audio_bytes += len(chunk)
                if self._record_audio:
                    self._output_audio_chunks.append(chunk)
                record[1] += len(chunk)
                await self._emit(AudioOutput(data=chunk))
                self.playback.on_audio_emitted(len(chunk))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("tts_error", error=str(e), chunks_so_far=chunk_count, exc_info=True)
            await self._emit(SessionError(message=f"TTS failed: {e}"))
        else:
            # INFO on purpose: chars-in vs bytes-out is the only signal that
            # catches a TTS provider returning truncated audio.
            logger.info(
                "turn_tts_stream_end",
                text_chars=len(record[0]),
                text_preview=record[0][:120],
                audio_chunks=chunk_count,
                audio_bytes=total_bytes,
                cancel_turn_set=self._cancel_turn.is_set(),
                **_trace_debug_fields(),
            )
        finally:
            self._turn_tts_ended_at = time.monotonic()

    async def _abort_tts_stream(self) -> None:
        """Barge-in / teardown: stop the streaming context and its pump task."""
        stream = self._turn_tts_stream
        pump = self._turn_tts_pump
        self._turn_tts_stream = None
        self._turn_tts_pump = None
        if stream is not None:
            try:
                await stream.abort()
            except Exception as e:
                logger.debug("tts_stream_abort_failed", error=str(e))
        if pump is not None and not pump.done():
            pump.cancel()
            try:
                await pump
            except (asyncio.CancelledError, Exception):
                pass

    async def _speak(self, text: str) -> None:
        text = _strip_markdown(text)
        logger.debug(
            "turn_tts_synthesize_begin",
            text_chars=len(text),
            text_preview=text[:120],
            cancel_turn_set=self._cancel_turn.is_set(),
            **_trace_debug_fields(),
        )
        chunk_count = 0
        total_bytes = 0
        if self._turn_tts_started_at is None:
            self._turn_tts_started_at = time.monotonic()
        self._turn_tts_segments += 1
        segment_record: list = [text, 0]
        self._turn_tts_segment_records.append(segment_record)
        try:
            async for chunk in self.tts.synthesize(text):
                if self._cancel_turn.is_set():
                    # INFO on purpose: explains truncated audio_bytes in metrics.
                    logger.info(
                        "turn_tts_synthesize_break",
                        chunks_sent=chunk_count,
                        audio_bytes=total_bytes,
                        cancel_turn_set=True,
                        **_trace_debug_fields(),
                    )
                    break
                chunk_count += 1
                total_bytes += len(chunk)
                if self._turn_first_audio_at is None:
                    self._turn_first_audio_at = time.monotonic()
                self._turn_audio_bytes += len(chunk)
                if self._record_audio:
                    self._output_audio_chunks.append(chunk)
                segment_record[1] += len(chunk)
                await self._emit(AudioOutput(data=chunk))
                self.playback.on_audio_emitted(len(chunk))
        except Exception as e:
            logger.error(
                "tts_error",
                error=str(e),
                text_preview=text[:120],
                chunks_so_far=chunk_count,
                exc_info=True,
            )
            await self._emit(SessionError(message=f"TTS failed: {e}"))
        else:
            # INFO on purpose: chars-in vs bytes-out is the only signal that
            # catches a TTS provider returning truncated audio (is_final too
            # early) — at ~16kHz PCM expect very roughly >1.5KB per char.
            logger.info(
                "turn_tts_synthesize_end",
                text_chars=len(text),
                text_preview=text[:120],
                audio_chunks=chunk_count,
                audio_bytes=total_bytes,
                cancel_turn_set=self._cancel_turn.is_set(),
                **_trace_debug_fields(),
            )
        finally:
            self._turn_tts_ended_at = time.monotonic()

    # -- Internal: interruption truncation ------------------------------------

    async def _align_continue_memory(self, *, fragment_user_text: str) -> None:
        """Mirror CONTINUE_TURN transcript cleanup onto parent-chain memory.

        ``interrupt()`` truncation writes ``assistant:heard`` into
        ``_last_run_context`` memory. Transcript pops that entry before merging
        the user line; without the same cleanup here the next turn still
        "remembers" answering the fragment.

        The fragment *user* entry is popped too (not rewritten to the combined
        text): the merged turn passes the combined text as its prompt, which
        the agent appends to this memory — rewriting left the combined
        utterance in the LLM input twice.
        """
        if not fragment_user_text:
            return
        ctx = self._last_run_context
        if ctx is None:
            return
        root = ctx.root_span()
        if root is None or not isinstance(root.memory, list) or len(root.memory) < 1:
            return
        last = root.memory[-1]
        if getattr(last, "role", None) == "assistant":
            prev = root.memory[-2] if len(root.memory) >= 2 else None
            if prev is not None and getattr(prev, "role", None) == "user":
                if (prev.collect_text() or "") == fragment_user_text:
                    root.memory.pop()
                    last = root.memory[-1] if root.memory else None
        if last is not None and getattr(last, "role", None) == "user":
            if (last.collect_text() or "") == fragment_user_text:
                root.memory.pop()
        try:
            await asyncio.shield(ctx._save_trace())
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug("continue_memory_resave_failed", error=str(e))

    def _apply_interruption_truncation(self, ctx: RunContext | None, *, replace_last: bool = False) -> None:
        """Align transcript and agent memory with what the user actually *heard*.

        On barge-in the LLM has usually generated (and we may have synthesized)
        more text than was played. The trace keeps the full generation for
        observability, but:

        * the session transcript gets an assistant entry with the heard prefix
          (previously interrupted turns recorded no assistant text at all), and
        * the assistant message in the run's memory — what the next turn
          resolves as conversation history — is truncated to the heard prefix,
          or dropped entirely if nothing was played. Without this the agent
          "remembers saying" things that were never spoken.

        Two call sites: the cancelled turn's ``finally`` (mid-generation
        barge-in, appends the heard prefix) and ``interrupt()`` when the turn
        already completed but buffered audio was still playing
        (``replace_last=True`` — the full reply is already committed, so the
        transcript entry is rewritten in place).
        """
        heard_bytes = self._turn_heard_bytes
        if heard_bytes is None:
            heard_bytes = max(0, self.playback.played_bytes - self._turn_played_baseline)
        segments = [(str(t), int(b)) for t, b in self._turn_tts_segment_records]
        heard_text = map_played_bytes_to_text(segments, heard_bytes)
        # "" means "nothing was heard"; None (never set) means "unknown".
        self._last_interruption_heard_text = heard_text
        # INFO on purpose: this rewrite decides what the transcript/memory keep
        # of an interrupted reply (heard_chars == 0 erases it entirely).
        logger.info(
            "turn_interruption_truncation",
            heard_bytes=heard_bytes,
            heard_chars=len(heard_text),
            generated_chars=len(self._turn_assistant_text),
            tts_segments=len(segments),
            replace_last=replace_last,
            **_trace_debug_fields(),
        )

        if replace_last and self._transcript and self._transcript[-1].role == "assistant":
            if heard_text:
                self._transcript[-1] = TranscriptEntry(role="assistant", text=heard_text)
            else:
                self._transcript.pop()
        elif heard_text:
            self._transcript.append(TranscriptEntry(role="assistant", text=heard_text))

        if ctx is None:
            return
        root = ctx.root_span()
        if root is None or not isinstance(root.memory, list) or not root.memory:
            return
        last = root.memory[-1]
        if getattr(last, "role", None) != "assistant":
            return
        # Only touch the message if it is this turn's reply — guard against
        # mangling an earlier assistant/tool_use message.
        last_text = last.collect_text()
        turn_text = self._turn_assistant_text
        if not last_text or not turn_text:
            return
        if not (last_text.startswith(turn_text) or turn_text.startswith(last_text)):
            return
        non_text_content = [c for c in last.content if not isinstance(c, TextContent)]
        if heard_text:
            last.content = [TextContent(text=heard_text), *non_text_content]
        elif non_text_content:
            last.content = non_text_content
        else:
            root.memory.pop()

    # -- Internal: metrics ----------------------------------------------------

    def _build_turn_metrics(self, user_text: str, *, interrupted: bool) -> TurnMetrics:
        """Compute :class:`TurnMetrics` from the current turn's monotonic stamps."""
        from .metrics import TurnMetrics

        def _ms(t0: float | None, t1: float | None) -> float | None:
            if t0 is None or t1 is None:
                return None
            return round((t1 - t0) * 1000, 1)

        eou = self._turn_eou_at or None
        eou_to_first_audio = _ms(eou, self._turn_first_audio_at)
        return TurnMetrics(
            turn_index=self._turn_index,
            user_text_chars=len(user_text),
            eou_to_llm_first_token_ms=_ms(eou, self._turn_first_token_at),
            eou_to_tts_first_byte_ms=eou_to_first_audio,
            eou_to_first_audio_ms=eou_to_first_audio,
            llm_total_ms=_ms(self._turn_started_at, self._turn_llm_done_at),
            tts_total_ms=_ms(self._turn_tts_started_at, self._turn_tts_ended_at),
            turn_total_ms=_ms(self._turn_started_at, time.monotonic()) or 0.0,
            interrupted=interrupted,
            tts_segments=self._turn_tts_segments,
            audio_bytes=self._turn_audio_bytes,
            playback_acks_received=self.playback.ack_received,
            heard_bytes=self._turn_heard_bytes if interrupted else None,
            vad_endpointed=self._turn_vad_endpointed,
        )

    def _attach_metrics_to_trace(self, metrics: TurnMetrics) -> None:
        """Best-effort: store turn metrics on the run's root span metadata."""
        ctx = get_run_context()
        if ctx is None:
            return
        try:
            root = ctx.root_span()
            if root is not None:
                root.metadata["voice_turn_metrics"] = metrics.model_dump()
        except Exception as e:
            logger.debug("turn_metrics_trace_attach_failed", error=str(e))

    # -- Internal: helpers --------------------------------------------------

    async def _emit(self, event: VoiceSessionEvent | None) -> None:
        await self._event_queue.put(event)

    async def _cleanup(self) -> None:
        self._cancel_hold()
        self._held_user_text = None
        if self._llm_warmup_task is not None and not self._llm_warmup_task.done():
            self._llm_warmup_task.cancel()
        self._llm_warmup_task = None
        if self._endpointer is not None:
            await self._endpointer.close()
            self._endpointer = None
        for t in list(self._tts_tasks):
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks.clear()
        self._tts_tail = None
        await self._abort_tts_stream()
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.stt.close()
        await self.tts.close()
        await self.turn_detector.close()


def _reconcile_final_assistant_text(streamed: str, final_text: str) -> tuple[str, str | None]:
    """If the model's final ``Message`` extends streamed deltas, return (canonical, suffix).

    Some providers stream only a prefix then put the full reply on the terminal
    ``OutputEvent``; without this, the UI and TTS stop at the streamed prefix.

    If a leading ``Text`` block was skipped and only tail deltas were processed,
    ``streamed`` can equal ``final_text``'s suffix — recover the missing prefix.
    """
    if not final_text:
        return streamed, None
    if not streamed:
        return final_text, final_text
    if final_text.startswith(streamed) and len(final_text) > len(streamed):
        return final_text, final_text[len(streamed) :]
    end = _nfc_aligned_prefix_end(final_text, streamed)
    if end is not None and end < len(final_text):
        return final_text, final_text[end:]
    # Tail-only stream (e.g. missed first ``Text`` item): require non-trivial overlap.
    if len(final_text) > len(streamed) >= 12 and final_text.endswith(streamed):
        prefix = final_text[: -len(streamed)]
        if prefix.strip():
            return final_text, prefix
    return streamed, None


def _last_sentence_boundary_end(text: str) -> int | None:
    """Index in *text* right after the last ``SENTENCE_BOUNDARY`` match, or ``None``."""
    end = None
    for m in SENTENCE_BOUNDARY.finditer(text):
        end = m.end()
    return end


def _flush_segment(text: str, *, first_segment: bool, audio_playing: bool = False) -> str | None:
    """Return the prefix of *text* to send to TTS now, or ``None`` to keep buffering.

    When the buffer contains a sentence boundary in the *middle* (e.g. ``mucho. ¿Hay…``),
    we split there so each TTS segment sees complete sentences.  ElevenLabs produces much
    better prosody with full clauses — feeding ``"…te pueda"`` as a segment causes the
    classic trailing-elongation ("puedaaaa") because TTS thinks the sentence ended.

    When *audio_playing* is True the browser still has audio queued, so there is no rush
    to produce the next chunk.  We keep buffering until ``MAX_TTS_BUFFER_CHARS`` to give
    ElevenLabs the largest possible context for natural prosody.
    """
    if not text:
        return None
    stripped = text.rstrip()
    if not stripped:
        return None
    ends_clause = stripped[-1] in _CLAUSE_END_CHARS

    # Hard cap — always flush at MAX to avoid unbounded buffering.
    if len(text) >= MAX_TTS_BUFFER_CHARS:
        if first_segment:
            for m in SENTENCE_BOUNDARY.finditer(text):
                if m.end() >= FIRST_SEGMENT_MIN_CHARS:
                    return text[: m.end()]
        last = _last_sentence_boundary_end(text)
        return text[:last] if last is not None else text

    # While audio is still playing, keep buffering — no need to rush.
    if audio_playing:
        return None

    lo_clause = FIRST_SEGMENT_MIN_CHARS if first_segment else 2
    if ends_clause and len(stripped) >= lo_clause:
        return text

    lo = FIRST_SEGMENT_MIN_CHARS if first_segment else MIN_FLUSH_CHARS
    if len(text) >= lo:
        if first_segment:
            for m in SENTENCE_BOUNDARY.finditer(text):
                if m.end() >= lo:
                    return text[: m.end()]
        else:
            last = _last_sentence_boundary_end(text)
            if last is not None:
                return text[:last]
    return None
