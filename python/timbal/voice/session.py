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
from difflib import SequenceMatcher
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field

from ..core.agent import Agent
from ..state import get_run_context, set_run_context
from ..state.context import RunContext
from ..types.content import TextContent
from ..types.events import OutputEvent
from ..types.events.delta import DeltaEvent, Text, TextDelta
from ..types.message import Message

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


class TextToSpeech(ABC):
    """Abstract TTS provider.

    Lifecycle: ``connect`` → ``synthesize`` (repeatable) → ``close``.
    """

    @abstractmethod
    async def connect(self, config: AudioOutputConfig) -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

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
        record_audio: bool = False,
    ):
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.audio_input = audio_input or AudioInputConfig()
        self.audio_output = audio_output or AudioOutputConfig()

        self._event_queue: asyncio.Queue[VoiceSessionEvent | None] = asyncio.Queue()
        self._cancel_turn = asyncio.Event()
        self._current_turn_task: asyncio.Task | None = None
        self._is_speaking = False
        self._closed = False

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

        # Serial TTS runs off the agent ``async for`` critical path so we keep pulling
        # LLM/Agent events (and emit trace OUTPUT) while audio still synthesizes.
        self._tts_tail: asyncio.Task | None = None
        self._tts_tasks: set[asyncio.Task] = set()
        # Concatenation of all strings passed to ``_schedule_tts`` this turn (OUTPUT tail catch-up).
        self._turn_tts_scheduled_text: str = ""

        # Estimated wall-clock time when the browser finishes playing the queued
        # audio.  Mirrors the browser's ``nextPlayTime`` gapless scheduling so we
        # can allow barge-in even after ``_is_speaking`` goes False (the agent
        # turn completed but buffered audio is still playing).
        self._playback_end_estimate: float = 0.0

        # -- Session recording --------------------------------------------------
        self._transcript: list[TranscriptEntry] = []
        self._record_audio = record_audio
        self._input_audio_chunks: list[bytes] = []
        self._output_audio_chunks: list[bytes] = []

    # -- Public: session recording ------------------------------------------

    @property
    def transcript(self) -> list[TranscriptEntry]:
        """Ordered transcript of committed user/assistant text for this session."""
        return list(self._transcript)

    @property
    def input_audio(self) -> bytes:
        """Raw PCM of mic input (empty when ``record_audio=False``)."""
        return b"".join(self._input_audio_chunks)

    @property
    def output_audio(self) -> bytes:
        """Raw PCM of TTS output (empty when ``record_audio=False``)."""
        return b"".join(self._output_audio_chunks)

    # -- Playback tracking --------------------------------------------------

    _PCM_BYTES_PER_SEC = 32_000  # PCM16 mono 16 kHz

    @property
    def _assistant_audio_playing(self) -> bool:
        """True if the browser likely still has queued audio to play."""
        return time.monotonic() < self._playback_end_estimate

    @property
    def _assistant_active(self) -> bool:
        """True if the agent is generating OR audio is still playing in the browser."""
        return self._is_speaking or self._assistant_audio_playing

    # -- Public API ---------------------------------------------------------

    async def run(self, audio_in: AsyncIterable[bytes]) -> AsyncIterator[VoiceSessionEvent]:
        """Main loop.  Yields events until the session is closed or errors out."""
        try:
            await self.stt.connect(self.audio_input)
            await self.tts.connect(self.audio_output)
            await self._emit(SessionStarted())

            audio_task = asyncio.create_task(self._forward_audio(audio_in))
            stt_task = asyncio.create_task(self._process_stt_events())

            try:
                while True:
                    event = await self._event_queue.get()
                    if event is None:  # sentinel → stop
                        break
                    yield event
            finally:
                for task in (audio_task, stt_task):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(audio_task, stt_task, return_exceptions=True)

        except Exception as e:
            logger.error("voice_session_error", error=str(e), exc_info=True)
            yield SessionError(message=str(e))
        finally:
            await self._cleanup()
            yield SessionEnded()

    async def interrupt(self) -> None:
        """Cancel TTS playback and the current agent turn."""
        if not self._assistant_active:
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
            **_trace_debug_fields(),
        )
        self._is_speaking = False
        self._playback_end_estimate = 0.0
        self._cancel_turn.set()
        for t in list(self._tts_tasks):
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks.clear()
        self._tts_tail = None
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._emit(SessionInterrupted())
        logger.debug("session_interrupt_emitted", **_trace_debug_fields())

    async def close(self) -> None:
        """Gracefully shut down the session."""
        if self._closed:
            return
        self._closed = True
        await self.interrupt()
        await self._emit(None)  # sentinel stops the run() iterator

    # -- Internal: audio → STT ---------------------------------------------

    async def _forward_audio(self, audio_in: AsyncIterable[bytes]) -> None:
        try:
            async for chunk in audio_in:
                if self._record_audio:
                    self._input_audio_chunks.append(chunk)
                await self.stt.push_audio(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("audio_forward_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Audio input error: {e}"))

    # -- Internal: STT → turns ---------------------------------------------

    async def _process_stt_events(self) -> None:
        try:
            async for event in self.stt.events():
                if event.type == "partial":
                    text = event.text.strip()
                    self._partials_since_last_commit += 1
                    await self._emit(TranscriptPartial(text=text))
                    is_playing = self._assistant_audio_playing
                    if is_playing and text:
                        is_noise = _is_noise(text)
                        is_echo = _likely_stt_echo(text, self._turn_assistant_text) if not is_noise else False
                        too_short = len(text) < 4
                        if not is_noise and not is_echo and not too_short:
                            logger.debug(
                                "stt_partial_barge_in",
                                text_preview=text[:80],
                                assistant_chars=len(self._turn_assistant_text),
                                audio_playing=is_playing,
                            )
                            await self.interrupt()
                        else:
                            logger.debug(
                                "stt_partial_skipped",
                                text_preview=text[:80],
                                too_short=too_short,
                                is_noise=is_noise,
                                is_echo=is_echo,
                                audio_playing=is_playing,
                            )
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

    async def _handle_committed(self, text: str) -> None:
        partials_seen = self._partials_since_last_commit
        self._partials_since_last_commit = 0
        logger.debug(
            "stt_committed_received",
            text_preview=text[:160],
            text_len=len(text),
            partials_before=partials_seen,
            is_speaking=self._is_speaking,
            audio_playing=self._assistant_audio_playing,
            active_user_preview=(self._active_turn_user_text[:100] if self._active_turn_user_text else ""),
            assistant_so_far_chars=len(self._turn_assistant_text),
            **_trace_debug_fields(),
        )
        if _is_noise(text):
            logger.debug("stt_noise_ignored", text=text)
            return
        if _is_garbage_commit(text):
            logger.debug("stt_garbage_commit_ignored", text=text)
            return
        if partials_seen == 0 and len(text) > 40 and not self._assistant_active:
            logger.debug("stt_hallucination_ignored", text=text[:160], text_len=len(text))
            return
        if self._assistant_active and _likely_stt_echo(text, self._turn_assistant_text):
            logger.debug("stt_echo_ignored", text=text)
            return
        if self._assistant_active and self._active_turn_user_text:
            if _is_same_user_utterance_refinement(self._active_turn_user_text, text):
                logger.debug(
                    "stt_committed_refinement_ignored",
                    active=self._active_turn_user_text[:100],
                    new=text[:100],
                )
                return
            # Very soon after turn start, VAD often double-fires near-identical commits.
            if time.monotonic() - self._turn_started_at < 1.5:
                a, b = _normalize_echo(self._active_turn_user_text), _normalize_echo(text)
                if min(len(a), len(b)) >= 5 and SequenceMatcher(None, a, b).ratio() >= 0.58:
                    logger.debug(
                        "stt_committed_early_duplicate_ignored",
                        active=self._active_turn_user_text[:100],
                        new=text[:100],
                    )
                    return
            # VAD split a single utterance into two fast commits (e.g. "Hola, ¿qué tal" + "estás?").
            # Combine and restart the turn instead of treating the fragment as a new query.
            since_last_commit = time.monotonic() - self._last_commit_at
            if since_last_commit < 3.0 and len(text) < 30:
                combined = self._active_turn_user_text.rstrip(", ") + " " + text
                logger.debug(
                    "stt_committed_continuation",
                    prev=self._active_turn_user_text[:100],
                    new=text[:100],
                    combined=combined[:160],
                    since_last_commit=round(since_last_commit, 2),
                )
                await self.interrupt()
                self._cancel_turn.clear()
                self._last_commit_at = time.monotonic()
                # Replace the previous user entry with the combined text.
                if self._transcript and self._transcript[-1].role == "user":
                    self._transcript[-1] = TranscriptEntry(role="user", text=combined)
                else:
                    self._transcript.append(TranscriptEntry(role="user", text=combined))
                await self._emit(TranscriptCommitted(text=combined))
                self._active_turn_user_text = combined
                self._current_turn_task = asyncio.create_task(self._run_turn(combined))
                return
        logger.debug(
            "stt_committed_accepted_new_turn",
            had_active_speech=self._is_speaking,
            text_preview=text[:160],
            **_trace_debug_fields(),
        )
        await self.interrupt()
        self._cancel_turn.clear()
        self._last_commit_at = time.monotonic()
        self._transcript.append(TranscriptEntry(role="user", text=text))
        await self._emit(TranscriptCommitted(text=text))
        self._active_turn_user_text = text
        self._current_turn_task = asyncio.create_task(self._run_turn(text))
        logger.debug(
            "stt_turn_task_created",
            text_preview=text[:120],
            **_trace_debug_fields(),
        )

    # -- Internal: agent turn → TTS ----------------------------------------

    async def _run_turn(self, user_text: str) -> None:
        self._is_speaking = True
        self._turn_assistant_text = ""
        self._turn_started_at = time.monotonic()
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
            else:
                turn_phase = "agent_generator_exhausted"
                logger.debug(
                    "turn_agent_loop_complete",
                    reason="generator_exhausted",
                    response_chars=len(full_response),
                    **_trace_debug_fields(),
                )

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
            logger.debug(
                "turn_finally",
                turn_phase=turn_phase,
                will_aclose=agen is not None,
                response_chars=len(full_response),
                **_trace_debug_fields(),
            )
            # Normal turn already awaited the chain; cancel only if we bailed before
            # ``AgentTextDone`` (interrupt / error) and tasks may still be running.
            self._tts_tail = None
            if turn_phase != "emit_agent_text_done":
                for t in list(self._tts_tasks):
                    if not t.done():
                        t.cancel()
                if self._tts_tasks:
                    await asyncio.gather(*self._tts_tasks, return_exceptions=True)
            self._tts_tasks.clear()
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

            ctx = get_run_context()
            if ctx is not None:
                self._last_run_context = ctx
            self._is_speaking = False
            self._active_turn_user_text = ""

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
        prev = self._tts_tail

        async def chain() -> None:
            if prev is not None:
                try:
                    await prev
                except (asyncio.CancelledError, Exception):
                    pass
            if self._cancel_turn.is_set():
                return
            await self._speak(text)

        t = asyncio.create_task(chain())
        self._tts_tasks.add(t)
        t.add_done_callback(lambda _t: self._tts_tasks.discard(_t))
        self._tts_tail = t

    async def _await_tts_chain(self) -> None:
        """Wait for all segments scheduled with :meth:`_schedule_tts` for this turn."""
        if self._tts_tail is not None and not self._tts_tail.done():
            try:
                await self._tts_tail
            except (asyncio.CancelledError, Exception):
                pass
        self._tts_tail = None

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
        try:
            async for chunk in self.tts.synthesize(text):
                if self._cancel_turn.is_set():
                    logger.debug(
                        "turn_tts_synthesize_break",
                        chunks_sent=chunk_count,
                        audio_bytes=total_bytes,
                        cancel_turn_set=True,
                        **_trace_debug_fields(),
                    )
                    break
                chunk_count += 1
                total_bytes += len(chunk)
                if self._record_audio:
                    self._output_audio_chunks.append(chunk)
                await self._emit(AudioOutput(data=chunk))
                now = time.monotonic()
                play_start = max(self._playback_end_estimate, now)
                self._playback_end_estimate = play_start + len(chunk) / self._PCM_BYTES_PER_SEC
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
            logger.debug(
                "turn_tts_synthesize_end",
                text_chars=len(text),
                text_preview=text[:120],
                audio_chunks=chunk_count,
                audio_bytes=total_bytes,
                cancel_turn_set=self._cancel_turn.is_set(),
                **_trace_debug_fields(),
            )

    # -- Internal: helpers --------------------------------------------------

    async def _emit(self, event: VoiceSessionEvent | None) -> None:
        await self._event_queue.put(event)

    async def _cleanup(self) -> None:
        for t in list(self._tts_tasks):
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks.clear()
        self._tts_tail = None
        if self._current_turn_task and not self._current_turn_task.done():
            self._current_turn_task.cancel()
            try:
                await self._current_turn_task
            except (asyncio.CancelledError, Exception):
                pass
        await self.stt.close()
        await self.tts.close()


_NOISE_PATTERN = re.compile(r"^\s*\(.*\)\s*$")
# Word + closing paren only, no leading "(" — e.g. Scribe hallucinating "(music playing)" as "Music)"
_STT_ORPHAN_CLOSE_CAPTION = re.compile(r"^[A-Za-z][A-Za-z\s'-]{0,48}\)\s*$")
# Opens "(" but no ")" on this commit (split caption or noise)
_STT_INCOMPLETE_OPEN_PAREN = re.compile(r"^\([^)]*$")


def _is_noise(text: str) -> bool:
    """Filter STT artifacts like '(wind blowing)', '(silence)', '(music)', etc."""
    return bool(_NOISE_PATTERN.match(text))


def _is_garbage_commit(text: str) -> bool:
    """Filter silence/noise hallucinations: lone punctuation, broken captions, etc.

    Realtime models often emit parenthetical *sound* labels; on silence you can get
    fragments like ``(`` or ``Music)`` without the user speaking.
    """
    t = text.strip()
    if not t:
        return True
    # Single non-alphanumeric (e.g. "(" from a split "(music)")
    if len(t) == 1 and not t.isalnum():
        return True
    # Tiny all-punctuation runs
    if len(t) <= 3 and all(not c.isalnum() for c in t):
        return True
    if t in {")", "]", "}", "…"}:
        return True
    if _STT_ORPHAN_CLOSE_CAPTION.match(t):
        return True
    if _STT_INCOMPLETE_OPEN_PAREN.match(t):
        return True
    return False


def _normalize_echo(s: str) -> str:
    return " ".join(s.lower().split())


def _likely_stt_echo(committed: str, assistant_so_far: str) -> bool:
    """True if STT text is probably the assistant's own speech leaking into the mic."""
    c = _normalize_echo(committed)
    a = _normalize_echo(assistant_so_far)
    if not a:
        return False
    # Short leaks (punctuation / tail of TTS) often transcribe as 1–9 chars.
    if len(c) < 10:
        if len(c) >= 2 and c in a:
            return True
        return False
    if c in a:
        return True
    tail_len = min(len(a), max(len(c) * 3, 100))
    tail = a[-tail_len:]
    return SequenceMatcher(None, c, tail).ratio() >= 0.68


def _is_same_user_utterance_refinement(active: str, new: str) -> bool:
    """True if ``new`` is a longer/corrected transcript of the same user phrase.

    Realtime STT commonly emits multiple ``committed`` events per pause; the
    second is usually an extension, not intentional barge-in.
    """
    if active.strip() == new.strip():
        return True
    a, b = _normalize_echo(active), _normalize_echo(new)
    if not a or not b:
        return False
    if a == b:
        return True
    r = SequenceMatcher(None, a, b).ratio()
    # Near-duplicate punctuation / spacing (possibly seconds apart).
    if min(len(a), len(b)) >= 6 and r >= 0.84:
        return True
    if len(b) > len(a):
        if (len(a) >= 5 and b.startswith(a)) or (len(a) >= 8 and a in b) or r >= 0.74:
            return True
    elif len(a) > len(b) and len(b) >= 8:
        # Second pass shorter but same gist (rare).
        if a.startswith(b) or r >= 0.86:
            return True
    return False


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
