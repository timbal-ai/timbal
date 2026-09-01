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
from collections.abc import AsyncIterable, AsyncIterator
from typing import TYPE_CHECKING, Any

import structlog
from uuid_extensions import uuid7

from ..core.agent import Agent
from ..state import get_run_context, set_run_context
from ..state.context import RunContext
from ..state.tracing.providers import TRACING_UNSET
from ..types.content import TextContent, ToolUseContent
from ..types.events import ApprovalEvent, InteractionEvent, OutputEvent
from ..types.events.delta import DeltaEvent, Text, TextDelta, ToolUse
from ..types.message import Message
from .config import (
    DEFAULT_GREETING_SYSTEM_PROMPT,
    FillerConfig,
    GreetingConfig,
    coerce_greeting,
)
from .events import (
    AgentApproval,
    AgentInteraction,
    AgentStatus,
    AgentTextDelta,
    AgentTextDone,
    AudioOutput,
    FillerSpoken,
    SessionEnded,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    TranscriptCommitted,
    TranscriptEntry,
    TranscriptPartial,
    VoiceSessionEvent,
)
from .metrics import TurnMetrics, TurnMetricsEvent
from .playback import BufferedPlaybackTracker, PlaybackTracker, map_played_bytes_to_text
from .providers import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TTSStream,
)
from .turn_detection import (
    CommitAction,
    PartialDecision,
    TurnDetector,
    TurnState,
    _is_same_user_utterance_refinement,
    _likely_stt_echo,
    resolve_turn_detector,
)

if TYPE_CHECKING:
    from .endpointing import VadEndpointer
    from .recording import CallRecorder

logger = structlog.get_logger("timbal.voice.session")

# Wall-clock cap on a single agent turn (LLM stream + in-turn TTS drains).
# Without this a hung provider leaves the caller in silence forever.
DEFAULT_TURN_TIMEOUT_SECS = 60.0
# Spoken when the turn times out before any audio was produced. Empty / None
# disables the apology; the session stays open either way.
DEFAULT_TURN_TIMEOUT_FALLBACK = "Sorry, I lost that for a moment. Could you say that again?"


async def _no_audio_score() -> float | None:
    """Audio-EOU scorer for detectors that have none — always abstains.

    Bound instead of the detector's ``score_recent_audio`` so the endpointer
    takes its documented "no audio score" path and sizes the delay from the
    text EOU, rather than the session having to special-case the binding.
    """
    return None


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
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?\n;:\u061f\u061b])\s+")
# Minimum chars before flushing to TTS.  Bigger segments = better prosody (ElevenLabs
# multi-context has no cross-context continuity, so each segment's intonation is
# independent).  Too small → choppy "final" intonation at every boundary.
# When audio is already playing, _flush_segment skips these thresholds entirely and
# buffers up to MAX_TTS_BUFFER_CHARS for maximum prosody quality.
MIN_FLUSH_CHARS = 24
FIRST_SEGMENT_MIN_CHARS = 6
MAX_TTS_BUFFER_CHARS = 200

# Clause-ending chars for flush heuristics (ASCII + common Spanish + fullwidth variants +
# Arabic ؟/؛ — Arabic uses the Latin period for full stops, so "." already covers those).
_CLAUSE_END_CHARS = frozenset(".!?;\n:\uff1f\uff01\u061f\u061b")


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

    Turn timeout
    ------------
    Each agent turn is capped by ``turn_timeout_secs`` (default 60s). On
    expiry the session emits :class:`SessionError`, speaks
    ``turn_timeout_fallback`` when no audio has gone out yet, and stays open
    for the next user turn. Set ``turn_timeout_secs <= 0`` to disable.

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
        recorder: CallRecorder | None = None,
        session_id: str | None = None,
        hold_timeout_secs: float = 1.5,
        turn_timeout_secs: float = DEFAULT_TURN_TIMEOUT_SECS,
        turn_timeout_fallback: str | None = DEFAULT_TURN_TIMEOUT_FALLBACK,
        vad_endpointing: bool | VadEndpointer | None = None,
        model: str | None = None,
        filler: FillerConfig | dict[str, Any] | None = None,
        greeting: GreetingConfig | dict[str, Any] | str | None = None,
        call_context: dict[str, str] | None = None,
        parent_run_id: str | None = None,
    ):
        self.agent = agent
        self.stt = stt
        self.tts = tts
        # Optional per-session LLM override (playground model picker). Passed
        # through to ``agent(prompt=..., model=...)`` — does not mutate the Agent.
        self.model = model.strip() if isinstance(model, str) and model.strip() else None
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
        # Wall-clock cap for one agent turn. ``<= 0`` disables. On expiry the
        # session emits SessionError, speaks ``turn_timeout_fallback`` if no
        # audio has gone out yet, and stays open for the next user turn.
        self.turn_timeout_secs = turn_timeout_secs
        self.turn_timeout_fallback = turn_timeout_fallback
        self._turn_deadline: float | None = None
        # VAD endpointing fast path (Silero speech-stop → audio EOU → stt.commit()).
        # None = auto: on when the detector exposes an audio EOU model and the
        # timbal[voice] extra is installed; False = off; True = on (warn when
        # unavailable); a VadEndpointer instance = use as-is (custom knobs).
        self._vad_endpointing = vad_endpointing
        self._endpointer: VadEndpointer | None = None
        # Whether Silero's speech history may be used as *evidence* about the
        # user (barge-in vetoes, hold extension), as opposed to merely driving
        # the commit fast path. False when the endpointer armed on a text EOU
        # alone: those three behaviours were written for detectors that have
        # always had an audio model, and switching them on as a side effect of
        # a latency fix would change barge-in and hold behaviour under the
        # guise of committing sooner.
        self._vad_evidence = False
        # time.monotonic() of the last endpointer-forced stt.commit(); the next
        # committed transcript within a short window is attributed to it.
        self._endpoint_commit_sent_at: float | None = None
        self._turn_vad_endpointed = False
        self._llm_warmup_task: asyncio.Task[None] | None = None
        # Tool-call filler: an LLM-generated phrase masks tool dead air (see
        # FillerConfig). The generator Agent is built lazily on first use.
        self.filler = FillerConfig.model_validate(filler) if isinstance(filler, dict) else filler
        self._filler_agent: Agent | None = None
        self._turn_filler_task: asyncio.Task[None] | None = None
        self._turn_filler_count = 0
        self._turn_filler_text = ""
        # Segments/bytes as of the last filler — "spoken since then" checks.
        self._turn_filler_segments_baseline = 0
        self._turn_filler_audio_baseline = 0

        # Agent-speaks-first opener (see GreetingConfig). Spoken from run() at
        # t=0 instead of from a turn, so it lives entirely outside the turn
        # machinery: no RunContext, no agent memory (there is no prior run to
        # chain onto yet) — the first turn is told about it through a
        # system-prompt line instead. See _greeting_flow.
        self.greeting = coerce_greeting(greeting)
        self._greeting_agent: Agent | None = None
        self._greeting_task: asyncio.Task[None] | None = None
        self._greeting_text = ""
        # Set while the opener is *synthesizing*. Barge-in stays gated past
        # this — until the audio has actually drained — see
        # _greeting_holds_interrupt.
        self._greeting_speaking = False
        # The opener's speak task. Held separately from ``_tts_tail`` because a
        # reply cancelled behind the opener nulls the tail in its ``finally``,
        # and the next reply must still chain behind the opener rather than
        # interleave with it.
        self._greeting_speak_task: asyncio.Task[None] | None = None
        # The opener's ``_speak`` segment record (text, emitted_bytes), kept
        # live so a barge-in can map played bytes back to heard words and the
        # first turn can account for audio still queued ahead of its reply.
        self._greeting_record: list | None = None
        # The opener's transcript entry, held by identity so a barge-in can
        # rewrite it to the heard prefix after later entries have piled on top.
        self._greeting_entry: TranscriptEntry | None = None

        # Telephony identity (rep_id, task, from/to/call_id). Not VoiceConfig —
        # leftover custom params that must survive turn-2 context forks via
        # the session bag. Do not pass these as agent(prompt=..., rep_id=...);
        # leftover kwargs leak into ``_llm._stream(**kwargs)``.
        self.call_context: dict[str, str] = dict(call_context or {})

        # Run id this call continues from (text → voice). Session identity, not
        # a voice knob: it says which conversation the caller is joining, so it
        # must come from whoever authorized the call (the dial / boot env), not
        # the browser hello. Applies to turn 1 only — from turn 2 on the
        # session's own ``_last_run_context`` chaining takes over unchanged.
        self.parent_run_id = parent_run_id.strip() if isinstance(parent_run_id, str) and parent_run_id.strip() else None

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
        # Last time the assistant was observed speaking, for the echo grace window.
        # Sampled rather than hooked because `_assistant_active` is derived from
        # playback draining, which has no event to hang a callback on. Every partial
        # and commit samples it, and echo is exactly what produces partials, so the
        # reading is dense when it matters. A stale sample can only over-state the
        # elapsed time and close the window early, which is the safe direction.
        self._last_assistant_active_at: float = 0.0

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
        # Persistent call recording (MP3 + manifest; see voice/recording.py).
        # Distinct from the in-memory record_audio seam above.
        self.session_id = session_id or uuid7(as_type="hex")
        self._recorder = recorder
        #: Wall-clock session start (set when run() begins); transcript offsets
        #: in the recording manifest and session_transcript are relative to it.
        self.started_at: float | None = None
        #: Resolved identity (model, stt, transport, ...) a transport may attach
        #: for the recording manifest.
        self.recording_meta: dict[str, Any] | None = None

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

    async def submit_user_text(
        self,
        text: str,
        *,
        interaction_id: str | None = None,
        run_id: str | None = None,
    ) -> bool:
        """Start a user turn as if STT committed ``text``.

        The LiveKit / WS ``interaction_answer`` path (tap an ``ask_user``
        option) lands here so it shares interrupt + turn-begin with speech.
        A runnable that implements ``answer_interaction`` can bind the
        value to the exact parked id; otherwise the text is the utterance.

        Returns whether a turn was started.
        """
        text = (text or "").strip()
        if self._closed or not text:
            return False
        if self.started_at is None:
            logger.info("interaction_answer_dropped", reason="session_not_started")
            return False
        answer = getattr(self.agent, "answer_interaction", None)
        if callable(answer) and interaction_id:
            answer(interaction_id=interaction_id, value=text, run_id=run_id)
        await self.interrupt()
        if self._closed:
            return False
        self._cancel_turn.clear()
        await self._begin_user_turn(text, replace_user_entry=False)
        return True

    async def run(self, audio_in: AsyncIterable[bytes]) -> AsyncIterator[VoiceSessionEvent]:
        """Main loop.  Yields events until the session is closed or errors out."""
        try:
            self.started_at = time.time()
            self._start_llm_warmup()
            await self.stt.connect(self.audio_input)
            await self.tts.connect(self.audio_output)
            await self.turn_detector.start(self.audio_input)
            await self._maybe_start_endpointer()
            await self._emit(SessionStarted())
            # Seed before the greeting task so turn 1 (and a callable
            # system_prompt resolved for the opener) sees identity and the
            # parent run this call continues. Empty ``_trace`` is what lets
            # runnable.py reuse this context.
            await self._seed_call_context()
            # A task, not an await: the opener's TTS (and, for the LLM-authored
            # variant, its completion) must overlap the reader tasks below —
            # otherwise nothing is consuming mic audio while we speak.
            self._maybe_start_greeting()

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
        if self._greeting_holds_interrupt():
            # INFO on purpose: the caller *did* barge in and heard the agent
            # keep talking anyway — that only looks like a bug without this line.
            logger.info(
                "session_interrupt_skipped",
                reason="greeting_not_interruptible",
                audio_playing=self._assistant_audio_playing,
                **_trace_debug_fields(),
            )
            # The opener is spared, but a reply already running behind it is
            # not: leaving it alive would let the *next* commit start a second
            # turn on top of it.
            await self._interrupt_turn_behind_greeting()
            return
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
            if self._recorder is not None:
                # The recording truncates exactly like the transcript does: the
                # unheard tail of this turn never reaches the file. The recorder
                # clamps to its queue, so an estimate that lags the drain only
                # over-keeps a fraction of a second.
                self._recorder.drop_agent_tail(
                    max(0, self._turn_audio_bytes - self._turn_heard_bytes)
                )
        # Same rule as the reply's post-completion rewrite below: a real barge-in
        # rewrites what was left unheard, ``close()`` leaves the transcript alone.
        if was_active and truncate_completed and self._greeting_record is not None:
            self._truncate_greeting()
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
        # Long replies synthesize faster than the WS can drain: dozens of
        # AudioOutput frames sit in ``_event_queue``. Drop them *before*
        # SessionInterrupted so the client is not still scheduling backlog PCM
        # for seconds after a barge-in.
        dropped_audio = self._drop_queued_audio_output() if was_active else 0
        if was_active and self._turn_finalized_ok and truncate_completed and self._last_interruption_heard_text is None:
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
                dropped_queued_audio=dropped_audio,
                **_trace_debug_fields(),
            )

    async def close(self) -> None:
        """Gracefully shut down the session."""
        if self._closed:
            return
        self._closed = True
        self._cancel_hold()
        self._held_user_text = None
        # A non-interruptible opener is deliberately outside ``_tts_tasks``, so
        # ``interrupt()`` cannot reach it. The caller has hung up — cut it here
        # (cancelling the flow propagates to its speak task).
        if self._greeting_task is not None and not self._greeting_task.done():
            self._greeting_task.cancel()
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
        # Prefer the per-session override (playground model picker) — turns use
        # ``self.model`` the same way. Warming ``agent.model`` alone misses the
        # provider the first reply will actually hit.
        model = self.model or getattr(self.agent, "model", None)
        if not (isinstance(model, str) and "/" in model):
            return

        from ..core.llm import warmup_llm_connection

        self._llm_warmup_task = asyncio.create_task(warmup_llm_connection(model))

    # -- Internal: VAD endpointing fast path ---------------------------------

    async def _maybe_start_endpointer(self) -> None:
        """Arm the local VAD endpointing loop when the pieces exist.

        Needs the ``timbal[voice]`` extra for Silero, plus *some* EOU signal to
        size the delay with: an audio EOU (``score_recent_audio`` with a
        non-None ``audio_eou``), or failing that a text EOU. Silently stays off
        otherwise (warning when explicitly requested). See
        :mod:`timbal.voice.endpointing`.

        Arming without an audio EOU exists because the fast path's job — telling
        the STT to finalize — has nothing to do with *how* the turn end was
        decided, while the coupling meant the configuration that needs it most
        could never have it. A text-only detector on an STT that does not
        endpoint aggressively waits on the provider: measured on
        deepgram-nova/lexical, a spoken account number sat 7.5s from speech end
        to commit, against 750ms for the same audio under ``local``.

        The VAD *evidence* behaviours stay off in that mode — see
        :attr:`_vad_evidence`. Silero being loaded is not a reason to start
        vetoing barge-ins on a detector that has never done so.
        """
        spec = self._vad_endpointing
        if spec is False:
            return
        explicit = spec is not None
        score_fn = getattr(self.turn_detector, "score_recent_audio", None)
        audio_eou = getattr(self.turn_detector, "audio_eou", None)
        has_audio_eou = score_fn is not None and audio_eou is not None
        if not has_audio_eou and not self._has_text_eou():
            if explicit:
                logger.warning(
                    "vad_endpointing_unavailable",
                    reason="turn detector exposes neither an audio nor a text EOU",
                    detector=type(self.turn_detector).__name__,
                )
            return
        if spec is None or spec is True:
            from .endpointing import VadEndpointer

            endpointer = VadEndpointer()
        else:
            endpointer = spec
        endpointer.bind(
            # No audio EOU: the endpointer falls back to text_score to size the
            # delay rather than skipping the commit entirely.
            score=score_fn if has_audio_eou else _no_audio_score,
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
        self._vad_evidence = has_audio_eou
        logger.info(
            "vad_endpointing_active",
            stop_silence_secs=endpointer.stop_silence_secs,
            max_delay_secs=endpointer.max_delay_secs,
            driver="audio_eou" if has_audio_eou else "text_eou",
        )

    def _has_text_eou(self) -> bool:
        """Whether the detector can score a partial's completeness."""
        return any(
            getattr(self.turn_detector, name, None) is not None
            for name in ("effective_text_eou", "fallback_text_eou", "text_eou")
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

        Prefers :meth:`~timbal.voice.LocalAudioTurnDetector.effective_text_eou`
        (Namo blended with lexical) so finished questions don't inflate the
        incomplete-text delay when the model under-scores. Falls back to raw
        ``fallback_text_eou`` / punctuation baseline. Returns ``None`` when
        there is no fresh partial — endpointer keeps the audio-only delay.
        """
        if self._last_partial_at <= self._last_commit_at or not self._latest_partial_text:
            return None
        effective = getattr(self.turn_detector, "effective_text_eou", None)
        if callable(effective):
            return await effective(self._latest_partial_text)
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
        if self._endpointer is None or not self._vad_evidence:
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
        if self._endpointer is None or not self._vad_evidence:
            return False
        speech_secs = self._endpointer.speech_secs_in_window(self.BARGE_IN_VAD_WINDOW_SECS)
        return speech_secs is not None and speech_secs < self.MIN_BARGE_IN_VAD_SPEECH_SECS

    def _vad_confirms_speech_since(self, since_monotonic: float) -> bool:
        """True when Silero saw real speech *after* ``since_monotonic``.

        HOLD expiry used to call :meth:`_vad_contradicts_recent_partial`, whose
        2s lookback still contains the utterance that armed the hold — so any
        late STT refinement after commit "confirmed" speech and floored every
        short text-complete HOLD at the 2s grace window (live: armed 0.35s,
        expired ~2.0s). Missing VAD → ``False`` (don't stretch; a real resume
        still supersedes via COMMIT). No endpointer → ``True`` so unit tests
        without Silero keep the partial-extends-hold behavior.
        """
        if self._endpointer is None or not self._vad_evidence:
            return True
        window = time.monotonic() - since_monotonic
        if window <= 0:
            return False
        window = min(window, self.BARGE_IN_VAD_WINDOW_SECS)
        speech_secs = self._endpointer.speech_secs_in_window(window)
        return speech_secs is not None and speech_secs >= self.MIN_BARGE_IN_VAD_SPEECH_SECS

    # Trailing window for "the user is still talking right now". Long enough to
    # bridge Silero's between-phoneme dips, short enough that a HOLD fires
    # promptly once they really stop.
    HOLD_VAD_SPEECH_WINDOW_SECS = 0.5
    MIN_HOLD_VAD_SPEECH_SECS = 0.1
    # Ceiling on how long mic energy alone may defer one HOLD. Echo surviving an
    # imperfect canceller carries energy, so without a cap the assistant's own
    # playback could hold a turn open for as long as it speaks.
    HOLD_VAD_MAX_EXTENSION_SECS = 3.0

    def _vad_hears_speech_now(self, since_monotonic: float) -> bool:
        """Silero saw speech in the last :attr:`HOLD_VAD_SPEECH_WINDOW_SECS`,
        counting only after ``since_monotonic``.

        Two departures from the neighbouring VAD gates, both because this one
        *triggers* a hold extension instead of corroborating an STT partial:

        Missing evidence returns ``False``, not ``True`` — a permissive default
        would defer every hold to its cap on any session without Silero.

        It does not require :attr:`_vad_evidence` (an audio EOU model). That
        flag guards behaviours that can *suppress* something the user did, like
        vetoing a barge-in, where being wrong makes the assistant
        uninterruptible. Declining to end a turn while the mic is live only
        delays it, by at most :attr:`HOLD_VAD_MAX_EXTENSION_SECS`, and the
        user's own commit supersedes the hold either way — and the text-only
        detectors that lack an audio EOU are exactly the ones whose STT splits
        utterances soonest.

        The anchor matters only for holds shorter than the window: no shipped
        tier is (1.2s is the shortest, and the provider's own silence threshold
        precedes it), but without it a 0.2s hold would count the held
        utterance's own tail as the user resuming and extend on itself.
        """
        if self._endpointer is None:
            return False
        window = min(self.HOLD_VAD_SPEECH_WINDOW_SECS, time.monotonic() - since_monotonic)
        if window <= 0:
            return False
        speech_secs = self._endpointer.speech_secs_in_window(window)
        return speech_secs is not None and speech_secs >= self.MIN_HOLD_VAD_SPEECH_SECS

    # Silero speech inside a window that still counts as quiet. A stray frame or
    # two is a blip on breath or a click rather than the user resuming, and
    # demanding exactly zero would let one of them keep a stranded transcript
    # hostage for as long as the session lives.
    MAX_QUIET_SPEECH_SECS = 0.1

    # Never look further back than the endpointer keeps speech history for, or a
    # window longer than the buffer would read as silence and convict a real turn.
    MAX_HALLUCINATION_LOOKBACK_SECS = 3.0

    def _mic_speech_since_last_commit(self) -> bool | None:
        """Whether Silero heard the user speak since the previous commit.

        ``None`` means no opinion, and the caller must not suppress on it: no
        endpointer, no commit to measure from, or a gap longer than the speech
        history, where "no speech in the buffer" says nothing about the utterance
        that produced this commit.

        Anchored on the last commit rather than a trailing window because a fixed
        window cannot work here. ElevenLabs commits roughly 1.6s after the audio, so
        any window wide enough not to convict a real late commit is also wide enough
        to still contain the *previous* utterance and acquit an invented one. The gap
        since the last commit is the interval the new text must account for.
        """
        if self._endpointer is None or not self._last_commit_at:
            return None
        window = time.monotonic() - self._last_commit_at
        if window <= 0 or window > self.MAX_HALLUCINATION_LOOKBACK_SECS:
            return None
        speech_secs = self._endpointer.speech_secs_in_window(window)
        if speech_secs is None:
            return None
        return speech_secs >= self.MIN_BARGE_IN_VAD_SPEECH_SECS

    # Trailing window Silero must find quiet before a partial can be called invented.
    # Shorter than the barge-in window because this fires between utterances, where
    # the pause is the signal, not during one.
    HALLUCINATION_QUIET_WINDOW_SECS = 0.5

    def _partial_clobbers_stranded(self, text: str) -> bool:
        """A partial that is neither the stranded utterance refined nor backed by mic energy.

        ElevenLabs invents stock phrases on trailing silence — ``"Yeah."``, ``"Yes."``,
        ``"I don't know."`` — and each one costs twice over. It overwrites the real
        utterance the watchdog would have rescued, and it refreshes the staleness anchor
        so the watchdog stays disarmed by the very churn it exists to catch. The
        provider's own VAD silence timer restarts too, so the real words never commit by
        either route: measured on `medical_barge_in`, the interrupting turn was never
        committed by anyone.

        Deliberately narrow, because silence alone does not make a partial invented. The
        watchdog exists for the user who speaks quietly under playback, where an
        over-eager AEC ducks the mic and *neither* the provider VAD nor Silero registers
        an utterance — refusing every uncorroborated partial would break the case the
        thing was built for. So all three must hold: a stranded partial is already
        waiting, the new text is not that one refined, and Silero heard nothing recent.

        The detector still sees the text; only the rescue payload and its anchor are
        protected. Deciding whether to interrupt on it is
        :meth:`_vad_vetoes_barge_in`'s job, and it applies a stricter standard.
        """
        if not self._latest_partial_text or self._last_partial_at <= self._last_commit_at:
            return False
        if _is_same_user_utterance_refinement(self._latest_partial_text, text):
            return False
        return self._mic_quiet_for(self.HALLUCINATION_QUIET_WINDOW_SECS)

    def _mic_quiet_for(self, secs: float) -> bool:
        """Silero heard essentially nothing in the trailing ``secs``.

        Missing evidence returns ``False`` — no Silero means no opinion, and the
        caller falls back to transcript staleness.

        Not gated on :attr:`_vad_evidence`, for the reason
        :meth:`_vad_hears_speech_now` is not: that flag guards inferences which
        can *suppress* something the user did. This one only rescues speech that
        would otherwise be lost outright.
        """
        if self._endpointer is None:
            return False
        speech_secs = self._endpointer.speech_secs_in_window(secs)
        return speech_secs is not None and speech_secs <= self.MAX_QUIET_SPEECH_SECS

    # -- Internal: audio → STT ---------------------------------------------

    async def _forward_audio(self, audio_in: AsyncIterable[bytes]) -> None:
        try:
            async for chunk in audio_in:
                if self._record_audio:
                    self._input_audio_chunks.append(chunk)
                if self._recorder is not None:
                    self._recorder.add_mic(chunk)
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
        ``_handle_committed`` path.

        A *newer partial* is the wrong thing to wait on by itself. ElevenLabs
        hallucinates on trailing silence, and each invented partial both restarts
        its own VAD silence timer — so it never commits — and refreshes this
        watchdog's anchor, so the safety net stays disarmed by the very churn it
        exists to catch. Measured on `medical_barge_in`: partials 1.0-1.2s apart
        (``"Sorry, one more thing."``, ``"Yeah."``, ``"Yeah."``) against a 2.5s
        threshold, and the interrupting turn was never committed at all. So mic
        silence counts too: if the user demonstrably stopped speaking that long
        ago, the provider clearly will not commit, whatever it is still emitting.

        Providers whose ``commit()`` is a no-op (Deepgram Flux) never answer
        that nudge. After a short grace for Finalize-capable providers, we
        synthesize a committed event from the stranded partial text so the
        caption cannot hang forever.
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
                mic_quiet = self._mic_quiet_for(self._stale_partial_commit_secs)
                if stale_secs < self._stale_partial_commit_secs and not mic_quiet:
                    continue
                # Never rescue the assistant's own voice. An IGNOREd commit does
                # not bump _last_commit_at, so suppressed echo still looks like a
                # stranded partial — and this path is where it comes back to life:
                # traced under --aec-leak, "memory access." was refused as echo
                # while the reply played, then synthesized into a turn of its own
                # three seconds later. Nothing downstream catches it, because by
                # then the detector's echo grace window has closed.
                #
                # This rescue exists for user speech an over-eager AEC ducked
                # below the provider's commit threshold, which is the one thing
                # echo is not.
                # Verbatim only, deliberately: this runs outside the detector and so
                # outside its leak latch, and rescuing is the last chance a stranded
                # utterance gets. The traced case was an exact substring anyway.
                if _likely_stt_echo(self._latest_partial_text, self._spoken_assistant_text()):
                    logger.debug("stt_stale_partial_echo_skipped", text_preview=self._latest_partial_text[:80])
                    continue
                self._stale_commit_sent_at = time.monotonic()
                stranded = self._latest_partial_text
                # INFO on purpose: this is the only trace that a transcript was
                # rescued from a provider that silently refused to commit.
                logger.info("stt_stale_partial_commit", stale_secs=round(stale_secs, 1), mic_quiet=mic_quiet)
                try:
                    await self.stt.commit()
                except Exception as e:
                    logger.warning("stt_stale_partial_commit_failed", error=str(e))
                # Give Finalize-capable providers a beat to emit committed.
                await asyncio.sleep(0.4)
                if self._closed or not stranded:
                    continue
                # The text holding still across the grace is load-bearing, and not
                # merely a guard against racing a mid-flight commit: it is the
                # only thing separating real speech from provider churn. Dropping
                # it (so a partial that mutated inside the 400ms still synthesized)
                # was measured at 4/12 on the ElevenLabs barge-ins versus 7/12
                # with it, and synthesized a hallucinated ``"Yeah."`` as a turn of
                # its own. A stranded turn is better than an invented one.
                if self._latest_partial_text == stranded and self._last_partial_at > self._last_commit_at:
                    logger.info(
                        "stt_stale_partial_synthesized",
                        text_preview=stranded[:80],
                    )
                    await self._handle_committed(stranded)
        except asyncio.CancelledError:
            return

    # -- Internal: STT → turns ---------------------------------------------

    async def _process_stt_events(self) -> None:
        try:
            async for event in self.stt.events():
                if event.type == "partial":
                    text = event.text.strip()
                    self._partials_since_last_commit += 1
                    if text and self._partial_clobbers_stranded(text):
                        # INFO on purpose: this is the only trace that a real
                        # utterance was kept alive against provider churn.
                        logger.info(
                            "stt_partial_hallucination_ignored",
                            text_preview=text[:80],
                            stranded_preview=self._latest_partial_text[:80],
                        )
                    elif text:
                        self._last_partial_at = time.monotonic()
                        self._latest_partial_text = text
                    decision = await self.turn_detector.on_partial(text, self._turn_state())
                    if decision is PartialDecision.BARGE_IN and self._vad_vetoes_barge_in(text):
                        # Hallucinated multi-word partials during TTS (no mic
                        # energy) — do not flash them in the playground caption.
                        # (_vad_vetoes_barge_in already logs stt_partial_barge_in_vetoed.)
                        decision = PartialDecision.IGNORE
                    else:
                        await self._emit(TranscriptPartial(text=text))
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
        assistant_active = self._assistant_active
        if assistant_active:
            self._last_assistant_active_at = now
        return TurnState(
            assistant_active=assistant_active,
            audio_playing=self._assistant_audio_playing,
            assistant_text=self._spoken_assistant_text(),
            active_user_text=active,
            seconds_since_turn_start=now - self._turn_started_at,
            seconds_since_last_commit=now - self._last_commit_at,
            seconds_since_assistant_active=(
                None if assistant_active or not self._last_assistant_active_at else now - self._last_assistant_active_at
            ),
            mic_speech_since_last_commit=self._mic_speech_since_last_commit(),
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
                vad_extended_secs = 0.0
                while True:
                    await asyncio.sleep(remaining)
                    # Never fire mid-utterance: a *new* STT partial since the
                    # commit means the user resumed speaking, and their commit
                    # is about to merge with / supersede this hold. Require
                    # post-commit mic energy — not "any speech in the last 2s",
                    # which still includes the held utterance itself and used
                    # to stretch every short HOLD out to the grace window.
                    since_partial = time.monotonic() - self._last_partial_at
                    if (
                        self._last_partial_at > anchor
                        and since_partial < self._hold_partial_grace_secs
                        and self._vad_confirms_speech_since(anchor)
                    ):
                        remaining = self._hold_partial_grace_secs - since_partial
                        logger.debug(
                            "stt_hold_extended",
                            remaining=round(remaining, 3),
                            timeout_secs=timeout_secs,
                        )
                        continue
                    # Mic says the user is still going but the STT has not caught
                    # up. Waiting for a partial loses this race whenever the
                    # provider commits on a short silence: measured on
                    # ElevenLabs at a 0.3s VAD threshold, an interior pause
                    # committed the fragment and the next one's audio was in
                    # flight ~1.3s before any partial existed to extend on, so
                    # a 1.2s hold fired mid-sentence and split the utterance.
                    if vad_extended_secs < self.HOLD_VAD_MAX_EXTENSION_SECS and self._vad_hears_speech_now(anchor):
                        remaining = min(
                            self.HOLD_VAD_SPEECH_WINDOW_SECS,
                            self.HOLD_VAD_MAX_EXTENSION_SECS - vad_extended_secs,
                        )
                        vad_extended_secs += remaining
                        logger.debug(
                            "stt_hold_extended_vad",
                            remaining=round(remaining, 3),
                            extended_secs=round(vad_extended_secs, 3),
                            timeout_secs=timeout_secs,
                        )
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
                # A HOLD exists because the fragment looked incomplete. If it is
                # still just a dangling token ("I", "the", "and") when the timer
                # fires, promoting it to a user turn invents ghost replies
                # (live: force-committed "I" → "The capital of France is Paris.").
                from .eou import _DANGLING_TOKENS, _WORD_RE

                words = _WORD_RE.findall(held)
                if len(words) == 1 and words[0].lower() in _DANGLING_TOKENS:
                    logger.info(
                        "stt_hold_expired_dropped",
                        text_preview=held[:120],
                        timeout_secs=timeout_secs,
                        reason="dangling_token",
                        **_trace_debug_fields(),
                    )
                    return
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
            replace_user_entry = False
        await self._emit(TranscriptCommitted(text=final_text, replace=replace_user_entry))
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
            self._endpoint_commit_sent_at is not None and time.monotonic() - self._endpoint_commit_sent_at < 2.0
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
        # Late twin of a commit we already accepted (Flux EndOfTurn after a
        # session-synthesized stale rescue, or provider double-final). The
        # active-turn refinement gate misses this once the reply has finished
        # and ``_active_turn_user_text`` is cleared — only then should this
        # fire. Mid-turn / HOLD commits that look like refinements
        # ("hello can" → "hello can you help…") must reach the detector so
        # CONTINUE_TURN / merge can run.
        if (
            not self._active_turn_user_text
            and self._held_user_text is None
            and self._transcript
            and self._transcript[-1].role == "user"
            and time.monotonic() - self._last_commit_at < 3.0
            and _is_same_user_utterance_refinement(self._transcript[-1].text, text)
        ):
            logger.info(
                "stt_commit_ignored",
                reason="late_duplicate",
                text_preview=text[:160],
            )
            self._partials_since_last_commit = 0
            return
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
            # HOLD = "not sure yet" — do NOT chop an audible reply for a
            # deferred fragment (echo-ish "Hello, hello." mid-TTS was wiping
            # greetings). Partials that meant barge-in already interrupted.
            # NEW_TURN / hold expiry / CONTINUE interrupt when the turn starts.
            if self._assistant_audio_playing:
                logger.info(
                    "stt_hold_defer_during_tts",
                    text_preview=final_text[:80],
                    reason=decision.reason,
                )
            else:
                await self.interrupt()
                self._cancel_turn.clear()
            # Detector returns the full utterance to hold (refine/merge already applied).
            timeout = decision.hold_timeout_secs if decision.hold_timeout_secs is not None else self.hold_timeout_secs
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

    async def _await_turn_deadline(self, awaitable: Any) -> Any:
        """Await ``awaitable``, raising :class:`TimeoutError` if the turn deadline elapses.

        ``turn_timeout_secs <= 0`` disables the deadline. Used for agent
        ``__anext__`` and in-turn TTS drains so a hung LLM/tool cannot leave
        the caller in silence indefinitely.

        Uses :func:`asyncio.timeout` (same-task) rather than
        :func:`asyncio.wait_for`. On Python 3.11 ``wait_for`` wraps the
        awaitable in a child Task, so ``set_run_context`` inside the agent
        generator is discarded when the wait returns — ``_last_run_context``
        stays ``None``, metrics never attach to the trace, and barge-in
        truncation cannot rewrite memory. 3.12+ rewrote ``wait_for`` onto
        ``timeout`` and hid the bug; keep the same-task form everywhere.
        """
        deadline = self._turn_deadline
        if deadline is None:
            return await awaitable
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("agent turn timed out")
        try:
            async with asyncio.timeout(remaining):
                return await awaitable
        except TimeoutError:
            # Re-raise as a bare TimeoutError so the turn's ``except TimeoutError``
            # path stays distinct from an inner CancelledError / BaseExceptionGroup
            # that ``asyncio.timeout`` can surface on some versions.
            raise TimeoutError("agent turn timed out") from None

    async def _run_turn(self, user_text: str) -> None:
        self._is_speaking = True
        self._turn_assistant_text = ""
        self._turn_started_at = time.monotonic()
        self._turn_deadline = (
            self._turn_started_at + self.turn_timeout_secs if self.turn_timeout_secs > 0 else None
        )
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
        # A greeting still draining when this turn starts (deferred barge-in)
        # sits ahead of the reply on the played axis. Seed a text-less segment
        # for it so interruption truncation walks past those bytes instead of
        # crediting the reply with audio the caller never reached.
        greeting_pending = self._greeting_pending_bytes()
        if greeting_pending:
            self._turn_tts_segment_records.append(["", greeting_pending])
        self._turn_heard_bytes = None
        self._turn_finalized_ok = False
        self._turn_tts_stream = None
        self._turn_tts_pump = None
        self._turn_stream_record = None
        self._turn_filler_task = None
        self._turn_filler_count = 0
        self._turn_filler_text = ""
        self._turn_filler_segments_baseline = 0
        self._turn_filler_audio_baseline = 0
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
                turn_timeout_secs=self.turn_timeout_secs,
                **_trace_debug_fields(),
            )
            text_buffer = ""
            self._turn_tts_scheduled_text = ""

            turn_phase = "creating_agent_generator"
            msg = Message(role="user", content=[TextContent(text=user_text)])
            agent_kwargs: dict[str, Any] = {"prompt": msg}
            if self.model:
                agent_kwargs["model"] = self.model
            # First turn after an opener. The opener was spoken outside any run,
            # so it is in no memory the agent can resolve — without telling it,
            # the model opens by greeting a caller it has already greeted. Turn
            # one only: from turn two on, memory chains through turn one and
            # carries the exchange that followed the greeting. Resolving the
            # agent's own prompt here is not extra work, only earlier work — the
            # override is exactly what ``Agent.handler`` would have resolved.
            #
            # What the caller *heard*, not what was configured: an interruptible
            # opener cut mid-sentence leaves a prefix, and one cut before a
            # single word landed leaves nothing to not-repeat.
            greeting_heard = self._greeting_heard_text()
            if greeting_heard and self._last_run_context is None:
                agent_kwargs["system_prompt"] = _dont_greet_again_prompt(
                    await self._agent_system_prompt(), greeting_heard
                )
            agen = self.agent(**agent_kwargs)
            # Timed ``__anext__`` so a hung LLM/tool cannot stall forever. The
            # for-body is unchanged; only the pull is deadline-aware.
            while True:
                try:
                    event = await self._await_turn_deadline(agen.__anext__())
                except StopAsyncIteration:
                    turn_phase = "agent_generator_exhausted"
                    logger.debug(
                        "turn_agent_loop_complete",
                        reason="generator_exhausted",
                        response_chars=len(full_response),
                        **_trace_debug_fields(),
                    )
                    break

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

                if isinstance(event, DeltaEvent) and isinstance(event.item, ToolUse) and event.item.name:
                    # Dead air while a tool runs — surface it so the playground
                    # doesn't look hung (live: get_datetime slept 3–5s with no UI).
                    await self._emit(AgentStatus(text=f"Calling {event.item.name}…"))
                    # Earliest tool-call signal for streaming providers.
                    self._maybe_schedule_filler(event.item.name)
                    continue

                # The run parked waiting for a value or a decision. Nothing else on
                # the wire says so — a caller just hears silence — so lift it into a
                # session event here, at the event, not after the generator drains:
                # end-of-turn would put it behind the span close-out, the terminal
                # OutputEvent and a trace save. The run still ends ``cancelled``
                # (reason ``input_required`` / ``approval_required``); resuming is an
                # HTTP call against ``run_id``, which is why it is carried.
                if isinstance(event, InteractionEvent):
                    turn_phase = "emit_agent_interaction"
                    await self._emit(
                        AgentInteraction(
                            run_id=event.run_id,
                            interaction_id=event.interaction_id,
                            kind=event.kind,
                            payload=event.payload or {},
                            response_schema=event.response_schema,
                            tool_call_id=event.tool_call_id,
                        )
                    )
                    continue

                if isinstance(event, ApprovalEvent):
                    turn_phase = "emit_agent_approval"
                    await self._emit(
                        AgentApproval(
                            run_id=event.run_id,
                            approval_id=event.approval_id,
                            kind=event.kind,
                            prompt=event.prompt,
                            ui=event.ui,
                            input=event.input,
                            input_schema=event.input_schema,
                            description=event.description,
                            tool_call_id=event.tool_call_id,
                        )
                    )
                    continue

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
                            # Non-streaming providers (and TestModel) surface
                            # tool calls only here, on the final Message — the
                            # second filler hook point, after the text handling
                            # above so spoken text this turn suppresses it.
                            # Also the backup UI status when there was no
                            # ToolUse delta.
                            for block in out.content:
                                if isinstance(block, ToolUseContent) and block.name:
                                    await self._emit(AgentStatus(text=f"Calling {block.name}…"))
                                    self._maybe_schedule_filler(block.name)
                                    break
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
                                await self._await_turn_deadline(self._await_tts_chain())
                                # Attach provisional metrics: the outer Agent's trace is first
                                # persisted in its generator ``finally``, before this turn's own
                                # ``finally`` builds the final numbers. The finally re-saves the
                                # trace with final metrics; this keeps the intermediate snapshot
                                # meaningful if the process dies before then.
                                self._attach_metrics_to_trace(self._build_turn_metrics(user_text, interrupted=False))

            # Normally stamped at the .llm OUTPUT event (before the TTS drain);
            # fall back here for runs that never produced one.
            if self._turn_llm_done_at is None:
                self._turn_llm_done_at = time.monotonic()

            if text_buffer and not self._cancel_turn.is_set():
                turn_phase = "tts_synthesize_tail_buffer"
                self._schedule_tts(text_buffer)

            if not self._cancel_turn.is_set():
                turn_phase = "await_tts_before_done"
                await self._await_turn_deadline(self._await_tts_chain())
                turn_phase = "emit_agent_text_done"
                if full_response.strip():
                    self._transcript.append(TranscriptEntry(role="assistant", text=full_response))
                # run_id makes this a current pointer to the conversation: pass
                # it as parent_id over HTTP to continue on another transport.
                await self._emit(AgentTextDone(text=full_response, run_id=self._turn_run_id()))
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
        except TimeoutError:
            # Hung LLM / tool / TTS drain hit the wall-clock cap. Surface an
            # error, speak a short apology if the caller heard nothing, and let
            # the session stay open for a retry — do not re-raise.
            turn_phase = "timeout"
            logger.error(
                "turn_timeout",
                timeout_secs=self.turn_timeout_secs,
                response_chars=len(full_response),
                had_audio=self._turn_first_audio_at is not None,
                **_trace_debug_fields(),
            )
            await self._emit(SessionError(message=f"Turn timed out after {self.turn_timeout_secs:g}s"))
            fallback = (self.turn_timeout_fallback or "").strip()
            if fallback and self._turn_first_audio_at is None and not self._cancel_turn.is_set():
                try:
                    await self._abort_tts_stream()
                    # The hung component may be the TTS itself, in which case an
                    # unbounded rescue would hang exactly like the turn it is
                    # rescuing. Bound it; on expiry we give up on the apology.
                    async with asyncio.timeout(min(self.turn_timeout_secs, 10.0)):
                        await self._speak(fallback)
                    self._transcript.append(TranscriptEntry(role="assistant", text=fallback))
                    # The apology is session-synthesized, but the timed-out run
                    # persisted in the generator's aclose — its id still names
                    # the thread. None if the run never started.
                    await self._emit(AgentTextDone(text=fallback, run_id=self._turn_run_id()))
                    self._turn_finalized_ok = True
                except Exception as e:
                    logger.warning("turn_timeout_fallback_failed", error=str(e), exc_info=True)
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
                # A filler that hasn't spoken yet is pointless once the turn is
                # over (reply done, interrupted, or errored) — cancel it before
                # the TTS teardown below.
                if self._turn_filler_task is not None and not self._turn_filler_task.done():
                    self._turn_filler_task.cancel()
                    try:
                        await asyncio.shield(asyncio.gather(self._turn_filler_task, return_exceptions=True))
                    except asyncio.CancelledError:
                        pass
                # Normal turn already awaited the chain; cancel only if we bailed before
                # ``AgentTextDone`` (interrupt / error) and tasks may still be running.
                self._tts_tail = None
                if turn_phase != "emit_agent_text_done":
                    for t in list(self._tts_tasks):
                        if not t.done():
                            t.cancel()
                    if self._tts_tasks:
                        try:
                            await asyncio.shield(asyncio.gather(*self._tts_tasks, return_exceptions=True))
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
                # An empty trace means no run happened this turn — the ambient
                # context is the pre-turn seed (call_context / parent_run_id)
                # after a pre-run timeout or cancel. Adopting it would make the
                # NEXT turn's run invisible to _turn_run_id: the agent reuses
                # the empty-trace seed, so the retry's context is identical to
                # _last_run_context and its genuine run would report None.
                if ctx is not None and ctx._trace:
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

    # -- Internal: tool-call filler ------------------------------------------

    def _spoken_assistant_text(self) -> str:
        """Everything the assistant said out loud this turn — filler included.

        Echo suppression must match against the filler phrase too (it plays
        through the same speaker); reconciliation and truncation keep using
        ``_turn_assistant_text``, which stays reply-only.

        The greeting joins it for as long as it can still be echoing: before
        the first turn, and afterwards while its audio is out on the wire. That
        second half is the case that matters — a deferred barge-in starts turn
        one *mid-opener*, and a window that closed on turn start would hand the
        speakerphone's echo of the rest of the opener back as a user utterance.

        Dropping it once the audio has drained is deliberate:
        ``_likely_stt_echo`` suppresses any verbatim substring, so a permanent
        entry would silently swallow a caller who later says a phrase the
        opener happened to contain ("your appointment").
        """
        spoken = self._turn_assistant_text
        if self._greeting_text and (
            self._turn_index == 0 or self._greeting_speaking or self._greeting_pending_bytes() > 0
        ):
            spoken = f"{self._greeting_text} {spoken}".strip()
        if self._turn_filler_text:
            return f"{self._turn_filler_text} {spoken}".strip()
        return spoken

    def _maybe_schedule_filler(self, tool_name: str) -> None:
        """Kick off the filler flow for this turn's first tool call.

        At most one flow per turn — later tool calls (same or subsequent LLM
        iterations) reuse the guard. The flow itself decides whether anything
        is actually spoken.
        """
        if self.filler is None or not self.filler.enabled or self._turn_filler_task is not None:
            return
        self._turn_filler_task = asyncio.create_task(self._filler_flow(tool_name))

    def _filler_ok_to_speak(self) -> bool:
        """The turn is still live and nothing has been spoken since the last
        filler (baselines are 0 before the first one — nothing spoken at all).

        ``_turn_assistant_text`` matters too: streamed deltas can sit in the
        TTS buffer below the flush threshold (nothing *scheduled* yet), but a
        spoken preamble is coming — a filler on top of it would double-talk.
        """
        return (
            not self._cancel_turn.is_set()
            and not self._turn_finalized_ok
            and not self._turn_assistant_text
            and not self._turn_tts_scheduled_text
            and self._turn_tts_segments == self._turn_filler_segments_baseline
            and self._turn_audio_bytes == self._turn_filler_audio_baseline
        )

    async def _filler_flow(self, tool_name: str) -> None:
        """Speak the first filler after the grace delay; optionally repeat.

        The first generation overlaps the delay (latency-critical). Follow-ups
        (``repeat_secs``) fire only on continued silence and generate on
        demand — no speculative LLM call per window.
        """
        spoken: list[str] = []
        try:
            gen = asyncio.create_task(self._generate_filler(tool_name, spoken))
            try:
                await asyncio.sleep(self.filler.delay_secs)
                if not self._filler_ok_to_speak():
                    return
                async with asyncio.timeout(self.filler.timeout_secs):
                    phrase = await asyncio.shield(gen)
            except TimeoutError:
                logger.debug("filler_generation_timeout", tool=tool_name, **_trace_debug_fields())
                return
            finally:
                if not gen.done():
                    gen.cancel()
            # Re-check: the reply may have started streaming while we generated.
            if not phrase or not self._filler_ok_to_speak():
                return
            await self._speak_filler(phrase, tool_name)
            spoken.append(phrase)

            while self.filler.repeat_secs is not None and len(spoken) < self.filler.max_per_turn:
                await asyncio.sleep(self.filler.repeat_secs)
                if not self._filler_ok_to_speak():
                    return
                try:
                    async with asyncio.timeout(self.filler.timeout_secs):
                        phrase = await self._generate_filler(tool_name, spoken)
                except TimeoutError:
                    logger.debug("filler_generation_timeout", tool=tool_name, **_trace_debug_fields())
                    return
                if not phrase or not self._filler_ok_to_speak():
                    return
                await self._speak_filler(phrase, tool_name)
                spoken.append(phrase)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Dead air is the status quo — a broken filler must never surface
            # to the caller as an error.
            logger.warning("filler_failed", error=str(e), tool=tool_name, **_trace_debug_fields())

    async def _speak_filler(self, phrase: str, tool_name: str) -> None:
        """Synthesize one filler phrase as a dedicated task on the TTS chain.

        A *separate* task claims ``_tts_tail`` (not the flow task, which lives
        through repeat windows) so reply text scheduled mid-filler waits only
        for the speech, never for a sleeping repeat loop.
        """
        # No await between the caller's ok-check and this claim.
        speak_task = asyncio.create_task(self._speak_filler_task(phrase, tool_name))
        self._tts_tail = speak_task
        self._tts_tasks.add(speak_task)
        speak_task.add_done_callback(lambda t: self._tts_tasks.discard(t))
        try:
            await speak_task
        except asyncio.CancelledError:
            speak_task.cancel()
            raise

    async def _speak_filler_task(self, phrase: str, tool_name: str) -> None:
        # Text goes in *before* synthesis: audio can start (and echo back
        # through the mic) mid-``_speak``, so suppression must already know
        # the phrase even if we're cancelled halfway through it.
        self._turn_filler_text = f"{self._turn_filler_text} {phrase}".strip()
        logger.info("filler_speak", phrase=phrase, tool=tool_name, **_trace_debug_fields())
        await self._speak(phrase, filler=True)
        # Everything below commits only after synthesis finishes: a turn
        # cancelled mid-speak must not count a filler that never reached the
        # transcript or the client.
        self._turn_filler_count += 1
        # Follow-up condition is "nothing spoken since this filler" — snapshot
        # what the filler itself contributed.
        self._turn_filler_segments_baseline = self._turn_tts_segments
        self._turn_filler_audio_baseline = self._turn_audio_bytes
        # Part of what was said on the call (recordings include it), but
        # never `full_response` / agent memory.
        self._transcript.append(TranscriptEntry(role="assistant", text=phrase, filler=True))
        await self._emit(FillerSpoken(text=phrase))

    async def _generate_filler(self, tool_name: str, previous: list[str] | None = None) -> str | None:
        """One-shot LLM completion for the phrase.

        Runs as its own root run (Runnable creates a fresh RunContext when it
        sees a live sibling context), so it never nests into the turn's trace.
        """
        if self._filler_agent is None:
            self._filler_agent = Agent(
                name="voice_filler",
                model=self.filler.model or self.model or self.agent.model,
                system_prompt=self.filler.system_prompt,
                max_tokens=64,
                tracing_provider=None,
            )
        prompt = (
            f"The user just said: {self._active_turn_user_text!r}\n"
            f"You are now running the {tool_name!r} tool to handle it. Say your filler phrase."
        )
        if previous:
            said = "; ".join(repr(p) for p in previous)
            prompt += (
                f"\nThe work is taking a while and you already told the caller: {said}. "
                "Say a different short follow-up so they know you're still there."
            )
        out = await self._filler_agent(prompt=prompt).collect()
        if out.status.code != "success" or out.output is None:
            logger.debug("filler_generation_failed", status=out.status.code, error=out.error)
            return None
        return out.output.collect_text().strip().strip('"').strip() or None

    async def _seed_call_context(self) -> None:
        """Plant turn one's identity before any turn runs: telephony call
        context on the session bag, and the run this call continues from.

        Extra RunContext attrs are dropped when turn 2 forks a child context
        (runnable.py: existing ``_trace`` → new ``RunContext(parent_id=…)``).
        The session bag is not: turn 1 reuses this empty-``_trace`` context,
        ``_save_trace`` writes ``root.session``, and turn 2+ reloads it via
        ``get_session()``.

        ``parent_run_id`` (text → voice continuity) rides the same seed. It
        must be *on the context*, not passed as ``agent(parent_id=…)``: the
        reuse path in ``Runnable.__call__`` keeps an ambient empty-trace
        context as-is and would drop the kwarg. The seed carries the agent's
        tracing provider so ``get_session()`` (here) and memory resolution
        (turn 1) can actually reach the parent run's trace.
        """
        if not self.call_context and not self.parent_run_id:
            return
        ctx = get_run_context()
        if ctx is None:
            # getattr: the session accepts duck-typed agents (tests, custom
            # wrappers); missing attribute falls back to env auto-detection,
            # which is what a bare RunContext() did before the seed carried
            # the provider at all.
            ctx = RunContext(
                parent_id=self.parent_run_id,
                tracing_provider=getattr(self.agent, "tracing_provider", TRACING_UNSET),
            )
        elif self.parent_run_id and ctx.parent_id is None and not ctx._trace:
            # A transport seeded an ambient context of its own (platform run
            # wrapper). Point it at the conversation this call joins; a context
            # that already names a parent or carries spans is left alone.
            ctx.parent_id = self.parent_run_id
        session_data = await ctx.get_session()
        session_data.update(self.call_context)
        set_run_context(ctx)

    # -- Internal: agent-speaks-first greeting --------------------------------

    def _maybe_start_greeting(self) -> None:
        """Arm the opener flow. No-op when no greeting is configured."""
        if self.greeting is None or self._closed:
            return
        self._greeting_task = asyncio.create_task(self._greeting_flow())

    def _greeting_holds_interrupt(self) -> bool:
        """True while a non-interruptible opener still owns the client's ears.

        ``interrupt()`` spares the opener wholesale rather than the greeting
        opting out of cancellation piecemeal, because the parts are not
        separable: the call that truncates ``_speak`` is the same one that
        clears the client's playback buffer and snapshots heard bytes. Skipping
        all of it keeps the "what the user heard" accounting consistent by
        never starting it.

        The window runs to the end of *playback*, not the end of synthesis.
        TTS outruns the wire by design, so a hold that ended with ``_speak``
        would leave the common barge-in shape — opener synthesized, audio still
        queued — cutting the very sentence that says who is calling.

        It closes early if the reply behind the opener has already put audio on
        the wire: the two then share one client-side buffer that cannot be
        cleared selectively, and the opener has fully drained by then anyway
        (``_schedule_tts`` holds the reply until it has). Fillers and the
        turn-timeout fallback reach ``_speak`` without that gate, which is why
        this is checked rather than assumed.

        Barge-in is deferred, not dropped — the commit behind it still reaches
        ``_begin_user_turn``, and that turn's TTS chains behind the opener on
        ``_tts_tail``. Same contract as Vapi's
        ``firstMessageInterruptionsEnabled: false`` and LiveKit's ``on_enter``
        reply node (``allow_interruptions=False``).

        Never gates ``close()``: that sets ``_closed`` before interrupting, and
        a caller who hung up must not hold the box open to the last syllable.
        """
        if self._closed or self.greeting is None or self.greeting.interruptible:
            return False
        if not (self._greeting_speaking or self._greeting_pending_bytes() > 0):
            return False
        return self._turn_audio_bytes == 0

    async def _interrupt_turn_behind_greeting(self) -> None:
        """Tear down a reply queued behind a non-interruptible opener.

        The opener keeps the wire: its audio, the client's playback buffer and
        its transcript entry are all untouched. Only the reply is cancelled —
        and by construction it has not been heard, because its TTS waits for
        the opener to drain before emitting a byte.

        Without this, a *second* commit inside the opener would find the first
        turn still running (``interrupt()`` spared it along with the greeting)
        and ``_begin_user_turn`` would simply overwrite the task handle,
        leaving two turns racing on one session's turn state.
        """
        if self._current_turn_task is None or self._current_turn_task.done():
            return
        streamed = self._turn_assistant_text
        logger.info(
            "session_turn_cancelled_behind_greeting",
            assistant_chars=len(streamed),
            **_trace_debug_fields(),
        )
        self._cancel_turn.set()
        # The opener is deliberately outside ``_tts_tasks`` while it is
        # non-interruptible, so this drains the reply's segments only.
        for t in list(self._tts_tasks):
            if not t.done():
                t.cancel()
        if self._tts_tasks:
            await asyncio.gather(*self._tts_tasks, return_exceptions=True)
        self._tts_tasks.clear()
        await self._abort_tts_stream()
        self._current_turn_task.cancel()
        try:
            await self._current_turn_task
        except (asyncio.CancelledError, Exception):
            pass
        self._active_turn_user_text = ""
        self._turn_finalized_ok = False
        self._is_speaking = False
        heard = self._last_interruption_heard_text
        self._last_interruption_heard_text = None
        # Only when the reply actually streamed text: the client rewrites its
        # newest assistant bubble on this event, and the opener's bubble is
        # still open. Announcing a reply that never produced a token would make
        # the client drop the opener instead.
        if streamed:
            await self._emit(SessionInterrupted(heard_text=heard or ""))
        # The cancelled turn nulled ``_tts_tail`` in its ``finally``. The opener
        # is still speaking and still owns the chain — put it back, or the next
        # reply will talk over it.
        if self._greeting_speak_task is not None and not self._greeting_speak_task.done():
            self._tts_tail = self._greeting_speak_task

    def _greeting_still_wanted(self) -> bool:
        """False once the caller has spoken first (or the session is closing).

        ``delay_ms`` and LLM-authored openers both leave a window in which the
        callee says "hello?" first. Speaking now would talk over their turn, and
        the reason for an opener — that the agent must not wait to be prompted —
        no longer applies once it has been.
        """
        if self._closed:
            return False
        if self._current_turn_task is not None or self._held_user_text is not None:
            logger.info("greeting_skipped", reason="user_spoke_first")
            return False
        return True

    async def _greeting_flow(self) -> None:
        """Hold ``delay_ms``, resolve the line, speak it."""
        try:
            if self.greeting.delay_ms:
                await asyncio.sleep(self.greeting.delay_ms / 1000)
                if not self._greeting_still_wanted():
                    return
            # Static text wins when both are set: ~300ms to first audio against
            # ~1.5s for a generated line, and the wording is known in advance.
            text = (self.greeting.text or "").strip()
            if not text:
                text = (await self._generate_greeting() or "").strip()
                if not text or not self._greeting_still_wanted():
                    return
            await self._await_greeting_speech(self._claim_greeting(text))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Silence at the top of the call is the status quo, not something to
            # surface to the caller as a session error.
            logger.warning("greeting_failed", error=str(e), exc_info=True)

    def _claim_greeting(self, text: str) -> asyncio.Task[None]:
        """Stake the TTS chain for the opener and start synthesizing it.

        Deliberately synchronous, and deliberately the *last* step of the flow:
        everything a racing STT commit could observe — the words, the interrupt
        gate, the tail of the TTS chain — is in place before any await, so a
        transcript already sitting in the provider's queue cannot open turn one
        in the gap and leave the opener half-armed.

        Same shape as :meth:`_speak_filler`: the *speak* task holds
        ``_tts_tail`` so a first reply scheduled mid-greeting queues behind this
        audio instead of interleaving with it.
        """
        self._greeting_text = text
        self._greeting_speaking = True
        # Recorded up front rather than on completion (the filler's shape),
        # because the opener is the first thing on the call: a barge-in appends
        # its user entry the moment it lands, so an entry written after
        # synthesis would sort *after* the words it interrupted. Truncation on
        # barge-in rewrites this entry — see :meth:`_truncate_greeting`.
        self._greeting_entry = TranscriptEntry(role="assistant", text=text)
        self._transcript.append(self._greeting_entry)
        logger.info(
            "greeting_speak",
            text_preview=text[:120],
            interruptible=self.greeting.interruptible,
            generated=not (self.greeting.text or "").strip(),
        )
        speak_task = asyncio.create_task(self._speak_greeting_task(text))
        self._tts_tail = speak_task
        self._greeting_speak_task = speak_task
        if self.greeting.interruptible:
            # ``_tts_tasks`` is the pool ``interrupt()`` cancels, so an
            # interruptible opener is cut like any other speech. A
            # non-interruptible one deliberately stays out of it: that also keeps
            # it out of reach of the *other* thing that drains that pool — a
            # first turn tearing down on an error while we are mid-sentence.
            # ``close()`` still reaches it by cancelling the flow task.
            self._tts_tasks.add(speak_task)
            speak_task.add_done_callback(lambda t: self._tts_tasks.discard(t))
        return speak_task

    async def _await_greeting_speech(self, speak_task: asyncio.Task[None]) -> None:
        try:
            await speak_task
        except asyncio.CancelledError:
            speak_task.cancel()
            raise

    async def _speak_greeting_task(self, text: str) -> None:
        # Delta first so clients open a reply bubble as the audio starts; the
        # interruption rewrite targets that bubble.
        await self._emit(AgentTextDelta(text=text))
        try:
            await self._speak(text, greeting=True)
        except asyncio.CancelledError:
            # Interruptible opener, cut off mid-sentence. ``interrupt()`` has
            # already mapped the played bytes to the heard prefix and rewritten
            # the transcript entry (``_truncate_greeting``); all that is left is
            # sealing the client's open bubble so it does not hang.
            await self._emit(AgentTextDone(text=self._last_interruption_heard_text or ""))
            raise
        finally:
            self._greeting_speaking = False
        # Normal end-of-speech event rather than a greeting-specific one: clients
        # seal the bubble on it, and the telephony bridge flushes its sub-20ms
        # downlink tail on it (``flush_events`` in server/telephony.py).
        await self._emit(AgentTextDone(text=text))

    def _truncate_greeting(self) -> None:
        """Align the opener's transcript entry with what the caller heard.

        Called from ``interrupt()`` while the played axis still means something —
        the next statement clears the client's buffer, discarding whatever the
        opener had queued. Covers both barge-in shapes: cut mid-synthesis, and
        cut after ``_speak`` returned but while audio was still playing (the
        common one, since TTS outruns playback).

        Uses ``playback.played_bytes`` directly rather than a turn baseline: the
        opener is by construction the session's first audio, so the played axis
        and its segment record share an origin.
        """
        record = self._greeting_record
        self._greeting_record = None
        if record is None or self._greeting_entry is None:
            return
        full = str(record[0])
        heard = map_played_bytes_to_text([(full, int(record[1]))], self.playback.played_bytes)
        if heard == full:
            return
        index = next(
            (i for i, e in enumerate(self._transcript) if e is self._greeting_entry),
            None,
        )
        if index is None:
            return
        logger.info(
            "greeting_interruption_truncation",
            heard_chars=len(heard),
            greeting_chars=len(full),
            **_trace_debug_fields(),
        )
        # Only when the opener is still the newest thing said does
        # ``SessionInterrupted`` describe *it* — that is what makes the client
        # rewrite (or drop) the opener's bubble. Once a reply sits on top, that
        # turn's own truncation owns the field.
        was_newest = index == len(self._transcript) - 1
        if heard:
            self._greeting_entry = TranscriptEntry(role="assistant", text=heard)
            self._transcript[index] = self._greeting_entry
        else:
            self._transcript.pop(index)
            self._greeting_entry = None
        if was_newest:
            self._last_interruption_heard_text = heard

    async def _await_greeting_drain(self) -> None:
        """Hold a reply's audio until a non-interruptible opener has played out.

        Emission order is playback order, so handing the reply to the transport
        early does not make it arrive sooner — it only stacks it into the same
        client buffer as the opener, at which point barge-in can no longer
        clear one without the other. Waiting keeps
        :meth:`_greeting_holds_interrupt`'s window exact.

        Bounded by the audio still outstanding: a playback tracker that stalls
        (missing acks, a transport that never drains) costs the reply a beat,
        never the call.
        """
        if self.greeting is None or self.greeting.interruptible:
            return
        pending = self._greeting_pending_bytes()
        if pending <= 0:
            return
        bytes_per_second = max(1, self.audio_output.sample_rate * 2)
        deadline = time.monotonic() + pending / bytes_per_second + 1.0
        while self._greeting_pending_bytes() > 0 and not self._cancel_turn.is_set():
            if time.monotonic() >= deadline:
                logger.info(
                    "greeting_drain_wait_timeout",
                    pending_bytes=self._greeting_pending_bytes(),
                    **_trace_debug_fields(),
                )
                return
            await asyncio.sleep(0.02)

    def _greeting_heard_text(self) -> str:
        """The opener as the caller received it, or ``""`` if they never did.

        Tracks the transcript entry rather than ``_greeting_text`` because
        :meth:`_truncate_greeting` rewrites that entry to the heard prefix on a
        barge-in, and drops it entirely when nothing was played.
        """
        if self._greeting_entry is None:
            return ""
        return self._greeting_entry.text

    def _greeting_pending_bytes(self) -> int:
        """Opener audio the caller has not reached yet, on the played axis.

        A deferred barge-in can start turn one while the greeting is still
        draining, and the turn's played baseline is taken *now* — so those bytes
        would otherwise be credited to the reply by interruption truncation,
        making it look like the caller heard words they had not reached.
        """
        record = self._greeting_record
        if record is None:
            return 0
        return max(0, int(record[1]) - self.playback.played_bytes)

    async def _agent_system_prompt(self) -> str | None:
        """The agent's own resolved system prompt, or ``None``.

        Reaches for ``Agent._resolve_system_prompt`` because that is the only
        thing that assembles all three spellings (string, sync/async callable,
        appended skills). Resolved from the session rather than the server's
        session builder because the ``Agent`` there is process-wide and shared
        across connections — a per-call greeting must not mutate it.
        """
        try:
            return await self.agent._resolve_system_prompt()
        except Exception as e:
            logger.debug("greeting_system_prompt_unavailable", error=str(e))
            return None

    async def _generate_greeting(self) -> str | None:
        """One-shot LLM completion for an ``instructions`` opener.

        Its own root run (like :meth:`_generate_filler`) so it never nests into
        a turn's trace, and carrying the agent's own system prompt so the line
        knows who it is calling as.

        Unlike the filler, this runs *before* any turn — on the very context
        :meth:`_seed_call_context` planted for turn one. ``Runnable.__call__``
        reuses a context whose trace is empty, so without clearing it first the
        generator would claim that context as its own run: turn one would then
        fork a child, and the call identity it was supposed to start with would
        have to survive a round-trip through the tracing provider.
        """
        instructions = (self.greeting.instructions or "").strip()
        if not instructions:
            return None
        parts = [
            p
            for p in (await self._agent_system_prompt(), DEFAULT_GREETING_SYSTEM_PROMPT, instructions)
            if p
        ]
        # Kept on the session (like ``_filler_agent``) so the assembled prompt is
        # readable after the fact — it is the whole configuration surface of a
        # generated opener, and it runs exactly once per call.
        self._greeting_agent = Agent(
            name="voice_greeting",
            model=self.greeting.model or self.model or self.agent.model,
            system_prompt="\n\n".join(parts),
            max_tokens=128,
            tracing_provider=None,
        )
        seeded_ctx = get_run_context()
        set_run_context(None)
        try:
            out = await self._greeting_agent(prompt="Say your opening line.").collect()
        finally:
            set_run_context(seeded_ctx)
        if out.status.code != "success" or out.output is None:
            logger.warning("greeting_generation_failed", status=out.status.code, error=out.error)
            return None
        return out.output.collect_text().strip().strip('"').strip() or None

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
                    # Shielded: ``Task.cancel()`` propagates to whatever the task
                    # is awaiting, so an unshielded ``await prev`` would let a
                    # segment torn down mid-chain reach back and cut the one
                    # ahead of it — including a greeting that is explicitly not
                    # interruptible. Everything that must die is cancelled
                    # directly through ``_tts_tasks`` instead.
                    await asyncio.shield(prev)
                except asyncio.CancelledError:
                    # The shield reports both directions. ``prev`` cancelled is
                    # the predecessor dying, which this segment outlives;
                    # anything else is *this* segment being torn down, and
                    # swallowing that would let it speak after the turn is gone.
                    if not prev.cancelled():
                        raise
                except Exception:
                    pass
            await self._await_greeting_drain()
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
                if self._recorder is not None:
                    self._recorder.add_agent(chunk)
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

    async def _speak(self, text: str, *, filler: bool = False, greeting: bool = False) -> None:
        text = _strip_markdown(text)
        logger.debug(
            "turn_tts_synthesize_begin",
            text_chars=len(text),
            text_preview=text[:120],
            filler=filler,
            greeting=greeting,
            cancel_turn_set=self._cancel_turn.is_set(),
            **_trace_debug_fields(),
        )
        chunk_count = 0
        total_bytes = 0
        if not greeting:
            if self._turn_tts_started_at is None:
                self._turn_tts_started_at = time.monotonic()
            self._turn_tts_segments += 1
        # Filler segments record empty text: interruption truncation then counts
        # their played bytes without attributing words to the *reply*.
        segment_record: list = ["" if filler else text, 0]
        if greeting:
            # The opener belongs to no turn: it is not in any turn's segment
            # list (``_run_turn`` seeds a text-less stand-in for whatever is
            # still draining when it starts) and it must not move that turn's
            # TTS timings. Keep a direct handle so its byte count stays
            # readable after the first turn rebinds the record list.
            self._greeting_record = segment_record
        else:
            self._turn_tts_segment_records.append(segment_record)
        try:
            async for chunk in self.tts.synthesize(text):
                # The opener predates every turn, so the per-turn cancel flag says
                # nothing about it: stopping it is ``interrupt()``'s job, by
                # cancelling this task, and only when it is interruptible.
                if self._cancel_turn.is_set() and not greeting:
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
                if not greeting:
                    if self._turn_first_audio_at is None:
                        self._turn_first_audio_at = time.monotonic()
                    self._turn_audio_bytes += len(chunk)
                if self._record_audio:
                    self._output_audio_chunks.append(chunk)
                if self._recorder is not None:
                    self._recorder.add_agent(chunk)
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
            if not greeting:
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

        def _ms(t0: float | None, t1: float | None) -> float | None:
            if t0 is None or t1 is None:
                return None
            return round((t1 - t0) * 1000, 1)

        eou = self._turn_eou_at or None
        eou_to_first_audio = _ms(eou, self._turn_first_audio_at)
        ctx = get_run_context()
        return TurnMetrics(
            turn_index=self._turn_index,
            run_id=ctx.id if ctx is not None else None,
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
            filler_spoken=self._turn_filler_count > 0,
            filler_count=self._turn_filler_count,
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

    def _turn_run_id(self) -> str | None:
        """Id of the in-flight turn's run, or None if no run started.

        Only valid inside ``_run_turn``: the first ``__anext__`` on the agent
        generator swaps the new run's context in (top-level calls leave it set
        on exit), while ``_last_run_context`` still points at the *previous*
        turn until the turn's ``finally`` — so ``ambient is _last_run_context``
        means the generator never got far enough to start a run.

        The identity check alone is not enough: ``_seed_call_context`` plants
        an empty-trace context before turn one, so a turn that dies before the
        first ``__anext__`` would otherwise report the seed's id — a pointer to
        a run that never started and persisted nothing. An empty trace means no
        run recorded anything, whoever owns the context.
        """
        ctx = get_run_context()
        if ctx is None or ctx is self._last_run_context or not ctx._trace:
            return None
        return ctx.id

    async def _emit(self, event: VoiceSessionEvent | None) -> None:
        await self._event_queue.put(event)

    def _drop_queued_audio_output(self) -> int:
        """Remove pending :class:`AudioOutput` frames so interrupt is not delayed.

        TTS can enqueue megabytes of PCM before the consumer (WS send) catches
        up. Barge-in must surface ``SessionInterrupted`` immediately; unplayed
        queued audio is discarded the same way the client clears its buffer.
        Non-audio events stay in order.
        """
        kept: list[VoiceSessionEvent | None] = []
        dropped = 0
        while True:
            try:
                event = self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if isinstance(event, AudioOutput):
                dropped += 1
                continue
            kept.append(event)
        for event in kept:
            self._event_queue.put_nowait(event)
        return dropped

    async def _cleanup(self) -> None:
        self._cancel_hold()
        self._held_user_text = None
        if self._llm_warmup_task is not None and not self._llm_warmup_task.done():
            self._llm_warmup_task.cancel()
        self._llm_warmup_task = None
        # The flow may be sleeping out ``delay_ms`` or waiting on a generation;
        # its speak task is cancelled with the rest of ``_tts_tasks`` below.
        if self._greeting_task is not None and not self._greeting_task.done():
            self._greeting_task.cancel()
        self._greeting_task = None
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
        await self._finalize_recording()

    async def _finalize_recording(self) -> None:
        """Flush the call recording and fire the on_saved hook. Never raises."""
        if self._recorder is None:
            return
        try:
            from .recording import build_manifest

            manifest = build_manifest(
                session_id=self.session_id,
                started_at=self.started_at,
                meta=self.recording_meta,
                transcript=self._transcript,
                turns=self._metrics,
                recorder=self._recorder,
            )
            result = self._recorder.close(manifest=manifest)
        except Exception as e:
            logger.error("recording_finalize_failed", error=str(e), exc_info=True)
            return
        hook = getattr(self._recorder, "on_saved", None)
        if result is not None and hook is not None:
            try:
                await hook(result)
            except Exception as e:
                logger.error("recording_on_saved_failed", error=str(e), exc_info=True)


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


def _dont_greet_again_prompt(system_prompt: str | None, greeting: str) -> str:
    """``system_prompt`` plus one line stating the call is already open.

    Passed as the first turn's ``system_prompt`` override — a note, not a
    rewrite, so a callable or skill-bearing prompt survives intact. ``greeting``
    is what the caller actually heard, which on a barge-in can be a prefix of
    the configured line.
    """
    note = (
        f'You already opened this call by saying: "{greeting}". '
        "The caller has heard it. Do not greet them or introduce yourself again — "
        "answer what they just said."
    )
    return f"{system_prompt}\n\n{note}" if system_prompt else note


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
