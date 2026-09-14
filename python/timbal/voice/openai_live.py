"""OpenAI GPT-Live (``gpt-live-1``) — full-duplex voice with a Timbal Agent as the backend.

GPT-Live is neither the cascaded STT → Agent → TTS pipeline
(:class:`~timbal.voice.VoiceSession`) nor a turn-based speech-to-speech model
(:class:`~timbal.voice.RealtimeSession`). Measured against the API
(2026-09-14, primary WebSocket, ``audio/pcm`` @ 24 kHz):

* **Full duplex, wall-clock paced.** The server emits one 100 ms PCM frame
  every 100 ms from ``session.started`` until ``session.closed`` — digital
  silence while nobody speaks, speech embedded in the same stream. There is
  no "turn": no start/done markers around a reply, no audio-done event, and
  a client-side playback queue never holds more than network jitter. What
  the caller heard *is* the output transcript, because speech is voiced at
  1x and stops when the model stops.
* **Server-owned turn-taking.** The model listens while it speaks and
  handles barge-in itself. There is no client interrupt, no truncate, no
  VAD, no turn detector. Input audio must flow continuously (silence
  included) or the model has nothing to time against.
* **Transcripts are fragments with timeline stamps.** ``session.input_transcript.delta``
  / ``session.output_transcript.delta`` carry ``delta`` + ``start_ms`` /
  ``end_ms`` on the session clock. Fragments of both speakers may overlap.
  Rows (turn-ish groupings) are a client heuristic: this module groups by
  gap on the session timeline and closes rows on a wall-clock timer.
* **Delegation.** The model decides when it needs a backend and sends
  ``session.delegation.created`` — metadata only, no task text. The client
  reconstructs the request from the transcript, runs the backend, and
  streams results back: ``session.thinking.append`` (quiet context) and
  ``session.commentary.append`` (spoken, paraphrased). Speech continues
  while the backend works ("Checking." … result). Interrupting speech does
  not cancel backend work.

:class:`OpenAILiveClient` is the transport (raw ``websockets``; the installed
``openai`` SDK predates the ``live`` namespace). :class:`LiveSession` adapts it
onto the :class:`~timbal.voice.VoiceSessionEvent` vocabulary and runs a
:class:`~timbal.core.agent.Agent` per delegation — tools, tracing, memory
chaining, approvals and suspensions all behave as in the cascaded session;
only the mouth and ears are OpenAI's.

Pricing (launch): $0.05 / minute of session, billed per second, reported as
``session.usage.updated`` / ``session.closed`` ``usage.seconds``. The backend
Agent bills through its own model as usual.

Requires ``websockets`` and ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from ..state import get_run_context, set_run_context
from ..state.context import RunContext
from ..state.tracing.providers import TRACING_UNSET
from ..types.content import TextContent
from ..types.events import ApprovalEvent, InteractionEvent, OutputEvent, StartEvent
from ..types.events.delta import DeltaEvent, Text, TextDelta, ToolUse
from ..types.message import Message
from .events import (
    AgentApproval,
    AgentInteraction,
    AgentStatus,
    AgentTextDelta,
    AgentTextDone,
    AudioOutput,
    DelegationCreated,
    DelegationResult,
    SessionEnded,
    SessionError,
    SessionStarted,
    TranscriptCommitted,
    TranscriptEntry,
    TranscriptPartial,
    VoiceSessionEvent,
)
from .metrics import TurnMetrics, TurnMetricsEvent
from .providers import AudioInputConfig, AudioOutputConfig

if TYPE_CHECKING:
    from ..core.agent import Agent

logger = structlog.get_logger("timbal.voice.openai_live")

LIVE_URL = "wss://api.openai.com/v1/live/sessions"
DEFAULT_MODEL = "gpt-live-1"
DEFAULT_VOICE = "marin"
PRICE_PER_MINUTE_USD = 0.05

_PCM16 = frozenset({"pcm_s16le", "linear16", "pcm16", "pcm"})
_PCMU = frozenset({"pcmu", "mulaw", "ulaw", "g711_ulaw", "g711u"})
_PCMA = frozenset({"pcma", "alaw", "g711_alaw", "g711a"})

# Appends are capped at 500 tokens by the API. ~4 chars/token, with headroom.
_APPEND_MAX_CHARS = 1400
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
# Caller speech that does not make a pending result stale: acknowledgments and
# fillers, one to four of them ("okay", "yeah sure", "mm-hm thanks", ...).
_BACKCHANNEL = re.compile(
    r"(?:(?:ok(?:ay)?|k|yes|yeah|yep|yup|sure|right|alright|fine|great|good|cool|thanks?|thank you|please|"
    r"go ahead|got it|i see|mm+-?hm+|mhm+|uh-?huh|hm+|mm+|oh|ah|aha|wow|nice|perfect|exactly|no|nope|wait|"
    r"one (?:sec|second|moment)|hold on|hello|hi|hey|vale|sí|si|claro|bueno|bien|gracias|d'accord|oui|ja|okey)"
    r"[.,!?…]*\s*){1,4}"
)


def _wire_format(sample_rate: int, encoding: str) -> dict[str, Any]:
    enc = encoding.lower()
    if enc in _PCM16:
        if sample_rate not in (16_000, 24_000):
            raise ValueError(f"GPT-Live PCM supports 16000 or 24000 Hz, got {sample_rate}")
        return {"type": "audio/pcm", "rate": sample_rate}
    if enc in _PCMU:
        return {"type": "audio/pcmu", "rate": 8000}
    if enc in _PCMA:
        return {"type": "audio/pcma", "rate": 8000}
    raise ValueError(f"Unsupported GPT-Live audio encoding: {encoding!r}")


def _resolve_api_key(explicit: str | SecretStr | None) -> str:
    if isinstance(explicit, SecretStr):
        explicit = explicit.get_secret_value()
    key = explicit or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("Set OPENAI_API_KEY or pass api_key to OpenAILiveClient.")
    return key


def split_for_append(text: str, max_chars: int = _APPEND_MAX_CHARS) -> list[str]:
    """Split backend text into append-sized chunks on sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur = ""
    for sent in _SENTENCE_END.split(text):
        if not sent:
            continue
        while len(sent) > max_chars:  # single overlong sentence: hard-cut
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(sent[:max_chars])
            sent = sent[max_chars:]
        if cur and len(cur) + 1 + len(sent) > max_chars:
            chunks.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


class LiveTransport(ABC):
    """Bidirectional GPT-Live connection. One instance == one session.

    ``connect`` performs ``session.start`` and resolves on ``session.started``.
    ``events`` yields raw server events (dicts) and ends when the socket
    closes. ``send`` ships a raw client command. ``close`` performs the
    graceful ``session.close`` → ``session.closed`` handshake and returns the
    final ``usage`` (``None`` when unconfirmed).
    """

    @abstractmethod
    async def connect(self) -> dict[str, Any]: ...

    @abstractmethod
    async def send_audio(self, chunk: bytes) -> None: ...

    @abstractmethod
    async def send(self, event: dict[str, Any]) -> None: ...

    @abstractmethod
    def events(self) -> AsyncIterator[dict[str, Any]]: ...

    @abstractmethod
    async def close(self, timeout: float = 15.0) -> dict[str, Any] | None: ...

    async def reconnect(self, *, input: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Open a *replacement* session after an unexpected drop.

        A fork (``/live/sessions/{id}/fork``) needs a finalized stored
        recording, which does not exist mid-call, so recovery is a fresh
        ``session.start`` seeded with ``input`` (Responses-style message items
        rebuilt from the transcript). New session id, timeline restarts at 0.
        Default: unsupported.
        """
        raise NotImplementedError("transport does not support reconnect")

    @property
    def finalized(self) -> bool:
        """True once ``session.closed`` was received — the drop was not unexpected."""
        return False


class OpenAILiveClient(LiveTransport):
    """Primary WebSocket to ``wss://api.openai.com/v1/live/sessions``.

    ``delegation`` — ``"client"`` (default; a :class:`LiveSession` answers
    delegations with a Timbal Agent) or a full ``{"type": "responses", ...}``
    dict for OpenAI-managed Responses delegation. ``input`` seeds prior text
    history (Responses-style message items). ``extra`` merges into the
    ``session`` object verbatim (``store``, future fields).
    """

    def __init__(
        self,
        *,
        api_key: str | SecretStr | None = None,
        model: str = DEFAULT_MODEL,
        instructions: str = "",
        voice: str | None = DEFAULT_VOICE,
        sample_rate: int = 24_000,
        encoding: str = "pcm_s16le",
        delegation: Literal["client"] | dict[str, Any] = "client",
        input: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
        url: str = LIVE_URL,
        connect_timeout: float = 20.0,
        safety_identifier: str | None = None,
    ) -> None:
        self._api_key_explicit = api_key
        self.model = model
        self.instructions = instructions
        self.voice = voice
        self.sample_rate = sample_rate
        self.encoding = encoding
        self.delegation = delegation
        self.input = input
        self.extra = extra or {}
        self.url = url
        self.connect_timeout = connect_timeout
        self.safety_identifier = safety_identifier

        self._ws: Any = None
        self._queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._reader: asyncio.Task | None = None
        self._started: asyncio.Future[dict[str, Any]] | None = None
        self._closed_event: asyncio.Future[dict[str, Any]] | None = None
        self.session: dict[str, Any] | None = None
        self.usage_seconds: float | None = None
        self.context_usage_ratio: float | None = None
        self.close_reason: str | None = None

    # -- Session config -------------------------------------------------------

    def session_config(self) -> dict[str, Any]:
        audio: dict[str, Any] = {"format": _wire_format(self.sample_rate, self.encoding)}
        if self.voice:
            audio["output"] = {"voice": self.voice}
        cfg: dict[str, Any] = {"model": self.model, "audio": audio}
        if self.instructions:
            cfg["instructions"] = self.instructions
        cfg["delegation"] = {"type": "client"} if self.delegation == "client" else dict(self.delegation)
        if self.input:
            cfg["input"] = list(self.input)
        cfg.update(self.extra)
        return cfg

    # -- LiveTransport --------------------------------------------------------

    async def connect(self) -> dict[str, Any]:
        from websockets.asyncio.client import connect as ws_connect

        headers = {"Authorization": f"Bearer {_resolve_api_key(self._api_key_explicit)}"}
        if self.safety_identifier:
            headers["OpenAI-Safety-Identifier"] = self.safety_identifier
        loop = asyncio.get_running_loop()
        self._started = loop.create_future()
        self._closed_event = loop.create_future()
        t0 = time.monotonic()
        self._ws = await asyncio.wait_for(
            ws_connect(self.url, additional_headers=headers, max_size=None, ping_interval=20, ping_timeout=20),
            self.connect_timeout,
        )
        self._reader = asyncio.create_task(self._read_loop(), name="openai-live-reader")
        await self.send({"type": "session.start", "event_id": "session_start", "session": self.session_config()})
        try:
            started = await asyncio.wait_for(asyncio.shield(self._started), self.connect_timeout)
        except TimeoutError:
            await self._abort()
            raise ConnectionError("GPT-Live: no session.started within connect_timeout") from None
        self.session = started.get("session") or {}
        logger.info(
            "openai_live_started",
            session_id=self.session.get("id"),
            model=self.model,
            connect_ms=round((time.monotonic() - t0) * 1000),
        )
        return started

    async def send_audio(self, chunk: bytes) -> None:
        # Audio is a lossy uplink by nature: a chunk that lands on a dead or
        # reconnecting socket is dropped, not an error (the mic keeps running).
        if not chunk or self._ws is None:
            return
        with contextlib.suppress(ConnectionError):
            await self.send({"type": "session.input_audio.append", "audio": base64.b64encode(chunk).decode("ascii")})

    async def reconnect(self, *, input: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        await self._abort()
        if input is not None:
            self.input = input
        self.close_reason = None
        return await self.connect()

    @property
    def finalized(self) -> bool:
        return self._closed_event is not None and self._closed_event.done()

    async def send(self, event: dict[str, Any]) -> None:
        if self._ws is None:
            raise ConnectionError("GPT-Live transport is not connected")
        from websockets.exceptions import ConnectionClosed

        try:
            await self._ws.send(json.dumps(event))
        except ConnectionClosed as e:
            raise ConnectionError(f"GPT-Live connection closed: {e}") from e

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            ev = await self._queue.get()
            if ev is None:
                return
            yield ev

    async def close(self, timeout: float = 15.0) -> dict[str, Any] | None:
        if self._ws is None:
            return None
        usage: dict[str, Any] | None = None
        try:
            if self._closed_event is not None and not self._closed_event.done():
                with contextlib.suppress(ConnectionError):
                    await self.send({"type": "session.close"})
                try:
                    closed = await asyncio.wait_for(asyncio.shield(self._closed_event), timeout)
                    usage = closed.get("usage")
                except TimeoutError:
                    logger.warning("openai_live_close_unconfirmed", session_id=(self.session or {}).get("id"))
            elif self._closed_event is not None:
                usage = self._closed_event.result().get("usage")
        finally:
            await self._abort()
        return usage

    # -- Convenience commands ---------------------------------------------------

    async def append_instructions(self, content: str, *, delegation_id: str | None = None, event_id: str | None = None):
        await self._append("session.instructions.append", content, delegation_id, event_id)

    async def append_thinking(self, content: str, *, delegation_id: str | None = None, event_id: str | None = None):
        await self._append("session.thinking.append", content, delegation_id, event_id)

    async def append_commentary(self, content: str, *, delegation_id: str | None = None, event_id: str | None = None):
        await self._append("session.commentary.append", content, delegation_id, event_id)

    async def mute(self) -> None:
        await self.send({"type": "session.input_audio.mute"})

    async def unmute(self) -> None:
        await self.send({"type": "session.input_audio.unmute"})

    async def _append(self, typ: str, content: str, delegation_id: str | None, event_id: str | None) -> None:
        ev: dict[str, Any] = {"type": typ, "delegation_id": delegation_id, "content": content}
        if event_id:
            ev["event_id"] = event_id
        await self.send(ev)

    # -- Internal ----------------------------------------------------------------

    async def _read_loop(self) -> None:
        from websockets.exceptions import ConnectionClosed

        try:
            async for raw in self._ws:
                try:
                    ev = json.loads(raw)
                except (TypeError, ValueError):
                    logger.warning("openai_live_bad_frame")
                    continue
                typ = ev.get("type")
                if typ == "session.started" and self._started and not self._started.done():
                    self._started.set_result(ev)
                elif typ == "error" and self._started and not self._started.done():
                    err = ev.get("error") or {}
                    self._started.set_exception(
                        ConnectionError(f"GPT-Live session.start rejected: {err.get('message') or json.dumps(err)}")
                    )
                elif typ == "session.usage.updated":
                    self.usage_seconds = (ev.get("usage") or {}).get("seconds", self.usage_seconds)
                    self.context_usage_ratio = (ev.get("context_window") or {}).get("usage_ratio")
                elif typ == "session.closed":
                    self.usage_seconds = (ev.get("usage") or {}).get("seconds", self.usage_seconds)
                    self.close_reason = ev.get("reason")
                    if self._closed_event and not self._closed_event.done():
                        self._closed_event.set_result(ev)
                await self._queue.put(ev)
        except ConnectionClosed as e:
            if self._started and not self._started.done():
                self._started.set_exception(ConnectionError(f"GPT-Live connection closed during start: {e}"))
            logger.debug("openai_live_ws_closed", code=getattr(e, "code", None))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("openai_live_reader_error", error=str(e), exc_info=True)
            await self._queue.put({"type": "error", "error": {"message": f"transport: {e}"}})
        finally:
            await self._queue.put(None)

    async def _abort(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._reader
        self._reader = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class TranscriptFragment(BaseModel):
    """One ``*_transcript.delta`` as received: verbatim text + session-timeline interval [start_ms, end_ms)."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    delta: str
    start_ms: int
    end_ms: int


class _Row:
    __slots__ = ("end_ms", "fragments", "start_ms", "text")

    def __init__(self, frag: TranscriptFragment):
        self.text = frag.delta
        self.start_ms = frag.start_ms
        self.end_ms = frag.end_ms
        self.fragments = [frag]

    def add(self, frag: TranscriptFragment) -> None:
        self.text += frag.delta
        self.end_ms = max(self.end_ms, frag.end_ms)
        self.fragments.append(frag)


DelegationPromptBuilder = Callable[["LiveSession", str], str]
"""``(session, delegation_id) -> prompt text`` for the backend agent."""


def default_delegation_prompt(session: LiveSession, delegation_id: str) -> str:  # noqa: ARG001
    """Transcript since the previous delegation, both speakers, plus the open rows.

    The wire event carries no task text (by design: the model may delegate on a
    half-finished sentence). The agent sees what was said since it last helped
    and is told the last user line is the request.
    """
    lines = [f"{f.role}: {f.delta}" for f in session.transcript_since_last_delegation() if f.delta]
    if not lines:
        return "The caller asked for help but no transcript is available yet. Ask what they need."
    return (
        "You are the backend for a live voice assistant. Below is the conversation since you last helped "
        "(`assistant` lines were spoken by the voice layer, not by you). Transcripts may contain ASR errors and "
        "self-corrections; prefer the latest statement. Handle the caller's most recent request. Reply with the "
        "facts to relay — short, plain sentences, no markdown, no greetings. If a required detail is missing, "
        "state exactly what to ask for.\n\n" + "\n".join(lines)
    )


class LiveSession:
    """Voice session on a full-duplex GPT-Live transport with a Timbal Agent backend.

    Same calling convention as :class:`~timbal.voice.VoiceSession`: feed an
    async iterable of mic bytes (``audio_input`` encoding — keep it flowing,
    silence included) to :meth:`run` and consume
    :class:`~timbal.voice.VoiceSessionEvent`s.

    Event mapping:

    * ``session.output_audio.delta`` → :class:`AudioOutput` (continuous, 1x paced).
      ``drop_silence=True`` skips all-zero frames (bandwidth for WS clients that
      play as-they-arrive; keep ``False`` for telephony which needs the clock).
    * user fragments → :class:`TranscriptPartial` (row so far) while the row is
      open, :class:`TranscriptCommitted` when it closes.
    * assistant fragments → :class:`AgentTextDelta` per fragment,
      :class:`AgentTextDone` (``run_id=None`` — the voice model spoke, not a
      run) when the row closes, then :class:`TurnMetricsEvent` whose
      ``eou_to_first_audio_ms`` is measured on the *session timeline*
      (assistant ``start_ms`` − preceding user ``end_ms``), not wall clock.
    * ``session.delegation.created`` → :class:`DelegationCreated`; the agent
      runs; tool calls surface as :class:`AgentStatus` and quiet
      ``thinking.append`` progress; suspensions as :class:`AgentInteraction` /
      :class:`AgentApproval`; the final text goes back as ``commentary.append``
      chunks and :class:`DelegationResult`.

    Rows split when the next fragment starts more than ``row_gap_ms`` after
    the previous one ended on the *session clock* — the precise signal. A
    wall-clock timer of ``row_gap_ms + row_close_slack_ms`` only finalizes a
    row once fragments stop arriving; the slack absorbs delivery jitter
    (fragments land ~1–1.5 s behind the clock, unevenly), so it must not be
    the thing that decides splits. Speakers overlap freely — a user row may
    close while an assistant row is open. There is no :class:`SessionInterrupted`:
    barge-in is server-side, and the assistant row text is what was heard.

    Delegations are serialized so agent memory chains ``parent_id`` in order;
    a delegation arriving while one runs waits and the model is told so.

    **Stale results.** Speech is not cancelled by backend work and vice versa,
    so an answer can land after the caller has moved on. If the caller said
    something substantive (not a backchannel like "okay"/"mm-hm") after the
    delegation was created, ``stale_policy`` decides: ``"thinking"`` (default)
    hands the answer to the model as quiet context with a note of what the
    caller said since — it speaks it only if still relevant; ``"drop"``
    discards it; ``"speak"`` ignores staleness. A newer delegation queued
    behind this one also marks it stale.

    **Reconnect.** An unexpected socket drop (no ``session.closed``) opens a
    replacement session up to ``reconnect_attempts`` times, seeded with the
    transcript so far as ``session.input`` (a fork needs a finalized stored
    recording, which does not exist mid-call). The mic uplink keeps running
    and drops chunks while disconnected; open rows are closed; in-flight
    delegations continue and answer into the new session. Clients see
    :class:`AgentStatus` ``"Reconnecting…"`` / ``"Reconnected"``. A
    ``session.closed`` the client did not request (safety, expiry) is final.
    """

    def __init__(
        self,
        transport: LiveTransport,
        agent: Agent | None = None,
        *,
        audio_input: AudioInputConfig | None = None,
        audio_output: AudioOutputConfig | None = None,
        model: str | None = None,
        parent_run_id: str | None = None,
        row_gap_ms: int = 800,
        row_close_slack_ms: int = 700,
        drop_silence: bool = False,
        record_audio: bool = False,
        delegation_prompt: DelegationPromptBuilder = default_delegation_prompt,
        delegation_timeout_secs: float | None = 120.0,
        on_delegation: Callable[[LiveSession, str, str], Awaitable[str | None]] | None = None,
        reconnect_attempts: int = 3,
        reconnect_backoff_secs: tuple[float, ...] = (0.5, 1.0, 2.0),
        stale_policy: Literal["thinking", "drop", "speak"] = "thinking",
        greeting: str | None = None,
        call_context: dict[str, Any] | None = None,
    ) -> None:
        self.transport = transport
        self.agent = agent
        self.audio_input = audio_input or AudioInputConfig(sample_rate=24_000)
        self.audio_output = audio_output or AudioOutputConfig(sample_rate=24_000)
        self.model = model
        self.parent_run_id = parent_run_id
        self.row_gap_ms = row_gap_ms
        self.row_close_slack_ms = row_close_slack_ms
        self.drop_silence = drop_silence
        self.delegation_prompt = delegation_prompt
        self.delegation_timeout_secs = delegation_timeout_secs
        self.on_delegation = on_delegation
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_backoff_secs = reconnect_backoff_secs
        self.stale_policy = stale_policy
        self.call_context = dict(call_context or {})
        """Per-call identity (``rep_id``, ``from``, ...) planted on the run's
        session bag before the first delegation, for callable system prompts."""
        self.greeting = greeting
        """Opener instruction spoken before the caller talks (e.g. "Greet the
        caller in Spanish and ask how you can help"). Sent once, on the first
        session only — a replacement session is seeded with the transcript."""
        self.reconnects = 0
        # Bumped on every replacement session: the server timeline restarts at 0,
        # so ``*_ms`` values are only comparable within one generation.
        self._generation = 0
        if agent is None and on_delegation is None:
            logger.warning("openai_live_no_backend", hint="delegations will be answered with a 'no backend' note")

        self._event_queue: asyncio.Queue[VoiceSessionEvent | None] = asyncio.Queue()
        self._closed = False
        self._rows: dict[str, _Row | None] = {"user": None, "assistant": None}
        self._row_timers: dict[str, asyncio.TimerHandle | None] = {"user": None, "assistant": None}
        self._fragments: list[TranscriptFragment] = []
        self._transcript: list[TranscriptEntry] = []
        self._metrics: list[TurnMetrics] = []
        self._turn_index = 0
        self._last_user_end_ms: int | None = None
        self._last_user_text = ""
        self._assistant_turn_first_audio_gap: float | None = None
        self._assistant_turn_speech_bytes = 0

        self._record_audio = record_audio
        self._in_chunks: list[bytes] = []
        self._out_chunks: list[bytes] = []
        self.speech_output_bytes = 0

        self._delegation_lock = asyncio.Lock()
        self._delegation_tasks: set[asyncio.Task] = set()
        self._delegations: dict[str, dict[str, Any]] = {}
        self._last_delegation_fragment_idx = 0
        self._last_run_context: RunContext | None = None
        # Voice-duration accounting across replacement sessions: snapshots are
        # per-session cumulative, so the dropped generations' totals are banked
        # in ``_usage_prior`` and the live one tracked in ``_usage_current``.
        self._usage_prior = 0.0
        self._usage_current: float | None = None
        self._generation_started_at = 0.0
        self.usage_confirmed = True
        """False when a generation ended without ``session.closed``: its seconds
        are the larger of the last snapshot and wall-clock elapsed, not billed truth."""
        self.close_reason: str | None = None
        self.session_id: str | None = None

    @property
    def usage_seconds(self) -> float | None:
        """Total voice seconds this session, all generations (None before any report)."""
        if self._usage_current is None and not self._usage_prior:
            return None
        return round(self._usage_prior + (self._usage_current or 0.0), 3)

    # -- Public: recording -----------------------------------------------------

    @property
    def transcript(self) -> list[TranscriptEntry]:
        return list(self._transcript)

    @property
    def fragments(self) -> list[TranscriptFragment]:
        """Every transcript fragment as received, both speakers, in arrival order."""
        return list(self._fragments)

    @property
    def metrics(self) -> list[TurnMetrics]:
        return list(self._metrics)

    @property
    def input_audio(self) -> bytes:
        return b"".join(self._in_chunks)

    @property
    def output_audio(self) -> bytes:
        return b"".join(self._out_chunks)

    @property
    def delegations(self) -> dict[str, dict[str, Any]]:
        """``delegation_id → {offset_ms, created_at, prompt, result, run_id, error, ...}``."""
        return {k: dict(v) for k, v in self._delegations.items()}

    def transcript_since_last_delegation(self) -> list[TranscriptFragment]:
        """Fragments (both speakers, arrival order) after the previous delegation's snapshot, merged per row."""
        frags = self._fragments[self._last_delegation_fragment_idx :]
        merged: list[TranscriptFragment] = []
        for f in frags:
            if merged and merged[-1].role == f.role and f.start_ms - merged[-1].end_ms <= self.row_gap_ms:
                m = merged[-1]
                merged[-1] = TranscriptFragment(
                    role=m.role, delta=m.delta + f.delta, start_ms=m.start_ms, end_ms=f.end_ms
                )
            else:
                merged.append(f)
        return [
            TranscriptFragment(role=m.role, delta=m.delta.strip(), start_ms=m.start_ms, end_ms=m.end_ms) for m in merged
        ]

    # -- Public: control -------------------------------------------------------

    async def run(self, audio_in: AsyncIterable[bytes]) -> AsyncIterator[VoiceSessionEvent]:
        try:
            started = await self.transport.connect()
            self._generation_started_at = time.monotonic()
            self.session_id = (started.get("session") or {}).get("id")
            await self._seed_call_context()
            await self._emit(SessionStarted())
            if self.greeting:
                await self.greet(self.greeting)

            audio_task = asyncio.create_task(self._forward_audio(audio_in), name="openai-live-uplink")
            events_task = asyncio.create_task(self._process_events(), name="openai-live-events")
            try:
                while True:
                    ev = await self._event_queue.get()
                    if ev is None:
                        break
                    yield ev
            finally:
                for t in (audio_task, events_task):
                    if not t.done():
                        t.cancel()
                await asyncio.gather(audio_task, events_task, return_exceptions=True)
        except Exception as e:
            logger.error("openai_live_session_error", error=str(e), exc_info=True)
            yield SessionError(message=str(e))
        finally:
            await self._cleanup()
            # Drain events queued by the close (row flushes, metrics).
            while not self._event_queue.empty():
                ev = self._event_queue.get_nowait()
                if ev is not None:
                    yield ev
            yield SessionEnded()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._emit(None)

    async def say(self, text: str) -> None:
        """Have the voice model say (paraphrase) ``text`` now — outside any delegation."""
        await self.transport.send({"type": "session.commentary.append", "delegation_id": None, "content": text})

    async def instruct(self, text: str) -> None:
        """Append a session-level instruction (greeting, redirect, guardrail block)."""
        await self.transport.send({"type": "session.instructions.append", "delegation_id": None, "content": text})

    async def context(self, text: str) -> None:
        """Quiet context for the voice model (UI state, facts) — not spoken on append."""
        await self.transport.send({"type": "session.thinking.append", "delegation_id": None, "content": text})

    async def greet(self, instruction: str) -> None:
        """Ask for an opener before the caller speaks (docs recipe: instruction, then a nudge)."""
        await self.instruct(instruction)
        await self.say("Begin the conversation now, following the instructions provided.")

    # -- Internal: uplink ------------------------------------------------------

    async def _forward_audio(self, audio_in: AsyncIterable[bytes]) -> None:
        try:
            async for chunk in audio_in:
                if self._record_audio:
                    self._in_chunks.append(chunk)
                await self.transport.send_audio(chunk)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("openai_live_uplink_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"Audio input error: {e}"))

    # -- Internal: downlink ----------------------------------------------------

    async def _process_events(self) -> None:
        try:
            while True:
                async for ev in self.transport.events():
                    await self._dispatch(ev)
                # Stream ended. Requested close or a server-side final event: done.
                if self._closed or self.transport.finalized:
                    break
                if not await self._try_reconnect():
                    break
        except asyncio.CancelledError:
            return
        except Exception as e:
            logger.error("openai_live_event_error", error=str(e), exc_info=True)
            await self._emit(SessionError(message=f"GPT-Live error: {e}"))
        await self.close()

    async def _try_reconnect(self) -> bool:
        """Replace a dropped session. Returns True when events may resume."""
        if self.reconnect_attempts <= 0:
            await self._emit(SessionError(message="GPT-Live connection lost"))
            return False
        await self._emit(AgentStatus(text="Reconnecting…"))
        # Rows straddling the gap cannot be extended: timelines do not line up.
        for role in ("user", "assistant"):
            await self._close_row(role)
        seed = self.history_items()
        last_error: Exception | None = None
        for attempt in range(self.reconnect_attempts):
            if self._closed:
                return False
            delay = self.reconnect_backoff_secs[min(attempt, len(self.reconnect_backoff_secs) - 1)]
            if delay:
                await asyncio.sleep(delay)
            try:
                started = await self.transport.reconnect(input=seed)
            except NotImplementedError:
                await self._emit(SessionError(message="GPT-Live connection lost (transport cannot reconnect)"))
                return False
            except Exception as e:
                last_error = e
                logger.warning("openai_live_reconnect_failed", attempt=attempt + 1, error=str(e))
                continue
            self._bank_generation_usage()
            self.reconnects += 1
            self._generation += 1
            self._generation_started_at = time.monotonic()
            self._last_user_end_ms = None
            self.session_id = (started.get("session") or {}).get("id")
            logger.info("openai_live_reconnected", session_id=self.session_id, attempt=attempt + 1, seeded=len(seed))
            await self._emit(AgentStatus(text="Reconnected"))
            return True
        await self._emit(SessionError(message=f"GPT-Live reconnect failed: {last_error}"))
        return False

    def _bank_generation_usage(self) -> None:
        """Close the books on a generation that died without ``session.closed``.

        Its last snapshot may be up to one reporting interval stale, so take the
        larger of snapshot and wall-clock elapsed and flag the total unconfirmed.
        """
        elapsed = max(0.0, time.monotonic() - self._generation_started_at) if self._generation_started_at else 0.0
        self._usage_prior += max(self._usage_current or 0.0, elapsed)
        self._usage_current = None
        self.usage_confirmed = False

    def history_items(self, *, max_items: int = 64, max_chars: int = 24_000) -> list[dict[str, Any]]:
        """Transcript so far as ``session.input`` message items (committed rows + open rows).

        Bounded from the tail: the service caps startup history (8,192 tokens).
        """
        entries = list(self._transcript)
        for role in ("user", "assistant"):
            row = self._rows[role]
            if row is not None and row.text.strip():
                entries.append(TranscriptEntry(role=role, text=row.text.strip()))  # type: ignore[arg-type]
        items: list[dict[str, Any]] = []
        budget = max_chars
        for e in reversed(entries[-max_items:]):
            if budget - len(e.text) < 0:
                break
            budget -= len(e.text)
            kind = "input_text" if e.role == "user" else "output_text"
            items.append({"type": "message", "role": e.role, "content": [{"type": kind, "text": e.text}]})
        items.reverse()
        return items

    async def _dispatch(self, ev: dict[str, Any]) -> None:
        typ = ev.get("type")
        if typ == "session.output_audio.delta":
            data = base64.b64decode(ev.get("delta") or "")
            if not data:
                return
            silent = data.count(b"\x00") == len(data)
            if not silent:
                self.speech_output_bytes += len(data)
                if self._rows["assistant"] is not None:
                    self._assistant_turn_speech_bytes += len(data)
            if self._record_audio:
                self._out_chunks.append(data)
            if silent and self.drop_silence:
                return
            await self._emit(AudioOutput(data=data))
        elif typ == "session.input_transcript.delta":
            await self._on_fragment("user", ev)
        elif typ == "session.output_transcript.delta":
            await self._on_fragment("assistant", ev)
        elif typ == "session.delegation.created":
            d = ev.get("delegation") or {}
            did = d.get("id")
            if not did:
                logger.warning("openai_live_delegation_without_id", event=ev)
                return
            if d.get("target", "client") != "client":
                # Responses delegation: OpenAI runs the backend; nothing for us to do
                # beyond surfacing it. Nested ``response.event`` envelopes follow.
                await self._emit(DelegationCreated(delegation_id=did, prompt=""))
                return
            # Cursor captured *here*: the task body runs a tick later, by which time
            # fragments that arrived after the delegation may already be dispatched.
            task = asyncio.create_task(
                self._run_delegation(did, ev.get("offset_ms"), created_frag_idx=len(self._fragments)),
                name=f"live-delegation-{did}",
            )
            self._delegation_tasks.add(task)
            task.add_done_callback(self._delegation_tasks.discard)
        elif typ == "response.event":
            inner = (ev.get("event") or {}).get("type", "")
            if inner == "response.output_text.delta":
                await self._emit(AgentStatus(text=(ev["event"].get("delta") or "")))
        elif typ == "session.usage.updated":
            self._usage_current = (ev.get("usage") or {}).get("seconds", self._usage_current)
        elif typ == "session.closed":
            self._usage_current = (ev.get("usage") or {}).get("seconds", self._usage_current)
            self.close_reason = ev.get("reason")
            if not self._closed and self.close_reason not in (None, "close_requested"):
                # Server-initiated (safety termination, expiry, ...): final, no reconnect.
                await self._emit(SessionError(message=f"GPT-Live session closed by server: {self.close_reason}"))
            await self.close()
        elif typ == "error":
            err = ev.get("error") or {}
            msg = err.get("message") or json.dumps(err) or "GPT-Live error"
            logger.warning("openai_live_error_event", error=err, client_event_id=err.get("client_event_id"))
            await self._emit(SessionError(message=msg))
        # *.appended / session.updated / muted / unmuted: acknowledgments, nothing to surface.

    # -- Internal: transcript rows ---------------------------------------------

    async def _on_fragment(self, role: Literal["user", "assistant"], ev: dict[str, Any]) -> None:
        delta = ev.get("delta") or ""
        if not delta:
            return
        frag = TranscriptFragment(
            role=role, delta=delta, start_ms=int(ev.get("start_ms") or 0), end_ms=int(ev.get("end_ms") or 0)
        )
        self._fragments.append(frag)
        row = self._rows[role]
        if row is not None and frag.start_ms - row.end_ms > self.row_gap_ms:
            await self._close_row(role)
            row = None
        if row is None:
            row = _Row(frag)
            self._rows[role] = row
            if role == "assistant":
                self._begin_assistant_turn(frag)
        else:
            row.add(frag)
        self._arm_row_timer(role)
        if role == "user":
            await self._emit(TranscriptPartial(text=row.text.strip()))
        else:
            await self._emit(AgentTextDelta(text=delta))

    def _arm_row_timer(self, role: str) -> None:
        h = self._row_timers[role]
        if h is not None:
            h.cancel()
        loop = asyncio.get_running_loop()
        self._row_timers[role] = loop.call_later(
            (self.row_gap_ms + self.row_close_slack_ms) / 1000, lambda: asyncio.ensure_future(self._close_row(role))
        )

    async def _close_row(self, role: str) -> None:
        row = self._rows[role]
        if row is None:
            return
        self._rows[role] = None
        h = self._row_timers[role]
        if h is not None:
            h.cancel()
            self._row_timers[role] = None
        text = row.text.strip()
        if role == "user":
            self._last_user_end_ms = row.end_ms
            self._last_user_text = text
            if text:
                self._transcript.append(TranscriptEntry(role="user", text=text))
                await self._emit(TranscriptCommitted(text=text))
        elif text:
            self._transcript.append(TranscriptEntry(role="assistant", text=text))
            await self._emit(AgentTextDone(text=text, run_id=None))
            await self._emit_turn_metrics(row)

    def _begin_assistant_turn(self, first: TranscriptFragment) -> None:
        self._turn_index += 1
        self._assistant_turn_speech_bytes = 0
        gap = None
        if self._last_user_end_ms is not None and first.start_ms >= self._last_user_end_ms:
            gap = float(first.start_ms - self._last_user_end_ms)
        elif self._rows["user"] is not None and first.start_ms >= self._rows["user"].end_ms:
            gap = float(first.start_ms - self._rows["user"].end_ms)
        self._assistant_turn_first_audio_gap = gap

    async def _emit_turn_metrics(self, row: _Row) -> None:
        user_row = self._rows["user"]
        m = TurnMetrics(
            turn_index=self._turn_index,
            user_text_chars=len(user_row.text.strip()) if user_row is not None else len(self._last_user_text),
            eou_to_first_audio_ms=self._assistant_turn_first_audio_gap,
            eou_to_tts_first_byte_ms=self._assistant_turn_first_audio_gap,
            turn_total_ms=float(max(0, row.end_ms - row.start_ms)),
            interrupted=False,
            tts_segments=0,
            audio_bytes=self._assistant_turn_speech_bytes,
        )
        self._metrics.append(m)
        await self._emit(TurnMetricsEvent(metrics=m))

    # -- Internal: delegation ----------------------------------------------------

    async def _run_delegation(self, delegation_id: str, offset_ms: int | None, *, created_frag_idx: int) -> None:
        rec: dict[str, Any] = {
            "offset_ms": offset_ms,
            "generation": self._generation,
            "created_at": time.time(),
            "seq": len(self._delegations),
            # Fragment cursor at *creation* (not at run start): anything the caller
            # says from here on happened while this work was pending.
            "created_frag_idx": created_frag_idx,
            "status": "pending",
        }
        self._delegations[delegation_id] = rec
        if self._delegation_lock.locked():
            with contextlib.suppress(Exception):
                await self.transport.send(
                    {
                        "type": "session.thinking.append",
                        "delegation_id": delegation_id,
                        "content": "Still finishing the previous request; this one is queued.",
                    }
                )
        async with self._delegation_lock:
            prompt = self.delegation_prompt(self, delegation_id)
            self._last_delegation_fragment_idx = len(self._fragments)
            rec.update(prompt=prompt, status="running")
            await self._emit(DelegationCreated(delegation_id=delegation_id, prompt=prompt))
            t0 = time.monotonic()
            text = ""
            run_id: str | None = None
            error: str | None = None
            try:
                if self.on_delegation is not None:
                    text = (await self.on_delegation(self, delegation_id, prompt)) or ""
                elif self.agent is not None:
                    text, run_id = await self._run_agent(delegation_id, prompt)
                else:
                    text = "No backend is connected to handle this request."
            except TimeoutError:
                error = "timeout"
                text = "The lookup is taking too long. Tell the caller you couldn't complete it right now."
            except Exception as e:
                logger.error("openai_live_delegation_failed", delegation_id=delegation_id, error=str(e), exc_info=True)
                error = str(e)
                text = "Something went wrong while handling that. Apologize briefly and offer to try again."
            stale_reason = None if error else self._stale_reason(rec)
            spoken = await self._deliver_result(delegation_id, text, stale_reason)
            rec.update(
                status="error" if error else "done",
                result=text,
                run_id=run_id,
                error=error,
                stale=stale_reason is not None,
                stale_reason=stale_reason,
                spoken=spoken,
                backend_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            await self._emit(
                DelegationResult(
                    delegation_id=delegation_id,
                    text=text,
                    run_id=run_id,
                    error=error,
                    stale=stale_reason is not None,
                    spoken=spoken,
                )
            )

    def _stale_reason(self, rec: dict[str, Any]) -> str | None:
        """Why this result may no longer be wanted, or None.

        Two signals: substantive caller speech after the delegation was created
        (a backchannel — "okay", "sure", "mm-hm" — does not count), or a newer
        delegation created after it (the model asked for something else).
        """
        if self.stale_policy == "speak":
            return None
        newer_user = " ".join(
            f.delta for f in self._fragments[rec["created_frag_idx"] :] if f.role == "user" and f.delta.strip()
        ).strip()
        newer_user = re.sub(r"\s+", " ", newer_user)
        if newer_user and not _BACKCHANNEL.fullmatch(newer_user.lower()):
            return f"the caller has since said: {newer_user!r}"
        if any(o["seq"] > rec["seq"] for o in self._delegations.values()):
            return "a newer request was delegated after this one"
        return None

    async def _deliver_result(self, delegation_id: str, text: str, stale_reason: str | None) -> bool:
        """Append the backend text to the live session. Returns True when sent as commentary."""
        if not text.strip():
            return False
        if stale_reason is None:
            typ = "session.commentary.append"
            chunks = split_for_append(text)
        elif self.stale_policy == "drop":
            logger.info("openai_live_stale_result_dropped", delegation_id=delegation_id, reason=stale_reason)
            return False
        else:
            typ = "session.thinking.append"
            note = (
                f"Late result for an earlier request; {stale_reason}. "
                "Mention it only if it is still what the caller wants; otherwise stay with the current topic. Result: "
            )
            chunks = split_for_append(note + text)
            logger.info("openai_live_stale_result_quiet", delegation_id=delegation_id, reason=stale_reason)
        for i, chunk in enumerate(chunks):
            try:
                await self.transport.send(
                    {
                        "type": typ,
                        "event_id": f"{delegation_id}_result_{i}",
                        "delegation_id": delegation_id,
                        "content": chunk,
                    }
                )
            except Exception as e:
                logger.warning("openai_live_result_append_failed", delegation_id=delegation_id, error=str(e))
                return False
        return stale_reason is None

    async def _run_agent(self, delegation_id: str, prompt: str) -> tuple[str, str | None]:
        assert self.agent is not None
        if self._last_run_context is not None:
            set_run_context(self._last_run_context)
        kwargs: dict[str, Any] = {"prompt": Message(role="user", content=[TextContent(text=prompt)])}
        if self.model:
            kwargs["model"] = self.model
        agen = self.agent(**kwargs)
        text = ""
        final_text: str | None = None
        run_id: str | None = None
        announced: set[str] = set()
        deadline = None if self.delegation_timeout_secs is None else time.monotonic() + self.delegation_timeout_secs
        try:
            while True:
                nxt = agen.__anext__()
                try:
                    if deadline is None:
                        event = await nxt
                    else:
                        event = await asyncio.wait_for(nxt, max(0.0, deadline - time.monotonic()))
                except StopAsyncIteration:
                    break
                tool_name: str | None = None
                if isinstance(event, DeltaEvent) and isinstance(event.item, ToolUse) and event.item.name:
                    tool_name = event.item.name  # streaming providers: earliest signal
                elif isinstance(event, StartEvent) and event.parent_call_id is not None:
                    leaf = str(event.path).rsplit(".", 1)[-1]
                    if leaf != "llm":
                        tool_name = leaf  # non-streaming providers: the tool runnable itself starting
                if tool_name is not None:
                    await self._emit(AgentStatus(text=f"Calling {tool_name}…"))
                    if tool_name not in announced:
                        announced.add(tool_name)
                        with contextlib.suppress(Exception):
                            await self.transport.send(
                                {
                                    "type": "session.thinking.append",
                                    "delegation_id": delegation_id,
                                    "content": f"Working on it: running {tool_name}. No result yet.",
                                }
                            )
                    continue
                if isinstance(event, StartEvent):
                    continue
                elif isinstance(event, InteractionEvent):
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
                    run_id = event.run_id
                    final_text = (
                        "I need a decision from the caller in the app before I can continue. Tell them briefly."
                    )
                elif isinstance(event, ApprovalEvent):
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
                    run_id = event.run_id
                    final_text = (
                        "This action needs the caller's approval in the app before I can continue. Tell them briefly."
                    )
                elif isinstance(event, DeltaEvent) and isinstance(event.item, TextDelta | Text):
                    chunk = event.item.text if isinstance(event.item, Text) else event.item.text_delta
                    text += chunk or ""
                elif isinstance(event, OutputEvent) and event.parent_call_id is None:
                    run_id = run_id or event.run_id
                    if event.status.code == "error" and event.error:
                        raise RuntimeError(event.error.get("message") or "agent run failed")
                    out = event.output
                    if isinstance(out, Message):
                        final_text = final_text or out.collect_text()
                    elif isinstance(out, str) and out:
                        final_text = final_text or out
                    elif out is not None and final_text is None:
                        final_text = (
                            json.dumps(out, default=str)
                            if not hasattr(out, "model_dump_json")
                            else out.model_dump_json()
                        )
        finally:
            with contextlib.suppress(Exception):
                await agen.aclose()
            ctx = get_run_context()
            if ctx is not None and ctx._trace:
                self._last_run_context = ctx
        return (final_text if final_text is not None else text).strip(), run_id

    # -- Internal: run context seed ---------------------------------------------

    async def _seed_call_context(self) -> None:
        # Same contract as VoiceSession._seed_call_context: the parent run and
        # per-call identity must be *on the ambient context* before turn one.
        if not self.parent_run_id and not self.call_context:
            return
        ctx = get_run_context()
        if ctx is None:
            ctx = RunContext(
                parent_id=self.parent_run_id,
                tracing_provider=getattr(self.agent, "tracing_provider", TRACING_UNSET),
            )
        elif self.parent_run_id and ctx.parent_id is None and not ctx._trace:
            ctx.parent_id = self.parent_run_id
        if self.call_context:
            session_data = await ctx.get_session()
            session_data.update(self.call_context)
        set_run_context(ctx)

    # -- Internal: helpers -------------------------------------------------------

    async def _emit(self, ev: VoiceSessionEvent | None) -> None:
        await self._event_queue.put(ev)

    async def _cleanup(self) -> None:
        for role in ("user", "assistant"):
            with contextlib.suppress(Exception):
                await self._close_row(role)
        if self._delegation_tasks:
            # Backend work outlives speech by design, but not the session.
            for t in list(self._delegation_tasks):
                t.cancel()
            await asyncio.gather(*self._delegation_tasks, return_exceptions=True)
        try:
            usage = await self.transport.close()
            if usage and "seconds" in usage:
                self._usage_current = usage["seconds"]
            # The dispatcher is already cancelled here; the final event's reason
            # lands on the transport.
            self.close_reason = self.close_reason or getattr(self.transport, "close_reason", None)
        except Exception as e:
            logger.debug("openai_live_transport_close_failed", error=str(e))
        if self.usage_seconds is not None:
            logger.info(
                "openai_live_session_closed",
                session_id=self.session_id,
                seconds=self.usage_seconds,
                est_usd=round(self.usage_seconds / 60 * PRICE_PER_MINUTE_USD, 4),
                reason=self.close_reason,
            )


class LiveSessionSummary(BaseModel):
    """Serializable end-of-session snapshot (transcript, metrics, delegations, usage)."""

    model_config = ConfigDict(extra="forbid")

    session_id: str | None
    usage_seconds: float | None
    usage_confirmed: bool = True
    reconnects: int = 0
    est_usd: float | None
    close_reason: str | None
    transcript: list[TranscriptEntry]
    delegations: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metrics: list[TurnMetrics] = Field(default_factory=list)

    @classmethod
    def from_session(cls, s: LiveSession) -> LiveSessionSummary:
        return cls(
            session_id=s.session_id,
            usage_seconds=s.usage_seconds,
            usage_confirmed=s.usage_confirmed,
            reconnects=s.reconnects,
            est_usd=None if s.usage_seconds is None else round(s.usage_seconds / 60 * PRICE_PER_MINUTE_USD, 4),
            close_reason=s.close_reason,
            transcript=s.transcript,
            delegations=s.delegations,
            metrics=s.metrics,
        )
