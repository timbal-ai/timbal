"""Observable surface of a voice session: transcript records and events.

Shared by :class:`~timbal.voice.session.VoiceSession` (STT → agent → TTS) and
:class:`~timbal.voice.realtime.RealtimeSession` (speech-to-speech).
"""

from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptEntry(BaseModel):
    """Single entry in the session transcript."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str
    timestamp: float = Field(default_factory=time.time)
    filler: bool = False
    """True for spoken latency-masking phrases ("let me check that") —
    part of what was said on the call, but not real reply text."""


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
    # True → rewrite last user bubble (CONTINUE_TURN merge), don't append another.
    replace: bool = False


class AgentTextDelta(VoiceSessionEvent):
    type: Literal["agent_text_delta"] = "agent_text_delta"
    text: str


class AgentTextDone(VoiceSessionEvent):
    type: Literal["agent_text_done"] = "agent_text_done"
    text: str
    run_id: str | None = None
    """Id of the run that produced this turn. Pass as ``parent_id`` to continue
    the conversation on another transport — together with the suspension events
    this gives a client a current pointer after every turn. ``None`` when no run
    is behind the text: the greeting (spoken before any turn exists), realtime
    sessions (provider-held state, no runs), and a turn that timed out before
    its run started."""


class AgentStatus(VoiceSessionEvent):
    """Non-transcript status for the UI (e.g. tool calls while the mic is idle)."""

    type: Literal["agent_status"] = "agent_status"
    text: str


class AgentInteraction(VoiceSessionEvent):
    """The run suspended waiting for a value (``ask_user``, ``confirm``, ...).

    A suspended voice agent is, from the caller's side, a voice agent that went
    quiet: the run is parked and only an external value restarts it. This is the
    session-level lift of :class:`~timbal.types.events.interaction.InteractionEvent`,
    so a client learns a decision is pending instead of waiting out the turn.

    Resuming is an HTTP call against the paused run, not a voice-channel message:
    ``POST /stream`` with ``parent_id=run_id`` and
    ``resume={interaction_id: value}``.
    """

    type: Literal["agent_interaction"] = "agent_interaction"
    run_id: str
    """Id of the paused run. Pass as ``parent_id`` to resume it — the only
    identifier a voice client gets for the run behind the call."""
    interaction_id: str
    """Key of the ``resume`` map entry that answers this suspension."""
    kind: str
    """Discriminator the frontend uses to pick a renderer (e.g. ``ask_user``)."""
    payload: dict[str, Any] = {}
    """JSON-serializable data describing what the caller must supply."""
    response_schema: dict[str, Any] | None = None
    """Optional JSON Schema the resume value must match. ``None`` accepts any value."""
    tool_call_id: str | None = None
    """The LLM tool_call id that triggered this suspension, to correlate it with a
    tool_use block in the transcript. ``None`` for direct (non-agent) calls."""


class AgentApproval(VoiceSessionEvent):
    """The run suspended on an approval gate.

    Split from :class:`AgentInteraction` — mirroring the run-event split — because
    the resume semantics differ: an approval takes a bool or an ``ApprovalResolution``,
    an interaction takes an arbitrary value, and a client should not have to sniff
    which one it got.

    Resume with ``POST /stream``, ``parent_id=run_id``, ``resume={approval_id: True}``.
    """

    type: Literal["agent_approval"] = "agent_approval"
    run_id: str
    """Id of the paused run. Pass as ``parent_id`` to resume it."""
    approval_id: str
    """Key of the ``resume`` map entry that approves or denies this invocation."""
    kind: str | None = None
    """Renderer discriminator for a rich approval card. ``None`` means render
    generically from ``input`` + ``input_schema``."""
    prompt: str | None = None
    """Human-readable summary. Text fallback for non-rich clients."""
    ui: dict[str, Any] | None = None
    """Structured, presentation-only card data, already redacted."""
    input: Any = None
    """Redacted input that would be passed to the runnable — the *values* for a
    structured approval card."""
    input_schema: dict[str, Any] | None = None
    """JSON Schema of the runnable's parameters. Render ``input`` against this for a
    generic typed form with no per-tool frontend code."""
    description: str | None = None
    """Optional runnable or policy description."""
    tool_call_id: str | None = None
    """The LLM tool_call id that triggered this gate. ``None`` for direct calls."""


class FillerSpoken(VoiceSessionEvent):
    """A latency-masking phrase was spoken while a tool runs.

    Separate from ``AgentTextDelta``/``AgentTextDone`` so clients can render it
    dimmed and never treat it as part of the reply."""

    type: Literal["filler"] = "filler"
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
