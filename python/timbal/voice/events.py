"""Observable surface of a voice session: transcript records and events.

Shared by :class:`~timbal.voice.session.VoiceSession` (STT → agent → TTS) and
:class:`~timbal.voice.realtime.RealtimeSession` (speech-to-speech).
"""

from __future__ import annotations

import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptEntry(BaseModel):
    """Single entry in the session transcript."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    text: str
    timestamp: float = Field(default_factory=time.time)
    # TODO(tool-filler): re-add `filler: bool = False` when tool-call filler
    # speech returns, so transcript entries for spoken latency-masking phrases
    # ("let me check that") are distinguishable from real reply text.


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


class AgentStatus(VoiceSessionEvent):
    """Non-transcript status for the UI (e.g. tool calls while the mic is idle)."""

    type: Literal["agent_status"] = "agent_status"
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
