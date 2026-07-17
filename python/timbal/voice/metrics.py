"""Per-turn latency metrics for :class:`~timbal.voice.VoiceSession`.

The headline voice metric is ``eou_to_first_audio_ms`` — the time between the
user's end of utterance (committed transcript) and the first byte of TTS audio
emitted back to the client.

All durations are wall-clock milliseconds derived from ``time.monotonic()``
stamps taken inside the session. Fields are ``None`` when the corresponding
stage never happened (e.g. the turn was interrupted before any audio).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from .session import VoiceSessionEvent


class TurnMetrics(BaseModel):
    """Latency and volume measurements for a single conversation turn."""

    model_config = ConfigDict(extra="forbid")

    turn_index: int
    user_text_chars: int
    eou_to_llm_first_token_ms: float | None = None
    """Committed transcript -> first LLM text delta."""
    eou_to_tts_first_byte_ms: float | None = None
    """Committed transcript -> first PCM chunk emitted."""
    eou_to_first_audio_ms: float | None = None
    """The headline number (same stamp as ``eou_to_tts_first_byte_ms``)."""
    llm_total_ms: float | None = None
    """Turn start -> agent generator done."""
    tts_total_ms: float | None = None
    """First TTS segment start -> last TTS segment end."""
    turn_total_ms: float
    interrupted: bool
    tts_segments: int
    audio_bytes: int


class TurnMetricsEvent(VoiceSessionEvent):
    """Emitted once per turn, after ``AgentTextDone`` (or on interruption)."""

    type: Literal["metrics"] = "metrics"
    metrics: TurnMetrics
