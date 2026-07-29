"""Typed configuration for the voice server.

``Agent(voice_config=...)`` — a dict, callable, or :class:`VoiceConfig` — is
validated against this model at server boot, so a typo'd key fails fast
instead of silently falling back to defaults on the first call. Defaults are
the ElevenLabs realtime stack.

Kept import-light on purpose: the server imports this at module load, while
provider SDKs stay behind ``timbal.voice``'s lazy ``__getattr__``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .ambience import validate_ambient_source

# Override with ELEVENLABS_VOICE_ID / TIMBAL_VOICE_ID (cloned/custom voices
# are account-specific).
DEFAULT_VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"


def _default_stt_extra() -> dict[str, Any]:
    return {
        "commit_strategy": "vad",
        # 100ms is what ElevenLabs' own realtime examples use. 300ms made
        # short replies ("work.", "yes.") transcribe as partials but never
        # commit — the session then stalls until the user speaks again.
        "min_speech_duration_ms": 100,
        "vad_silence_threshold_secs": 1.2,
        "vad_threshold": 0.4,
    }


DEFAULT_FILLER_SYSTEM_PROMPT = (
    "You write the one short phrase a voice assistant says out loud while it looks something "
    "up for the user. Reply with ONLY the phrase: a few natural spoken words, no quotes. "
    "Match the language the user is speaking. Do not answer the question and do not mention "
    'tool or function names — just signal you are on it (like "One sec, let me check that.").'
)


class FillerConfig(BaseModel):
    """Spoken tool-call filler: a short LLM-generated phrase masks tool dead air.

    Generation starts the moment a tool call is detected; the phrase is only
    spoken if the tool is still running after ``delay_secs`` and nothing else
    has been said this turn.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    """``{"enabled": false}`` lets a client switch off a server-default filler
    (plain ``None`` can't cross the override merge — ``None`` means "unset")."""
    system_prompt: str = DEFAULT_FILLER_SYSTEM_PROMPT
    model: Any = None
    """Generator LLM ("provider/model", or a TestModel in tests).
    None → the session's LLM; set something fast/cheap for best latency."""
    delay_secs: float = Field(default=1.0, ge=0.0)
    """Grace period — tools that finish sooner never get a filler."""
    timeout_secs: float = Field(default=5.0, gt=0.0)
    """Generation deadline (after the delay); expiry skips the filler silently."""
    repeat_secs: float | None = Field(default=None, gt=0.0)
    """Re-arm on prolonged silence: if the turn is still running and nothing
    has been spoken for this long since the previous filler, say a short
    follow-up ("still on it…"). ``None`` → one filler per turn, max."""
    max_per_turn: int = Field(default=3, ge=1)
    """Hard cap on fillers per turn (first one included) when repeating."""


class AmbientAudioConfig(BaseModel):
    """Looped background sound mixed into the agent's output.

    Server-side only — never client-settable (``source`` may be a file path,
    and a browser must not point the server at arbitrary files).
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    """Preset name (see ``timbal.voice.ambience.PRESETS``) or audio file path.
    Presets are fetched from the CDN on first use, not at boot."""
    volume: float = Field(default=0.3, ge=0.0, le=1.0)

    @field_validator("source")
    @classmethod
    def _source_is_valid(cls, v: str) -> str:
        validate_ambient_source(v)
        return v


class RecordingConfig(BaseModel):
    """Call-recording knobs. Server-side only — never client-settable."""

    model_config = ConfigDict(extra="forbid")

    dir: str | None = None
    layout: Literal["mixed", "split"] = "mixed"
    bitrate_kbps: int = 32
    on_saved: Any = None
    """Async callable invoked with the ``RecordingResult``. Python-only."""


class VoiceConfig(BaseModel):
    """Cross-transport voice session configuration (WS and WebRTC)."""

    model_config = ConfigDict(extra="forbid")

    stt_provider: str = "elevenlabs"
    stt_model: str = "scribe_v2_realtime"
    tts_model: str = "eleven_flash_v2_5"
    voice: str = DEFAULT_VOICE_ID
    language: str | None = None
    """None → provider auto-detect."""
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    stt_extra: dict[str, Any] = Field(default_factory=_default_stt_extra)
    tts_extra: dict[str, Any] = Field(default_factory=lambda: {"auto_mode": True})
    turn_detector: Any = None
    """Mode name, ``TurnDetector`` instance, or zero-arg factory.
    Clients may only send mode names (see ``select_turn_detector_spec``)."""
    vad_endpointing: bool | None = None
    """None → auto: on when the turn detector exposes an audio EOU model."""
    model: str | None = None
    """Per-session LLM override ("provider/model")."""
    turn_timeout_secs: float | None = None
    """None → ``VoiceSession`` default."""
    turn_timeout_fallback: str | None = None
    """None → ``VoiceSession`` default; "" → no spoken apology on timeout."""
    recording: RecordingConfig | None = None
    ambient: AmbientAudioConfig | None = None
    """None → no background audio."""
    filler: FillerConfig | None = None
    """None → no spoken tool-call fillers. ``{}`` enables with defaults."""
