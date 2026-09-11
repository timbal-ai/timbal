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

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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


DEFAULT_GREETING_SYSTEM_PROMPT = (
    "You are about to speak first on a live call: the other person has said nothing yet and is "
    "waiting to hear who this is. Reply with ONLY the words to say out loud — one or two short "
    "spoken sentences, no quotes, no stage directions, no questions the caller has not been "
    "given a chance to answer yet."
)


class GreetingConfig(BaseModel):
    """Agent speaks first: the opening line, before any user speech.

    A ``VoiceSession`` is otherwise purely reactive — it needs an STT commit
    before it produces audio. On an outbound call that means the callee answers
    to silence and pays for the first turn with a "Hello?" (measured live: the
    agent said nothing until prompted, and that turn was then cancelled by the
    callee's second "Hello.", wasting 2.4s).

    Two ways to author the line. ``text`` goes straight to TTS — ~300ms to
    first audio and deterministic wording, which is why it is the primary path.
    ``instructions`` spends a full LLM round-trip (~1.5s) to write it against
    the agent's own system prompt, for openers that must mention who is being
    called or why. ``text`` wins when both are set.
    """

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    """Exact words to say. No LLM turn — synthesized directly."""
    instructions: str | None = None
    """Brief for an LLM-authored opener ("greet them by name, mention the
    appointment"). Ignored when ``text`` is set."""
    interruptible: bool = False
    """``False`` (matching Vapi's ``firstMessageInterruptionsEnabled`` and
    LiveKit's ``on_enter`` reply node) defers barge-in until the opener
    finishes: the caller talking over "this is the clinic calling about…" must
    not truncate the one sentence that says who is on the line. The commit that
    barged in still opens a turn — its reply just queues behind the greeting."""
    delay_ms: int = Field(default=0, ge=0)
    """Silence held before speaking. Telephony connects the media stream the
    moment the carrier answers, which can be before the callee has the handset
    to their ear; a beat here keeps the opener from landing under their "hello?"."""
    after_user_silence_secs: float | None = Field(default=None, gt=0.0)
    """Let the other side speak first; fall back to the opener if they don't.

    Unset → speak the opener as soon as ``delay_ms`` has elapsed (the caller
    dialled in and is waiting to hear who this is). Set → hold the opener and
    listen: if the other side says anything within this many seconds, there is
    no opener — their words open turn one and the agent answers *them* (on a
    call we placed, "hello?" / "yes?" is the normal pickup). If the line stays
    silent that long — a hesitant callee, a voicemail beep, a bad first second
    of audio — speak the opener anyway rather than sit in mutual silence.
    Retell's ``begin_after_user_silence_ms``, ElevenLabs' ``initial_wait_time``.
    Counted from the end of ``delay_ms``."""
    model: Any = None
    """Generator LLM for ``instructions`` ("provider/model", or a TestModel in
    tests). None → the session's LLM. Unused by the ``text`` path."""

    @model_validator(mode="after")
    def _has_something_to_say(self) -> GreetingConfig:
        """Fail fast: an empty greeting block is a config typo, not "no greeting"
        (that is ``greeting=None``, the default)."""
        if not (self.text or "").strip() and not (self.instructions or "").strip():
            raise ValueError("greeting needs 'text' or 'instructions'")
        return self


DEFAULT_USER_IDLE_SYSTEM_PROMPT = (
    "The other person on this live call has gone quiet after your last sentence. Say ONE short "
    "spoken line to check they are still there or to gently move the conversation on — matching the "
    "language of the conversation so far. Reply with only the words to say out loud; no quotes, no "
    "stage directions."
)


class UserIdleConfig(BaseModel):
    """What to do when the user stops talking mid-call.

    A ``VoiceSession`` is reactive: after a reply it waits for the next
    utterance, indefinitely. On a phone that is the difference between a call
    that ends cleanly and one that burns ten minutes of STT on a caller who
    walked away — or, just as common, a transcriber that missed what they said
    so the agent *thinks* it is waiting. Every telephony stack re-engages after
    a few seconds (Vapi ``idleMessages``, Retell ``reminder_trigger_ms``,
    ElevenLabs ``turn_timeout``, Pipecat ``user_idle_timeout``) and hangs up
    after a longer silence.

    The clock starts once the agent has *finished being heard* (playback
    drained, no turn in flight, no tool running) and is reset by any user
    speech. After ``timeout_secs`` the agent speaks a prompt — ``text`` (one
    line, or a list to rotate through) or an LLM-authored line from
    ``instructions`` — at most ``max_count`` times per call. With
    ``hangup_after_secs`` set, that long without a word from the user ends the
    session; the agent's own prompts do not reset it.
    """

    model_config = ConfigDict(extra="forbid")

    timeout_secs: float = Field(default=8.0, gt=0.0)
    """Seconds of user silence, measured from when the agent's last audio has
    drained, before a prompt is spoken. Re-arms after each prompt."""
    text: str | list[str] | None = None
    """The line(s) to say. A list rotates in order; wraps around."""
    instructions: str | None = None
    """Brief for an LLM-authored line, in the agent's own voice and language.
    Ignored when ``text`` is set."""
    max_count: int = Field(default=2, ge=0)
    """Prompts per call, then silence (until ``hangup_after_secs``, if set).
    ``0`` → never prompt; only the hang-up applies."""
    hangup_after_secs: float | None = Field(default=None, gt=0.0)
    """Seconds since the user's last word (or since the call started, if they
    never spoke) after which the session ends. Not reset by our own prompts.
    Unset → never hang up on silence."""
    model: Any = None
    """Generator LLM for ``instructions``. None → the session's LLM."""

    @model_validator(mode="after")
    def _has_something_to_do(self) -> UserIdleConfig:
        has_line = bool(self.instructions and self.instructions.strip()) or bool(
            self.text if isinstance(self.text, str) and self.text.strip() else [t for t in (self.text or []) if str(t).strip()]
        )
        if self.max_count > 0 and not has_line:
            raise ValueError("user_idle needs 'text' or 'instructions' (or max_count=0 with hangup_after_secs)")
        if self.max_count == 0 and self.hangup_after_secs is None:
            raise ValueError("user_idle with max_count=0 does nothing without hangup_after_secs")
        return self

    def line_for(self, count: int) -> str | None:
        """The static line for the ``count``-th prompt (0-based), or ``None`` to generate."""
        if isinstance(self.text, str):
            return self.text.strip() or None
        lines = [str(t).strip() for t in (self.text or []) if str(t).strip()]
        if not lines:
            return None
        return lines[count % len(lines)]


def coerce_greeting(value: Any) -> GreetingConfig | None:
    """Normalize any accepted greeting spelling into a :class:`GreetingConfig`.

    A bare string is the common case ("Hi, thanks for calling Acme.") and the
    *only* thing some override channels can carry — a TeXML/TwiML
    ``<Parameter name="greeting">`` is a string, as is an env var. ``""`` means
    "no greeting", so a per-call override can also switch a server default off.
    """
    if value is None:
        return None
    if isinstance(value, GreetingConfig):
        return value
    if isinstance(value, str):
        return GreetingConfig(text=value) if value.strip() else None
    return GreetingConfig.model_validate(value)


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
    tts_provider: str = "elevenlabs"
    """``"elevenlabs"``, ``"munsit"`` (Arabic; requires ``MUNSIT_API_KEY``), or
    ``"fishaudio"`` (requires ``FISH_API_KEY``)."""
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
    ``None`` (unset) resolves to ``"local"`` — Smart Turn + Namo + Silero VAD
    endpointing — when ``timbal[voice]`` is installed, and degrades to
    ``"lexical"`` without it (an explicit ``"local"`` pin would skip that
    degradation and behave holdless). Clients may only send mode names
    (see ``select_turn_detector_spec``)."""
    vad_endpointing: bool | None = None
    """None → auto: on when the turn detector exposes an audio EOU model."""
    model: str | None = None
    """Per-session LLM override ("provider/model")."""
    turn_timeout_secs: float | None = None
    """None → ``VoiceSession`` default."""
    hello_wait_secs: float | None = Field(default=None, ge=0.0)
    """LiveKit transport: how long to hold the session build for a browser
    caller's config hello (STT/TTS/turn-detector picks on the data channel).
    None → ``TIMBAL_VOICE_HELLO_WAIT_SECS``, else 2.0. Server-side only — the
    hello is the thing being waited for, so it cannot carry this."""
    sip_hello_wait_secs: float | None = Field(default=None, ge=0.0)
    """Same window for a SIP caller. None → ``TIMBAL_VOICE_SIP_HELLO_WAIT_SECS``,
    else 0: a phone has no data channel to say hello on, so the window can
    only expire — measured live as 2s of dead air before the first word. Raise
    it only for a SIP bridge that does deliver a hello, or for a settle beat."""
    turn_timeout_fallback: str | None = None
    """None → ``VoiceSession`` default; "" → no spoken apology on timeout."""
    recording: RecordingConfig | None = None
    ambient: AmbientAudioConfig | None = None
    """None → no background audio."""
    filler: FillerConfig | None = None
    """None → no spoken tool-call fillers. ``{}`` enables with defaults."""
    user_idle: UserIdleConfig | None = None
    """None → wait for the user forever (status quo). See :class:`UserIdleConfig`."""
    greeting: GreetingConfig | None = None
    """None → the session stays reactive (waits for the user to speak first).
    A bare string is shorthand for ``{"text": ...}``; ``""`` means no greeting.
    This is the opener for calls the *other side* started (inbound PSTN, a
    browser joining) and the fallback for outbound — see ``outbound_greeting``."""
    outbound_greeting: GreetingConfig | None = None
    """Opener for calls *we* placed (outbound PSTN), where "thanks for calling"
    is the wrong sentence. Unset → ``greeting`` applies to both directions;
    ``""`` → speak nothing on outbound and wait for the callee's "hello"; a
    string / block → that opener on outbound only. Only a host that knows the
    call direction can apply it (:func:`greeting_for_direction`); the session
    itself never sees a direction."""

    @field_validator("greeting", "outbound_greeting", mode="before")
    @classmethod
    def _coerce_greeting(cls, v: Any) -> Any:
        return coerce_greeting(v)


def greeting_for_direction(config: VoiceConfig, *, outbound: bool) -> GreetingConfig | None:
    """The opener this call should use, given who placed it.

    Inbound (and anything that is not a phone call we dialled) → ``greeting``.
    Outbound → ``outbound_greeting`` when the agent set it — including an
    explicit ``None`` from ``""``, which is how it says "stay quiet" — else
    ``greeting``.
    """
    if outbound and "outbound_greeting" in config.model_fields_set:
        return config.outbound_greeting
    return config.greeting
