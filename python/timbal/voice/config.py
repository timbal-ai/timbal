"""Typed configuration for the voice server.

``Agent(voice_config=...)`` — a dict, callable, or :class:`VoiceConfig` — is
validated against this model at server boot, so a typo'd key fails fast
instead of silently falling back to defaults on the first call. Defaults are
the ElevenLabs realtime stack.

Kept import-light on purpose: the server imports this at module load, while
provider SDKs stay behind ``timbal.voice``'s lazy ``__getattr__``.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .ambience import validate_ambient_source

# Override with ELEVENLABS_VOICE_ID / TIMBAL_VOICE_ID (cloned/custom voices
# are account-specific).
DEFAULT_VOICE_ID = "1SM7GgM6IMuvQlz2BwM3"

# ``end_call`` confirmation window. None/0 keeps the historical immediate
# hangup. Capped so a client hello cannot pin the line open after goodbye.
HANGUP_GRACE_MAX_SECS = 30.0


def normalize_hangup_grace_secs(value: float | None) -> float:
    """Seconds to wait after a goodbye. ``None`` and ``0`` wait nothing."""
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("hangup_grace_secs must be a number of seconds")
    number = float(value)
    if number != number or number < 0 or number > HANGUP_GRACE_MAX_SECS:
        raise ValueError(f"hangup_grace_secs must be between 0 and {HANGUP_GRACE_MAX_SECS:g} seconds")
    return number


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
    "Do not answer the question and do not mention "
    'tool or function names — just signal you are on it (like "One sec, let me check that.").'
)


# Flux multilingual accepts these primaries. Locale subtags (``es-ES``) map
# to the primary. Catalan is not in the set. ``flux-general-en`` rejects hints.
FLUX_MULTI_LANGUAGES = frozenset({"en", "es", "fr", "de", "hi", "ru", "pt", "ja", "it", "nl"})
KEYTERMS_MAX = 100
_KEYTERM_WEIGHT = re.compile(r".+:-?\d+(?:\.\d+)?\s*$")


def validate_keyterms(value: Any) -> list[str]:
    """Plain phrases for speech recognition. No weights, at most 100."""
    if value is None:
        return []
    if isinstance(value, str):
        raise TypeError("keyterms must be a list of phrases, not one string")
    if not isinstance(value, (list, tuple)):
        raise TypeError("keyterms must be a list of phrases")
    if len(value) > KEYTERMS_MAX:
        raise ValueError(f"keyterms accepts at most {KEYTERMS_MAX} phrases")
    terms: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("each keyterm must be a string")
        text = item.strip()
        if not text:
            raise ValueError("keyterms cannot contain a blank phrase")
        if _KEYTERM_WEIGHT.fullmatch(text):
            raise ValueError(f"keyterms are plain phrases, not weighted keywords: {text!r}")
        terms.append(text)
    return terms


def validate_stt_languages(value: Any) -> list[str]:
    """Expected Flux multilingual hints. Does not set the reply language."""
    if value is None:
        return []
    if isinstance(value, str):
        raise TypeError("stt_languages must be a list of BCP-47 tags, not one string")
    if not isinstance(value, (list, tuple)):
        raise TypeError("stt_languages must be a list of BCP-47 tags")
    from .language import normalize_language_tag

    tags: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("each stt_languages entry must be a string")
        tag = normalize_language_tag(item)
        if tag is None:
            raise ValueError(f"stt_languages entries must be BCP-47 tags, got {item!r}")
        primary = language_key(tag)
        if primary not in FLUX_MULTI_LANGUAGES:
            raise ValueError(
                f"{tag} is not a Deepgram Flux transcription language. "
                "Flux multilingual accepts en, es, fr, de, hi, ru, pt, ja, it, and nl. "
                "Catalan is not among them. Set reply_language separately if the reply should use another language."
            )
        tags.append(tag)
    return tags


def _flux_multi(provider: str, model: str) -> bool:
    if provider not in ("deepgram", "deepgram-flux"):
        return False
    chosen = (model or "flux-general-multi").strip()
    if chosen.startswith("flux-general-en"):
        return False
    if chosen.startswith("nova") or chosen.startswith("scribe") or chosen.startswith("gpt-"):
        return False
    return True


def _accepts_keyterms(provider: str, model: str) -> bool:
    if provider in ("deepgram", "deepgram-flux"):
        chosen = (model or "").strip()
        return not chosen.startswith("nova-2")
    if provider == "deepgram-nova":
        chosen = (model or "nova-3").strip()
        return chosen.startswith("nova-3") or chosen == ""
    if provider == "elevenlabs":
        return True
    return False


def language_key(code: str | None) -> str:
    """``"es-ES"`` / ``"es_es"`` / ``" ES "`` → ``"es"``; ``None`` → ``""``."""
    return (code or "").strip().lower().replace("_", "-").split("-")[0]


class FillerConfig(BaseModel):
    """Spoken tool-call filler: a short phrase masks tool silence.

    Generation starts the moment a tool call is detected; the phrase is only
    spoken if the tool is still running after ``delay_secs`` and nothing else
    has been said this turn. ``think_delay_secs`` extends the same cover to the
    silence before the model has said or called anything, and ``phrases``
    makes the first filler instant instead of an LLM round trip.

    ``soft_timeout_secs`` is a separate, opt-in progress line for a tool that
    is still running after the caller has heard no agent audio for that long
    (ElevenLabs ``soft_timeout_config``, LiveKit ``ctx.with_filler(delay=)``).
    Unset, tool waits keep the rules above. A hangup tool (``end_call``,
    ``hangup``, or any call already inside ``commit_hangup``) does not get one.
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
    follow-up ("still on it…"). ``None`` → one filler per turn, except that a
    tool called after a ``think_delay_secs`` filler gets one of its own."""
    max_per_turn: int = Field(default=3, ge=1)
    """Hard cap on fillers per turn (first one included)."""
    think_delay_secs: float | None = Field(default=None, ge=0.0)
    """Cover the model's own silence, not only a tool's. Set → when this long
    after the turn began nothing has been said and no tool has been called
    (a reasoning model thinking, a slow first token), the first filler is
    spoken anyway. A tool call before then still gets its filler after
    ``delay_secs``, whichever comes first. A tool called after that filler
    gets its own, ``delay_secs`` after the later of the call and the filler,
    even with ``repeat_secs`` unset. ``None`` → fillers start only with a
    tool call."""
    phrases: list[str] | dict[str, list[str]] | None = None
    """Ready-made lines for the *first* filler of a turn ("Mm, let me see."),
    spoken with no LLM round trip, so it can land within ``delay_secs``
    exactly. A dict is keyed by the call's language (``"es"``, ``"en"``;
    region dropped on both sides, so ``"es-ES"`` is ``"es"``) with ``"*"``
    serving any other; a language with no entry falls back to generation.
    The language is ``AudioInputConfig.language`` as configured, not one the
    STT detects: with it unset (auto-detect) only ``"*"`` is ever used.
    Follow-ups are always generated, with the request in hand, so they can
    say what the assistant is working on. The line said on the previous turn
    is avoided when there is another to choose."""
    soft_timeout_secs: float | None = Field(default=None, ge=0.0)
    """Progress line during a tool wait. ``None`` (default) leaves tool waits
    on the rules above — ElevenLabs' ``timeout_seconds: -1`` (disabled).

    Set → once a tool call is in flight and the caller has heard no agent
    audio for this many seconds, speak one progress line and keep waiting.
    The wait is continuous silence: audio already playing (a preamble, the
    answer, an earlier filler) resets it, matching LiveKit
    ``ctx.with_filler(delay=)``. A tool that returns first stays silent.
    Their recommended range is 0.5–8 seconds; any non-negative value is
    accepted so a session can sit closer to its own tool latency.
    ``-1`` is accepted and means unset."""
    message: str | None = None
    """Exact progress line for ``soft_timeout_secs``, spoken with no extra
    model call. ElevenLabs ``message``; LiveKit ``with_filler("...")``.
    Blank is dropped. ``None`` uses ``phrases`` when one matches the call's
    language, otherwise the filler generator (their
    ``use_llm_generated_message``). Ignored unless ``soft_timeout_secs`` is set."""

    @field_validator("soft_timeout_secs", mode="before")
    @classmethod
    def _disabled_soft_timeout(cls, v: Any) -> Any:
        # ElevenLabs uses -1 for "disabled". Omit / None is the same switch here.
        # Env vars arrive as strings, before coercion, so "-1" must count too.
        if isinstance(v, str):
            v = v.strip()
        if v in (-1, -1.0, "-1", "-1.0"):
            return None
        return v

    @field_validator("message")
    @classmethod
    def _strip_message(cls, v: str | None) -> str | None:
        if v is None:
            return None
        return v.strip() or None

    @field_validator("phrases")
    @classmethod
    def _drop_blank_phrases(cls, v: list[str] | dict[str, list[str]] | None) -> list[str] | dict[str, list[str]] | None:
        if v is None:
            return None
        if isinstance(v, dict):
            kept: dict[str, list[str]] = {}
            for k, lines in v.items():
                if key := language_key(k):
                    kept.setdefault(key, []).extend(p.strip() for p in lines if p.strip())
            return {k: lines for k, lines in kept.items() if lines} or None
        return [p.strip() for p in v if p.strip()] or None


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
    "spoken line to check they are still there or to gently move the conversation on. Reply with "
    "only the words to say out loud; no quotes, no stage directions."
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
            self.text
            if isinstance(self.text, str) and self.text.strip()
            else [t for t in (self.text or []) if str(t).strip()]
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
    if value is None or (isinstance(value, dict) and not value):
        return None
    if isinstance(value, GreetingConfig):
        return value
    if isinstance(value, str):
        return GreetingConfig(text=value) if value.strip() else None
    if isinstance(value, dict) and _greeting_block_is_cleared(value):
        return None
    return GreetingConfig.model_validate(value)


def _greeting_block_is_cleared(block: dict[str, Any]) -> bool:
    """A block that *names* ``text``/``instructions`` but leaves both blank.

    That is what a form emits when the user empties the greeting field —
    ``{"text": ""}`` — and it means the same as ``""``: no opener. Seen live
    (2026-09-13): a Studio publish wrote ``{"text": ""}`` and the agent then
    refused to boot for every inbound call, which the caller experienced as
    endless ringing. Fail-fast still applies to a block with *no* speech key
    at all (``{"delay_ms": 500}``): that is a typo, not a cleared field.
    """
    if not any(k in block for k in ("text", "instructions")):
        return False
    return all(not str(block.get(k) or "").strip() for k in ("text", "instructions"))


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


class SpeculativeGeneration(BaseModel):
    """Start the reply before the turn is confirmed. Off unless ``enabled``.

    LiveKit Agents' ``PreemptiveGenerationOptions``, with Timbal's triggers:
    Smart Turn scoring a pause as probably complete, or Deepgram Flux
    ``EagerEndOfTurn``. Audio stays buffered until the committed transcript
    matches. See :mod:`timbal.voice.speculative`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    """``False`` (the default) waits for a confirmed transcript, as before."""
    preemptive_tts: bool = False
    """Also synthesize speech before confirmation. Off by default: the model
    runs early and synthesis starts once the turn is confirmed, so a resumed
    utterance does not spend a TTS request. Audio is held either way."""
    max_speech_duration: float = Field(default=10.0, gt=0)
    """Skip speculation once the caller has been speaking longer than this.
    Long utterances change more often. LiveKit's default is 10 seconds."""
    max_retries: int = Field(default=3, ge=0)
    """Speculative attempts per caller turn. The counter resets when a turn
    is confirmed. LiveKit's default is 3."""
    eou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    """Smart Turn ``P(complete)`` at or above which a pause starts a
    speculative reply. 0.5 is the model's complete/incomplete boundary.
    A pause the text scorer calls incomplete does not start one, even above
    this."""


class VoiceEventsConfig(BaseModel):
    """Event turns (see ``VoiceSession.notify``). Server-side only — never client-settable."""

    model_config = ConfigDict(extra="forbid")

    announce_background_results: bool = False
    """Start a turn when one of the agent's background tasks finishes, instead of
    leaving the result for the caller's next turn."""
    client_notify: bool = False
    """Accept ``notify`` frames from the caller's device (WebRTC data channel,
    the LiveKit caller's participant). Their text is screened by the agent's
    input guardrails and labelled for the model as information, not instructions."""
    on_busy: Literal["reject", "enqueue"] = "reject"
    """Another channel's turn on the call's conversation (a text turn chained on
    it) while a call turn runs: refused, or queued until that turn ends."""


_OPTIONAL_VOICE_BLOCKS = {
    "greeting": GreetingConfig,
    "outbound_greeting": GreetingConfig,
    "user_idle": UserIdleConfig,
    "ambient": AmbientAudioConfig,
    "filler": FillerConfig,
    "recording": RecordingConfig,
    "events": VoiceEventsConfig,
    "speculative_generation": SpeculativeGeneration,
}


def omit_empty_voice_blocks(config: dict[str, Any]) -> dict[str, Any]:
    """Treat empty optional behavior objects as absent, without mutating input.

    Forms and generated configs may include these objects before a behavior is
    configured. Omission preserves inherited settings, including the outbound
    opener's fallback to the inbound greeting. Nonempty blocks still validate.
    A typed block with no explicitly set fields is empty too, so Python objects
    and their sparse JSON representation behave alike. Provider extras are not
    behavior switches and retain their own merge semantics.
    """
    return {
        key: value
        for key, value in config.items()
        if not (
            key in _OPTIONAL_VOICE_BLOCKS
            and (
                (isinstance(value, dict) and not value)
                or (isinstance(value, _OPTIONAL_VOICE_BLOCKS[key]) and not value.model_fields_set)
            )
        )
    }


class VoiceConfig(BaseModel):
    """Cross-transport voice session configuration (WS and WebRTC)."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _omit_empty_blocks(cls, value: Any) -> Any:
        # Remove keys before validation so {} does not become an explicit None
        # in model_fields_set (which would disable the outbound fallback).
        return omit_empty_voice_blocks(value) if isinstance(value, dict) else value

    stt_provider: str = "deepgram"
    """Deepgram Flux (``DEEPGRAM_API_KEY``), which owns end-of-turn detection.
    Without that key a session falls back to ElevenLabs Scribe. Also supports
    ElevenLabs, Deepgram Nova, Munsit, and OpenAI (``OPENAI_API_KEY``)."""
    stt_model: str = "flux-general-multi"
    """Another provider's model id is ignored (that provider's default is used)."""
    tts_provider: str = "elevenlabs"
    """``"elevenlabs"``, ``"munsit"`` (Arabic; requires ``MUNSIT_API_KEY``), or
    ``"fishaudio"`` (requires ``FISH_API_KEY``), or ``"deepgram"``
    (Aura/Aura-2; requires ``DEEPGRAM_API_KEY``). For Deepgram, set
    ``tts_model`` to a full voice model id, e.g. ``aura-2-thalia-en``.
    ``"openai"`` uses ``OPENAI_API_KEY`` and defaults to ``gpt-4o-mini-tts``.
    OpenAI STT/TTS require ``timbal[voice]`` for rates other than 24 kHz."""
    tts_model: str = "eleven_v4_turbo"
    voice: str = DEFAULT_VOICE_ID
    language: str | None = None
    """Conversation language (BCP-47, e.g. ``"en"``, ``"es"``, ``"es-ES"``).

    When set, the speech-to-text adapter receives it as a language hint and,
    unless ``reply_language`` says otherwise, the text model is told to reply
    in that language. ``None`` leaves detection to the provider and tells the
    model to follow the user's latest substantive message, unless the agent's
    own prompt already names one reply language. It does not select a TTS
    voice. Change it on a live call with
    :meth:`timbal.voice.session.VoiceSession.set_language`.
    """
    keyterms: list[str] = Field(default_factory=list)
    """Phrases the speech provider should prefer: names, brands, acronyms.

    Plain strings, at most 100, with no Nova keyword weights (``term:0.15``).
    Set on the agent. A caller hello cannot set it, and it is not taken from
    the caller's words or from an unheard reply. Deepgram Flux and Nova-3
    and ElevenLabs Scribe accept them when the socket connects. Flux can
    replace the whole list on the open socket. Empty leaves recognition
    unchanged. This does not rewrite a transcript and does not promise the
    words will be recognized.
    """
    stt_languages: list[str] | None = None
    """Expected transcription languages for Deepgram ``flux-general-multi``.

    Unset keeps today's single ``language`` hint. A list replaces that hint
    for speech-to-text only. ``reply_language`` is unchanged. Catalan and
    any other language outside Flux's ten are rejected. ``flux-general-en``
    cannot take a hint.
    """
    reply_language: str | None = None
    """Reply policy, separate from the speech-to-text hint.

    Unset (the default) keeps the historical rule: the reply pin is
    ``language``. ``"follow"`` follows the latest substantive message even
    when ``language`` is set for transcription. A BCP-47 tag pins the reply
    to that tag and does not change the hint ``language`` sends to
    speech-to-text. It does not change behavior when left unset.
    """
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    stt_extra: dict[str, Any] = Field(default_factory=_default_stt_extra)
    tts_extra: dict[str, Any] = Field(default_factory=lambda: {"auto_mode": True})
    ignore_backchannels: bool = False
    """Drop pure acknowledgements while the assistant is speaking and for one
    second after playback. English and Spanish ("yeah", "vale") and Catalan
    ("d'acord", "entesos", "molt be") do not interrupt. "no", "stop", "para",
    "espera", and "sí, pero…" still do. Server-side only. Off leaves
    interruptions unchanged."""
    hold_unfinished_secs: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """With provider turn detection, wait this long when a commit ends on a
    trailing conjunction, preposition, determiner, or filler, then merge the
    follow-up. None leaves those commits as turns. Ignored for other detectors.
    Server-side only."""
    turn_detector: Any = None
    """Mode name, ``TurnDetector`` instance, or zero-arg factory.
    ``None`` (unset) resolves to ``"provider"`` on an STT with native
    end-of-turn (Deepgram Flux, the default; Munsit). On any other STT it
    resolves to ``"local"`` — Smart Turn + Namo + Silero VAD endpointing —
    when ``timbal[voice]`` is installed, and degrades to ``"lexical"`` without
    it (an explicit ``"local"`` pin would skip that degradation and behave
    holdless). Clients may only send mode names (see
    ``select_turn_detector_spec``)."""
    vad_endpointing: bool | None = None
    """None → auto: on when the turn detector exposes an audio EOU model."""
    min_delay_secs: float = Field(default=0.45, ge=0.0, le=3.0)
    """Extra silence after a confident end-of-turn score before local
    endpointing force-commits. Scoring overlaps this wait, so with the
    default and ``stop_silence_secs`` a confident turn commits about 0.65
    seconds after speech stops. Lower scores wait longer, up to 3 seconds.
    LiveKit's turn detector uses about 0.3–0.5 seconds; 0.3 here would commit
    at about 0.5 seconds and cut callers who pause for roughly 0.6 seconds
    and then continue, so the default stays 0.45. Set on the agent. A caller
    hello cannot change it. Ignored when endpointing is off, and a
    ``VadEndpointer`` passed straight to ``VoiceSession`` keeps its own value."""
    stop_silence_secs: float = Field(default=0.20, ge=0.0, le=2.0)
    """How long the line must be quiet before the end-of-turn model is scored.
    The default 0.2 seconds is the usual speech-stop gap; shorter hears a
    breath as the end of the turn. Added to ``min_delay_secs`` on a confident
    commit. Same exposure rules as ``min_delay_secs``."""
    first_flush_min_words: int | None = Field(default=None, ge=1, le=40)
    """Unset → the first spoken segment waits for a sentence, like every later
    segment. A whole number from 1 to 40 lets that first segment start at a
    comma once it has that many words. 4 is the useful value: long enough to
    speak, early enough in a long opening sentence. A caller hello cannot set it."""
    predict_during_speech: bool = False
    """Score the turn while the caller is still speaking.

    Off by default. Without a native end-of-turn provider, a finished
    sentence whose audio agrees is confirmed after 0.16 s of silence instead
    of the usual 0.20 s pause plus the 0.45 s floor. On Deepgram Flux, an
    EagerEndOfTurn at or above the eager threshold can start the turn;
    TurnResumed cancels it and EndOfTurn confirms it. Local VAD force-commit
    stays off for Flux."""
    entity_patience: bool = False
    """Wait out numbers, spelled letters, emails, and alphanumeric codes.

    Off by default. While a partial or commit still ends in one of those, the
    session holds the turn for ``entity_patience_secs`` and, on Deepgram Flux,
    ignores eager end-of-turn for that stretch. A word after the entity, a
    complete email, a trailing ``?`` or ``!``, or the patience window closing
    resumes normal endpointing.
    """
    entity_patience_secs: float = Field(default=2.5, gt=0.0, le=10.0)
    """Silence budget while an entity is still open. Ignored when
    ``entity_patience`` is off."""
    model: str | None = None
    """Per-session LLM override ("provider/model")."""
    tool_beat_secs: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    """Minimum silence after pre-tool audio before post-tool synthesis.

    None or zero disables. Tool execution overlaps the speech and the beat;
    tools with no preceding speech never delay the first spoken word.
    """
    max_tts_pause_secs: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    """Cap internal silence in synthesized PCM. ``None`` (default) leaves the
    provider audio unchanged. A quiet stretch longer than this, judged by
    ``tts_quiet_dbfs``, is shortened to this length. The first bytes of a
    reply are not held back. Requires the ``timbal[voice]`` extra when set."""
    tts_quiet_dbfs: float = Field(default=-45.0, allow_inf_nan=False)
    """Level below which PCM counts as silence for ``max_tts_pause_secs``.
    -45 dBFS keeps breaths. Unused while pause trimming is off."""
    tts_tempo: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    """Pitch-preserving speaking rate. ``None`` (default) and ``1.0`` leave
    the audio unchanged; ``1.1`` is a little faster. Requires the
    ``timbal[voice]`` extra when it is not 1. Applied after pause trimming."""
    hangup_grace_secs: float | None = Field(default=None, ge=0, le=HANGUP_GRACE_MAX_SECS)
    """Seconds to keep the line up after a goodbye has been heard, before
    ``end_call`` tears the session down.

    Unset or ``0`` hangs up as soon as that audio has played. A positive value
    is a confirmation window: any further caller speech cancels the hangup
    and the call continues; silence through the window hangs up. Speech is a
    new transcript or audible caller audio, not a list of phrases. One call
    can override it with ``end_call(grace_secs=...)``.
    """

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
    """Empty/unset → inherit recording policy; an empty object does not enable it."""
    events: VoiceEventsConfig | None = None
    """None → ``notify()`` from server code only; background results wait for the
    caller. See :class:`VoiceEventsConfig`."""
    ambient: AmbientAudioConfig | None = None
    """None → no background audio. An empty object is treated as unset."""
    filler: FillerConfig | None = None
    """Empty/unset → no override. Use ``{"enabled": true}`` to enable default fillers,
    or ``{"enabled": false}`` to explicitly disable inherited fillers."""
    user_idle: UserIdleConfig | None = None
    """None → wait for the user forever. ``{}`` is unset. See :class:`UserIdleConfig`."""
    greeting: GreetingConfig | None = None
    """None → the session stays reactive (waits for the user to speak first).
    A bare string is shorthand for ``{"text": ...}``; ``""`` means no greeting.
    This is the opener for calls the *other side* started (inbound PSTN, a
    browser joining) and the fallback for outbound — see ``outbound_greeting``."""
    speculative_generation: SpeculativeGeneration = Field(default_factory=SpeculativeGeneration)
    """Off unless ``enabled`` is set. ``True`` turns it on with the defaults.
    Server-side: a browser hello cannot enable it. See
    :class:`SpeculativeGeneration`."""
    outbound_greeting: GreetingConfig | None = None
    """Opener for calls *we* placed (outbound PSTN), where "thanks for calling"
    is the wrong sentence. Unset → ``greeting`` applies to both directions;
    ``{}`` → inherit, just like unset; ``""`` → speak nothing on outbound; a
    string / block → that opener on outbound only. Only a host that knows the
    call direction can apply it (:func:`greeting_for_direction`); the session
    itself never sees a direction."""

    @field_validator("keyterms", mode="before")
    @classmethod
    def _keyterms_are_phrases(cls, value: Any) -> Any:
        return validate_keyterms(value)

    @field_validator("stt_languages", mode="before")
    @classmethod
    def _stt_languages_are_tags(cls, value: Any) -> Any:
        if value is None:
            return None
        return validate_stt_languages(value)

    @model_validator(mode="after")
    def _transcription_matches_provider(self) -> VoiceConfig:
        provider = (self.stt_provider or "").strip()
        model = (self.stt_model or "").strip()
        extra = self.stt_extra if isinstance(self.stt_extra, dict) else {}
        if self.keyterms and any(key in extra for key in ("keyterm", "keyterms", "keywords", "hotwords")):
            raise ValueError(
                "Set vocabulary on keyterms. Do not also set keyterm, keyterms, keywords, or hotwords in stt_extra."
            )
        if self.keyterms and model.startswith("nova-2"):
            raise ValueError("Nova-2 does not accept keyterms. Use nova-3, or leave keyterms unset.")
        if self.keyterms and not _accepts_keyterms(provider, model):
            raise ValueError(
                f"{provider or 'this speech provider'} {model or ''}".strip()
                + " does not accept keyterms. Use Deepgram Flux, Nova-3, or ElevenLabs Scribe."
            )
        if self.stt_languages is not None and not _flux_multi(provider, model):
            raise ValueError("stt_languages is only supported on Deepgram flux-general-multi.")
        if _flux_multi(provider, model) and self.language:
            from .language import normalize_language_tag

            tag = normalize_language_tag(self.language)
            if tag is not None and language_key(tag) not in FLUX_MULTI_LANGUAGES:
                raise ValueError(
                    f"{tag} is not a Deepgram Flux transcription language. "
                    "Use a supported hint (en, es, fr, de, hi, ru, pt, ja, it, nl) "
                    "and set reply_language if the reply should be in another language."
                )
        return self

    @field_validator("reply_language", mode="before")
    @classmethod
    def _reply_language_is_explicit(cls, value: Any) -> Any:
        if value is None:
            return None
        from .language import parse_reply_language

        return parse_reply_language(value)

    @field_validator("min_delay_secs", "stop_silence_secs", "first_flush_min_words", mode="before")
    @classmethod
    def _reject_bool_seconds(cls, v: Any) -> Any:
        # ``True`` is an int. A JSON ``true`` must not become 1.
        if isinstance(v, bool):
            raise ValueError("must be a number, not a boolean")
        return v

    @field_validator("greeting", "outbound_greeting", mode="before")
    @classmethod
    def _coerce_greeting(cls, v: Any) -> Any:
        return coerce_greeting(v)

    @field_validator("speculative_generation", mode="before")
    @classmethod
    def _coerce_speculative(cls, v: Any) -> Any:
        if v is True:
            return {"enabled": True}
        if v is False:
            return {"enabled": False}
        # A dict is the feature turning on. ``{"enabled": false}`` is how a
        # caller keeps the other knobs and leaves it off. An empty dict is
        # removed before this runs (see omit_empty_voice_blocks).
        if isinstance(v, dict) and "enabled" not in v:
            return {**v, "enabled": True}
        return v

    @field_validator("hangup_grace_secs", mode="before")
    @classmethod
    def _hangup_grace_not_bool(cls, v: Any) -> Any:
        # ``bool`` is an ``int``; without this, ``True`` becomes ``1.0``.
        if isinstance(v, bool):
            raise ValueError("hangup_grace_secs must be a number of seconds")
        return v


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
