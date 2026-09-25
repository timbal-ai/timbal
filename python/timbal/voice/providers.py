"""STT/TTS provider interfaces and their cross-provider configuration.

Implement these to plug a new speech provider into
:class:`~timbal.voice.session.VoiceSession`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Callable
from typing import Any, Literal

import structlog
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from .events import VoiceUsageEvent

logger = structlog.get_logger(__name__)


class VoiceUsageEmitter:
    """Optional usage capability, without changing audio/transcript iterators.

    Listeners are synchronous, non-blocking callbacks (e.g. ``queue.put_nowait``).
    Persist/price events outside the callback. Listener failures are logged and
    isolated from speech; consumers are responsible for durable delivery.
    Other providers remain compatible and emit nothing until implemented.
    """

    def add_usage_listener(self, listener: Callable[[VoiceUsageEvent], None]) -> None:
        if not hasattr(self, "_usage_listeners"):
            self._usage_listeners: list[Callable[[VoiceUsageEvent], None]] = []
        if listener not in self._usage_listeners:
            self._usage_listeners.append(listener)

    def remove_usage_listener(self, listener: Callable[[VoiceUsageEvent], None]) -> None:
        listeners = getattr(self, "_usage_listeners", [])
        if listener in listeners:
            listeners.remove(listener)

    def _emit_usage(self, event: VoiceUsageEvent) -> None:
        for listener in tuple(getattr(self, "_usage_listeners", ())):
            try:
                listener(event.model_copy(deep=True))
            except Exception:
                logger.exception("voice_usage_listener_failed", usage_id=event.usage_id)


class AudioInputConfig(BaseModel):
    """Cross-provider STT configuration; provider-specific knobs go in ``extra``."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    language: str | None = None
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    extra: dict[str, Any] = Field(default_factory=dict)


class AudioOutputConfig(BaseModel):
    """Cross-provider TTS configuration; provider-specific knobs go in ``extra``."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None
    voice: str | None = None
    sample_rate: int = 16_000
    encoding: str = "pcm_s16le"
    extra: dict[str, Any] = Field(default_factory=dict)


class TranscriptEvent(BaseModel):
    """Single event from an STT provider."""

    type: Literal["partial", "committed", "error"]
    text: str


class SpeechToText(VoiceUsageEmitter, ABC):
    """Abstract STT provider.

    Lifecycle: ``connect`` → push audio / consume ``events`` → ``close``.
    """

    native_eou: bool = False
    """Provider-native end-of-utterance detection.

    ``True`` when committed transcripts are real turn boundaries decided
    server-side (Deepgram Flux ``EndOfTurn``, Munsit ``speech_final`` /
    ``UtteranceEnd``). The voice server then forces the ``provider`` turn
    detector and disables the local VAD endpointing fast path — stacking
    Smart Turn HOLD on top of provider EOU is a second turn policy fighting
    the first, and these providers have no force-commit for the fast path
    to trigger anyway.
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


class TextToSpeech(VoiceUsageEmitter, ABC):
    """Abstract TTS provider.

    Lifecycle: ``connect`` → ``synthesize`` (repeatable) → ``close``.
    """

    provider_id: str = "unknown"
    """Config-style provider id (``"elevenlabs"``, ``"munsit"``, ...) — matches
    playground / ``voice_config.tts_provider`` values, not the class name."""

    @abstractmethod
    async def connect(self, config: AudioOutputConfig) -> None: ...

    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...

    def open_stream(self) -> TTSStream | None:
        """Optional capability: a per-reply streaming context.

        Providers supporting incremental text input with prosody continuity
        (ElevenLabs multi-context WS) return a fresh :class:`TTSStream` per
        call, so the session feeds every flush into ONE context. Independent
        segments each get "final sentence" intonation and an audible seam
        between them. Default ``None`` → per-segment ``synthesize``.
        """
        return None

    @abstractmethod
    async def close(self) -> None: ...


def resolve_tts(
    provider: str | None = None,
    *,
    api_key: str | SecretStr | None = None,
) -> TextToSpeech:
    """TTS factory for the voice server — the counterpart of ``resolve_stt``.

    ``provider`` is case-insensitive: ``"elevenlabs"`` (default when empty;
    aliases ``"el"``, ``"11labs"``), ``"munsit"`` (alias ``"faseeh"``), or
    ``"fishaudio"`` (aliases ``"fish"``, ``"fish-audio"``), or ``"deepgram"``
    (alias ``"deepgram-aura"``), or ``"openai"``. Unknown ids raise
    ``ValueError`` so the caller can log and fall back explicitly.

    Provider modules import lazily — selecting Munsit never imports the
    ElevenLabs/Fish WebSocket stacks.
    """
    p = (provider or "").strip().lower() or "elevenlabs"
    if p == "openai":
        from .openai import OpenAIStreamTTS

        return OpenAIStreamTTS(api_key=api_key)
    if p in ("elevenlabs", "el", "11labs"):
        from . import elevenlabs

        return elevenlabs.ElevenLabsStreamTTS(api_key=api_key)
    if p in ("munsit", "faseeh"):
        from . import munsit

        return munsit.MunsitStreamTTS(api_key=api_key)
    if p in ("fishaudio", "fish-audio", "fish"):
        from . import fish_audio

        return fish_audio.FishAudioStreamTTS(api_key=api_key)
    if p in ("deepgram", "deepgram-aura"):
        from .deepgram_tts import DeepgramStreamTTS

        return DeepgramStreamTTS(api_key=api_key)
    raise ValueError(f"Unknown TTS provider: {provider!r}")
