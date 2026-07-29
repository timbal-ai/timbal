"""timbal.voice — voice pipeline: VoiceSession, STT/TTS ABCs, turn detection, metrics, and provider implementations."""

from .config import (
    AmbientAudioConfig,
    FillerConfig,
    RecordingConfig,
    VoiceConfig,
)
from .endpointing import (
    VadEndpointer,
    endpointing_delay,
)
from .eou import (
    AudioEouModel,
    EouPredictor,
    PunctuationEouPredictor,
    TextEouPredictor,
)
from .events import (
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
from .metrics import (
    TurnMetrics,
    TurnMetricsEvent,
)
from .playback import (
    BufferedPlaybackTracker,
    PlaybackTracker,
)
from .providers import (
    AudioInputConfig,
    AudioOutputConfig,
    SpeechToText,
    TextToSpeech,
    TranscriptEvent,
    TTSStream,
)
from .realtime import (
    RealtimeEvent,
    RealtimeModel,
    RealtimeSession,
)
from .session import VoiceSession
from .turn_detection import (
    CommitAction,
    CommitDecision,
    HeuristicTurnDetector,
    LexicalTurnDetector,
    LocalAudioTurnDetector,
    PartialDecision,
    ProviderTurnDetector,
    RawTurnDetector,
    SemanticTurnDetector,
    TurnDetector,
    TurnState,
    resolve_turn_detector,
)


def __getattr__(name: str):
    # Lazy: importing smart_turn / namo / vad pulls numpy/onnxruntime /
    # transformers (timbal[voice] extra), which must not be required just to
    # import timbal.voice.
    if name in ("DeepgramFluxSTT", "DeepgramNovaSTT", "resolve_stt", "stt_provider_id"):
        from . import deepgram

        return getattr(deepgram, name)
    if name == "ElevenLabsRealtimeSTT":
        from .elevenlabs import ElevenLabsRealtimeSTT

        return ElevenLabsRealtimeSTT
    if name == "SmartTurnEouModel":
        from .smart_turn import SmartTurnEouModel

        return SmartTurnEouModel
    if name == "NamoTextEouPredictor":
        from .namo import NamoTextEouPredictor

        return NamoTextEouPredictor
    if name == "SileroVad":
        from .vad import SileroVad

        return SileroVad
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentStatus",
    "AgentTextDelta",
    "AgentTextDone",
    "AmbientAudioConfig",
    "AudioEouModel",
    "AudioInputConfig",
    "AudioOutput",
    "AudioOutputConfig",
    "BufferedPlaybackTracker",
    "CommitAction",
    "CommitDecision",
    "DeepgramFluxSTT",
    "DeepgramNovaSTT",
    "ElevenLabsRealtimeSTT",
    "EouPredictor",
    "FillerConfig",
    "FillerSpoken",
    "HeuristicTurnDetector",
    "LexicalTurnDetector",
    "LocalAudioTurnDetector",
    "NamoTextEouPredictor",
    "PartialDecision",
    "PlaybackTracker",
    "ProviderTurnDetector",
    "RawTurnDetector",
    "PunctuationEouPredictor",
    "RealtimeEvent",
    "RealtimeModel",
    "RealtimeSession",
    "RecordingConfig",
    "SemanticTurnDetector",
    "SessionEnded",
    "SileroVad",
    "SessionError",
    "SessionInterrupted",
    "SessionStarted",
    "SmartTurnEouModel",
    "SpeechToText",
    "TextEouPredictor",
    "TextToSpeech",
    "TranscriptCommitted",
    "TranscriptEntry",
    "TranscriptEvent",
    "TranscriptPartial",
    "TTSStream",
    "TurnDetector",
    "TurnMetrics",
    "TurnMetricsEvent",
    "TurnState",
    "VadEndpointer",
    "VoiceConfig",
    "VoiceSession",
    "VoiceSessionEvent",
    "endpointing_delay",
    "resolve_stt",
    "resolve_turn_detector",
    "stt_provider_id",
]
