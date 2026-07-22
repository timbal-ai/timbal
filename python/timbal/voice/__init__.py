"""timbal.voice — voice pipeline: VoiceSession, STT/TTS ABCs, turn detection, metrics, and provider implementations."""

from .eou import (
    AudioEouModel,
    EouPredictor,
    PunctuationEouPredictor,
    TextEouPredictor,
)
from .metrics import (
    TurnMetrics,
    TurnMetricsEvent,
)
from .playback import (
    BufferedPlaybackTracker,
    PlaybackTracker,
)
from .realtime import (
    RealtimeEvent,
    RealtimeModel,
    RealtimeSession,
)
from .session import (
    AgentTextDelta,
    AgentTextDone,
    AudioInputConfig,
    AudioOutput,
    AudioOutputConfig,
    SessionEnded,
    SessionError,
    SessionInterrupted,
    SessionStarted,
    SpeechToText,
    TextToSpeech,
    TranscriptCommitted,
    TranscriptEntry,
    TranscriptEvent,
    TranscriptPartial,
    VoiceSession,
    VoiceSessionEvent,
)
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
    # Lazy: importing smart_turn pulls numpy/onnxruntime (timbal[voice] extra),
    # which must not be required just to import timbal.voice.
    if name == "SmartTurnEouModel":
        from .smart_turn import SmartTurnEouModel

        return SmartTurnEouModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentTextDelta",
    "AgentTextDone",
    "AudioEouModel",
    "AudioInputConfig",
    "AudioOutput",
    "AudioOutputConfig",
    "BufferedPlaybackTracker",
    "CommitAction",
    "CommitDecision",
    "EouPredictor",
    "HeuristicTurnDetector",
    "LexicalTurnDetector",
    "LocalAudioTurnDetector",
    "PartialDecision",
    "PlaybackTracker",
    "ProviderTurnDetector",
    "RawTurnDetector",
    "PunctuationEouPredictor",
    "RealtimeEvent",
    "RealtimeModel",
    "RealtimeSession",
    "SemanticTurnDetector",
    "SessionEnded",
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
    "TurnDetector",
    "TurnMetrics",
    "TurnMetricsEvent",
    "TurnState",
    "VoiceSession",
    "VoiceSessionEvent",
    "resolve_turn_detector",
]
