"""timbal.voice — voice pipeline: VoiceSession, STT/TTS ABCs, turn detection, metrics, and provider implementations."""

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
    TTSStream,
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
    # Lazy: importing smart_turn / namo / vad pulls numpy/onnxruntime /
    # transformers (timbal[voice] extra), which must not be required just to
    # import timbal.voice.
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
    "NamoTextEouPredictor",
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
    "VoiceSession",
    "VoiceSessionEvent",
    "endpointing_delay",
    "resolve_turn_detector",
]
