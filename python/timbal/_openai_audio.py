"""Dependency-free OpenAI audio model and voice constraints shared by adapters/tools."""

STT_MODELS = frozenset(
    {
        "gpt-transcribe",
        "whisper-1",
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "gpt-4o-mini-transcribe-2025-03-20",
        "gpt-4o-mini-transcribe-2025-12-15",
    }
)
TTS_MODELS = frozenset(
    {
        "gpt-4o-mini-tts",
        "gpt-4o-mini-tts-2025-03-20",
        "gpt-4o-mini-tts-2025-12-15",
        "tts-1",
        "tts-1-1106",
        "tts-1-hd",
        "tts-1-hd-1106",
    }
)
VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
    "marin",
    "cedar",
)
LEGACY_VOICES = frozenset({"alloy", "ash", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer"})


def is_openai_stt_model(model: str | None) -> bool:
    """Recognize the provider family, including unsupported models, for routing."""
    return (
        (model or "")
        .strip()
        .startswith(
            (
                "whisper-",
                "gpt-4o-transcribe",
                "gpt-4o-mini-transcribe",
                "gpt-transcribe",
                "gpt-live-transcribe",
                "gpt-realtime-whisper",
            )
        )
    )


def validate_speech(model: str, voice: str, instructions: str | None = None) -> None:
    if model not in TTS_MODELS:
        raise ValueError(f"Unsupported OpenAI speech model: {model!r}")
    voices = LEGACY_VOICES if model.startswith("tts-1") else VOICES
    if voice not in voices:
        raise ValueError(f"Voice {voice!r} is not supported by {model}. Choose from: {', '.join(sorted(voices))}")
    if instructions and model.startswith("tts-1"):
        raise ValueError("Speech instructions require gpt-4o-mini-tts.")
