"""Shared voice-related test constants for server tests."""

from __future__ import annotations

# Env vars that affect ``default_voice_config_from_env`` / ``merge_voice_config``.
VOICE_ENV_KEYS = (
    "TIMBAL_STT_MODEL",
    "TIMBAL_TTS_MODEL",
    "ELEVENLABS_VOICE_ID",
    "TIMBAL_VOICE_ID",
    "TIMBAL_VOICE_LANGUAGE",
)
