"""Fixtures shared by ``python/tests/server/`` tests."""

from __future__ import annotations

import pytest

from .voice_env import VOICE_ENV_KEYS


@pytest.fixture
def clear_voice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)
