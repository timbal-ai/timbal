"""Fixtures shared by ``python/tests/server/`` tests."""

from __future__ import annotations

import pytest
from timbal.server import capacity

from .voice_env import VOICE_ENV_KEYS


@pytest.fixture
def clear_voice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for k in VOICE_ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


@pytest.fixture(autouse=True)
def _isolate_voice_capacity():
    """The session cap is a process-wide counter, so a test that leaves a
    session dangling would otherwise shrink the box for every later test."""
    capacity.reset_for_tests()
    yield
    capacity.reset_for_tests()
