"""ElevenLabs adapters: what actually goes on the wire."""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest
from timbal.voice import elevenlabs as el
from timbal.voice.providers import AudioInputConfig


class _IdleWs:
    """A socket that never yields and accepts sends — enough to let ``connect`` return."""

    def __init__(self) -> None:
        self.sent: list[Any] = []
        self.closed = False

    def __aiter__(self) -> _IdleWs:
        return self

    async def __anext__(self) -> str:
        raise StopAsyncIteration

    async def send(self, data: Any) -> None:
        self.sent.append(data)

    async def close(self) -> None:
        self.closed = True


async def _connect_uri(monkeypatch: pytest.MonkeyPatch, config: AudioInputConfig) -> dict[str, list[str]]:
    seen: dict[str, str] = {}

    async def fake_connect(uri: str, **_kw: Any) -> _IdleWs:
        seen["uri"] = uri
        return _IdleWs()

    monkeypatch.setattr(el, "ws_connect", fake_connect)
    stt = el.ElevenLabsRealtimeSTT(api_key="test-key")
    await stt.connect(config)
    try:
        return parse_qs(urlsplit(seen["uri"]).query, keep_blank_values=True)
    finally:
        await stt.close()


class TestScribeRealtimeQuery:
    async def test_list_extras_become_repeated_params(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``keyterms`` is the one vocabulary-biasing knob Scribe has, and the
        API only reads it as ``keyterms=a&keyterms=b``. A stringified list
        (``keyterms=['a', 'b']``) silently biases towards one bogus term."""
        q = await _connect_uri(
            monkeypatch,
            AudioInputConfig(extra={"keyterms": ["Acme", "Timbal AI"], "secondary_languages": ["es", "fr"]}),
        )
        assert q["keyterms"] == ["Acme", "Timbal AI"]
        assert q["secondary_languages"] == ["es", "fr"]
        assert q["model_id"] == [el._DEFAULT_STT_MODEL]
        assert q["commit_strategy"] == ["vad"]

    async def test_scalars_and_booleans_are_wire_spelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        q = await _connect_uri(
            monkeypatch,
            AudioInputConfig(
                language="en",
                extra={"vad_threshold": 0.4, "include_timestamps": True, "no_verbatim": False, "_private": "x"},
            ),
        )
        assert q["language_code"] == ["en"]
        assert q["vad_threshold"] == ["0.4"]
        assert q["include_timestamps"] == ["true"]
        assert q["no_verbatim"] == ["false"]
        assert "_private" not in q
        assert "stt_host" not in q
