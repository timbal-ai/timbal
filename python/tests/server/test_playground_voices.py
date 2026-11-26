"""Playground voice catalog: region labels and the Gemini accent line of the preview."""

from __future__ import annotations

import base64
import json

import pytest
from timbal.server import playground_voices as pv
from timbal.voice.gemini_tts import DEFAULT_ACCENT


@pytest.mark.parametrize(
    ("accent", "locale", "text", "region"),
    [
        ("andalusian", "", "", "andalusian"),
        ("", "es-ES", "Voz de Sevilla", "andalusian"),
        ("", "es-ES", "", "castilian"),
        ("mexican", "es-MX", "", "latam"),
        ("argentine", "es-AR", "", "latam"),
        ("canarian", "", "", "canarian"),
    ],
)
def test_spanish_region(accent, locale, text, region):
    assert pv.spanish_region(accent, locale, text) == region


@pytest.mark.parametrize(("accent", "expected"), [("andaluz marcado", "andaluz marcado"), ("", DEFAULT_ACCENT)])
def test_gemini_preview_sends_the_accent_line(monkeypatch, accent, expected):
    """Hearing a Gemini voice with an accent needs the same director's brief a call would send."""
    sent: dict = {}

    def fake_request(*args, **_kwargs):
        sent["body"] = args[3]
        pcm = base64.b64encode(b"\x00\x00" * 100).decode()
        return json.dumps({"candidates": [{"content": {"parts": [{"inlineData": {"data": pcm}}]}}]}).encode()

    monkeypatch.setattr(pv, "_request", fake_request)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    wav = pv.tts_preview("gemini", "fenrir", "Hola, buenas.", accent=accent)
    assert wav[:4] == b"RIFF"
    assert f"Acento: {expected}" in sent["body"]["contents"][0]["parts"][0]["text"]
