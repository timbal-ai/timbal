"""Duplex acceptance gate.

The live provider run is ``integration`` and stays out of the default pytest
invocation. Everything else checks the analyzer and the opt-in switch with no
network.
"""

from __future__ import annotations

import array
import importlib.util
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("accept_duplex", ROOT / "benchmarks" / "voice" / "accept_duplex.py")
assert _SPEC and _SPEC.loader
accept = importlib.util.module_from_spec(_SPEC)
sys.modules["accept_duplex"] = accept
_SPEC.loader.exec_module(accept)


def _tone(secs: float, amplitude: int = 8000, freq: float = 440.0) -> bytes:
    n = int(accept.SAMPLE_RATE * secs)
    samples = array.array("h")
    for i in range(n):
        samples.append(int(amplitude * math.sin(2 * math.pi * freq * i / accept.SAMPLE_RATE)))
    return samples.tobytes()


def test_speech_regions_split_on_silence() -> None:
    silence = b"\x00" * accept.BYTES_PER_SECOND
    pcm = _tone(0.4) + silence + _tone(0.4)
    assert accept.speech_regions(silence) == []
    regions = accept.speech_regions(pcm)
    assert len(regions) == 2
    assert regions[0]["dur_secs"] >= 0.3
    assert regions[1]["start_secs"] > 1.0
    assert accept.pcm_rms(silence) == 0.0
    assert accept.pcm_rms(_tone(0.2)) > accept.SPEECH_RMS


def test_farewell_count_is_one_spoken_goodbye() -> None:
    assert accept.farewell_count("Goodbye.") == 1
    assert accept.farewell_count("Goodbye. Goodbye.") == 2
    assert accept.farewell_count("Sure, go ahead.") == 0
    assert accept.farewell_count("Bye bye.") == 0


def test_acceptance_is_opt_in_and_names_missing_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(accept.ACCEPTANCE_FLAG, raising=False)
    reason = accept.acceptance_skip_reason()
    assert reason is not None
    assert accept.ACCEPTANCE_FLAG in reason
    monkeypatch.setenv(accept.ACCEPTANCE_FLAG, "1")
    assert accept.acceptance_skip_reason() is None
    for name in accept.PROVIDER_KEYS:
        monkeypatch.delenv(name, raising=False)
    assert accept.missing_provider_keys() == list(accept.PROVIDER_KEYS)


def test_alright_checker_rejects_a_cancelled_hangup_and_a_second_goodbye() -> None:
    base = {
        "providers": {"stt": "DeepgramFluxSTT", "tts": "ElevenLabsStreamTTS", "llm": accept.LLM_MODEL},
        "execution": "live_network",
        "timeouts": [],
        "errors": [],
        "ended": True,
        "ended_t": 6.0,
        "last_audio_t": 4.0,
        "playhead_shortfall_secs": 0.0,
        "schedule_lag_secs": 0.0,
        "fillers": [],
        "partials": [{"text": "Al", "kind": "pending", "t": 5.1}],
        "assistant_spoken": "Goodbye.",
        "audio": {"regions": [{"dur_secs": 0.8}]},
        "tool_log": [{"tool": "end_call", "phase": "exit", "ok": True, "t": 5.5}],
    }
    assert accept._check_alright(dict(base)) == []
    cancelled = dict(base)
    cancelled["tool_log"] = [{"tool": "end_call", "phase": "exit", "ok": False, "t": 5.5}]
    assert any("cancelled" in item for item in accept._check_alright(cancelled))
    doubled = dict(base)
    doubled["assistant_spoken"] = "Goodbye. Goodbye."
    doubled["audio"] = {"regions": [{"dur_secs": 0.8}, {"dur_secs": 0.7}]}
    problems = accept._check_alright(doubled)
    assert any("farewell count" in item for item in problems)
    assert any("audible span" in item for item in problems)


def test_hangup_checker_rejects_filler_and_early_disconnect() -> None:
    report = {
        "providers": {"stt": "DeepgramFluxSTT", "tts": "ElevenLabsStreamTTS", "llm": accept.LLM_MODEL},
        "execution": "live_network",
        "timeouts": [],
        "errors": [],
        "ended": True,
        "ended_t": 3.0,
        "last_audio_t": 4.5,
        "playhead_shortfall_secs": 0.8,
        "schedule_lag_secs": 0.8,
        "fillers": [{"text": "One moment.", "t": 1.2}],
        "partials": [],
        "assistant_spoken": "Goodbye.",
        "audio": {"regions": [{"dur_secs": 0.8}]},
        "tool_log": [{"tool": "end_call", "phase": "exit", "ok": True, "t": 2.0}],
    }
    problems = accept._check_alright(report)
    assert any("filler" in item for item in problems)
    assert any("still scheduled" in item for item in problems)


@pytest.mark.integration
async def test_provider_duplex_acceptance(tmp_path: Path) -> None:
    """Real Flux + gpt-5.4-mini + ElevenLabs. Not part of the default run."""
    reason = accept.acceptance_skip_reason()
    if reason:
        pytest.skip(reason)
    missing = accept.missing_provider_keys()
    if missing:
        pytest.fail(f"{accept.ACCEPTANCE_FLAG}=1 but missing {', '.join(missing)}")
    report = await accept.run_acceptance(tmp_path)
    assert report["passed"], report["failures"]
    assert report["listening"] is False
    for scenario in report["scenarios"]:
        assert scenario["execution"] == "live_network"
        assert scenario["providers"]["llm"] == "openai/gpt-5.4-mini"
