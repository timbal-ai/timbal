"""OpenAI speech compatibility/latency audit (synthetic audio, not an accuracy eval).

Run offline: python benchmarks/voice/openai_speech.py --output /tmp/openai-audit.json
Add --live with OPENAI_API_KEY to exercise supported models (billable API calls).
Latency includes this machine's network; results are not production SLOs.
"""

# ruff: noqa: T201

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import platform
import statistics
import time
import wave
from datetime import UTC, datetime
from pathlib import Path

from timbal._openai_audio import LEGACY_VOICES, STT_MODELS, TTS_MODELS, VOICES
from timbal.tools import OpenAISpeechToText
from timbal.voice.openai import OpenAIRealtimeSTT, OpenAIStreamTTS
from timbal.voice.providers import AudioInputConfig, AudioOutputConfig
from timbal.voice.telephony import PcmResampler

PHRASE = "The blue bicycle is outside."


def cpu_cost() -> list[dict]:
    """20 ms frames, 60 s audio x 5 runs; CPU only, no network or sleeps."""
    rows = []
    for source, target, encode in [(r, 24000, True) for r in (8000, 16000, 24000, 48000)] + [(24000, 16000, False)]:
        frame = b"\x00\x01" * (source // 50)
        samples = []
        for _ in range(5):
            resampler = PcmResampler(source, target) if source != target else None
            start = time.process_time()
            for _ in range(3000):
                pcm = resampler.process(frame) if resampler else frame
                if encode:
                    json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode("ascii")})
            if resampler:
                resampler.flush()
            samples.append((time.process_time() - start) * 1000 / 60)
        rows.append(
            {
                "kind": "cpu",
                "source_rate": source,
                "target_rate": target,
                "base64_json": encode,
                "median_cpu_ms_per_audio_second": round(statistics.median(samples), 3),
            }
        )
    return rows


async def live_checks(record) -> None:
    seed = OpenAIStreamTTS()
    try:
        await seed.connect(AudioOutputConfig(sample_rate=24000))
        pcm = b"".join([chunk async for chunk in seed.synthesize(PHRASE)])
    finally:
        await seed.close()
    wav = io.BytesIO()
    with wave.open(wav, "wb") as file:
        file.setnchannels(1)
        file.setsampwidth(2)
        file.setframerate(24000)
        file.writeframes(pcm)
    wav_base64 = base64.b64encode(wav.getvalue()).decode()

    for model in sorted(TTS_MODELS):
        # Exercise every supported built-in voice on aliases and snapshots.
        voices = VOICES if model.startswith("gpt-4o-mini-tts") else sorted(LEGACY_VOICES)
        for voice in voices:
            tts = OpenAIStreamTTS()
            row = {"kind": "tts", "model": model, "voice": voice}
            try:
                await tts.connect(AudioOutputConfig(model=model, voice=voice))
                start, first, size = time.perf_counter(), None, 0
                async with asyncio.timeout(45):
                    async for chunk in tts.synthesize(PHRASE):
                        if first is None:
                            first = time.perf_counter() - start
                        size += len(chunk)
                assert size > 0 and first is not None
                row.update(
                    ok=True,
                    first_audio_ms=round(first * 1000),
                    total_ms=round((time.perf_counter() - start) * 1000),
                    audio_ms=round(size / 32),
                )
            except Exception as exc:
                row.update(ok=False, error=str(exc))
            finally:
                await tts.close()
            record(row)

    for model in sorted(STT_MODELS):
        # Both force-commit and silence-driven VAD on one persistent socket.
        stt = OpenAIRealtimeSTT()
        try:
            await stt.connect(AudioInputConfig(model=model, language="en", sample_rate=24000))
            for manual in (True, False):
                row = {"kind": "stt", "model": model, "manual_commit": manual}
                try:
                    async with asyncio.timeout(30):
                        for offset in range(0, len(pcm), 4800):
                            await stt.push_audio(pcm[offset : offset + 4800])
                        start = time.perf_counter()
                        if manual:
                            await stt.commit()
                        else:
                            await stt.push_audio(b"\0\0" * 24000)
                        async for event in stt.events():
                            if event.type == "committed":
                                assert "bicycle" in event.text.lower(), event.text
                                row.update(
                                    ok=True,
                                    transcript=event.text,
                                    audio_sent_to_final_ms=round((time.perf_counter() - start) * 1000),
                                )
                                break
                        assert row.get("ok"), "No committed transcript"
                    await stt.push_audio(b"\0\0" * 12000)
                except Exception as exc:
                    row.update(ok=False, error=str(exc))
                record(row)
        except Exception as exc:
            record({"kind": "stt_setup", "model": model, "ok": False, "error": str(exc)})
        finally:
            await stt.close()
        try:
            result = await OpenAISpeechToText()(
                model=model, audio_file_base64=wav_base64, filename="synthetic.wav", language="en"
            ).collect()
            assert not result.error, str(result.error)
            assert "bicycle" in result.output["text"].lower(), result.output
            record({"kind": "file_stt", "model": model, "ok": True})
        except Exception as exc:
            record({"kind": "file_stt", "model": model, "ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Make billable OpenAI requests with synthetic audio")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "results": cpu_cost(),
    }

    def record(row):
        report["results"].append(row)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in report["results"]:
        print(json.dumps(row), flush=True)
    if args.live:
        asyncio.run(live_checks(record))
    if any(row.get("ok") is False for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
