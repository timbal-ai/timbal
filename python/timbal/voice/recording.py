"""Call recording: one playable audio file per voice session, plus a manifest.

The recorder reconstructs the *call as heard*, not the TTS as synthesized.
Mic PCM is the master clock — it arrives continuously in real time on both
transports — and every mic chunk drains an equal span of the agent's queued
TTS onto the same timeline (silence when the agent wasn't talking). TTS
synthesizes faster than real time, so the agent side is a queue; a barge-in
drops its unheard tail (:meth:`CallRecorder.drop_agent_tail`) exactly like
interruption truncation drops unheard text.

Layouts:

* ``"mixed"`` (default) — mono, both voices summed with a clipping guard.
  Sounds like the call did; what a human reviewing the conversation wants.
* ``"split"`` — stereo, caller left / agent right. The call-center
  convention for re-transcription/diarization where voice bleed matters.

Output is MP3 (universal ``<audio>`` playback, and frames are self-contained
so a crashed process still leaves a playable file up to the crash point),
encoded progressively via av/libmp3lame — no extra dependency beyond the
``timbal[voice]`` extra.

The recorder is deliberately fail-safe: any encoding error disables it and
logs; it never raises into the audio path of a live call.
"""

from __future__ import annotations

import fractions
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import structlog

try:
    import av
    import numpy as np
except ImportError as e:  # pragma: no cover — exercised only without the extra
    raise ImportError(
        "Call recording requires the timbal[voice] extra (av + numpy): "
        "uv pip install 'timbal[voice]'"
    ) from e

logger = structlog.get_logger("timbal.voice.recording")

# libmp3lame's fixed frame size; used until the codec context reports one.
_MP3_FRAME_SAMPLES = 1152


@dataclass
class RecordingResult:
    """What :meth:`CallRecorder.close` hands to the ``on_saved`` hook."""

    audio_path: Path
    manifest_path: Path | None
    duration_secs: float


class CallRecorder:
    """Streaming MP3 recorder for one voice session.

    Not thread-safe; call from the session's event loop only (the session
    already does). All feed methods are no-ops after :meth:`close` or after
    an internal encoder failure.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        sample_rate: int = 16_000,
        layout: Literal["mixed", "split"] = "mixed",
        bitrate_kbps: int = 32,
        on_saved: Any = None,
        meta: dict[str, Any] | None = None,
    ) -> None:
        if layout not in ("mixed", "split"):
            raise ValueError(f"layout must be 'mixed' or 'split', got {layout!r}")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._sample_rate = sample_rate
        self._layout = layout
        self._bitrate_kbps = bitrate_kbps
        #: Optional async callable awaited by the session after files are written.
        self.on_saved = on_saved
        #: Extra manifest ``meta`` entries (e.g. platform identity: org_id,
        #: project_id, ...) merged under the session's own recording_meta.
        self.meta = meta

        self._channel_layout = "mono" if layout == "mixed" else "stereo"
        self._container = av.open(str(self._path), mode="w")
        self._stream = self._container.add_stream("libmp3lame", rate=sample_rate)
        self._stream.codec_context.layout = self._channel_layout
        self._stream.codec_context.format = "s16p"
        self._stream.codec_context.bit_rate = bitrate_kbps * 1000
        self._fifo = av.AudioFifo()

        # Sample alignment: chunks may split an s16 sample across calls.
        self._mic_rem = b""
        self._agent_pending = bytearray()
        self._mic_bytes = 0
        self._agent_bytes = 0
        self._samples_written = 0
        self._closed = False
        self._failed = False
        self._result: RecordingResult | None = None

    # -- Introspection --------------------------------------------------------

    @property
    def audio_path(self) -> Path:
        return self._path

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def layout(self) -> str:
        return self._layout

    @property
    def bitrate_kbps(self) -> int:
        return self._bitrate_kbps

    @property
    def agent_pending_bytes(self) -> int:
        """Agent PCM queued but not yet drained onto the timeline."""
        return len(self._agent_pending)

    @property
    def duration_secs(self) -> float:
        return self._samples_written / self._sample_rate

    # -- Feed points -----------------------------------------------------------

    def add_mic(self, chunk: bytes) -> None:
        """Caller audio: advances the call clock and drains the agent queue."""
        if self._closed or self._failed or not chunk:
            return
        data = self._mic_rem + chunk
        usable = len(data) - (len(data) % 2)
        self._mic_rem = data[usable:]
        data = data[:usable]
        if not data:
            return
        self._mic_bytes += len(data)
        agent = bytes(self._agent_pending[: len(data)])
        del self._agent_pending[: len(agent)]
        if len(agent) < len(data):
            agent += b"\x00" * (len(data) - len(agent))
        self._write(data, agent)

    def add_agent(self, chunk: bytes) -> None:
        """Agent TTS PCM: queued, drained in real time by the mic clock."""
        if self._closed or self._failed or not chunk:
            return
        self._agent_bytes += len(chunk)
        self._agent_pending.extend(chunk)

    def drop_agent_tail(self, nbytes: int) -> int:
        """Barge-in: drop the last ``nbytes`` of queued agent audio (unheard).

        Returns how many bytes were actually dropped (clamped to the queue,
        kept sample-aligned). The already-drained portion of the timeline is
        untouched — it was heard.
        """
        if self._closed or self._failed or nbytes <= 0:
            return 0
        drop = min(nbytes, len(self._agent_pending))
        drop -= drop % 2
        if drop:
            del self._agent_pending[len(self._agent_pending) - drop :]
        return drop

    # -- Finalization ----------------------------------------------------------

    def close(self, manifest: dict[str, Any] | None = None) -> RecordingResult | None:
        """Drain, flush, write the manifest. Idempotent; returns None on failure.

        Remaining queued agent audio is drained past the mic timeline — the
        call ended while it was still playing, and for uninterrupted turns it
        would have been heard.
        """
        if self._closed:
            return self._result
        self._closed = True
        if not self._failed:
            try:
                if self._agent_pending:
                    tail = bytes(self._agent_pending)
                    self._agent_pending.clear()
                    self._write(b"\x00" * len(tail), tail)
                self._flush()
            except Exception as e:
                self._failed = True
                logger.error("recording_finalize_failed", error=str(e), exc_info=True)
        try:
            self._container.close()
        except Exception as e:
            self._failed = True
            logger.error("recording_container_close_failed", error=str(e))

        manifest_path: Path | None = None
        if manifest is not None and not self._failed:
            # The manifest is built before the queued tail is drained above —
            # refresh the duration so it matches the file.
            if isinstance(manifest.get("audio"), dict):
                manifest["audio"]["duration_secs"] = round(self.duration_secs, 3)
            manifest_path = self._path.with_suffix(".json")
            try:
                # Atomic (tmp + rename): sweepers treat "manifest exists" as
                # "recording complete", so a half-written json must never be
                # observable.
                tmp_path = manifest_path.with_name(manifest_path.name + ".tmp")
                tmp_path.write_text(json.dumps(manifest, indent=2, default=str))
                tmp_path.replace(manifest_path)
            except Exception as e:
                manifest_path = None
                logger.error("recording_manifest_failed", error=str(e))

        if self._failed:
            return None
        self._result = RecordingResult(
            audio_path=self._path,
            manifest_path=manifest_path,
            duration_secs=self.duration_secs,
        )
        logger.info(
            "recording_saved",
            path=str(self._path),
            duration_secs=round(self._result.duration_secs, 2),
            layout=self._layout,
        )
        return self._result

    # -- Internal ---------------------------------------------------------------

    def _write(self, mic: bytes, agent: bytes) -> None:
        """Encode one span (equal-length mic/agent PCM16 mono) onto the timeline."""
        try:
            mic_arr = np.frombuffer(mic, dtype=np.int16).astype(np.int32)
            agent_arr = np.frombuffer(agent, dtype=np.int16).astype(np.int32)
            if self._layout == "mixed":
                planes = np.clip(mic_arr + agent_arr, -32768, 32767).astype(np.int16)[np.newaxis, :]
            else:
                planes = np.stack(
                    [mic_arr.astype(np.int16), agent_arr.astype(np.int16)]
                )
            frame = av.AudioFrame.from_ndarray(planes, format="s16p", layout=self._channel_layout)
            frame.sample_rate = self._sample_rate
            frame.time_base = fractions.Fraction(1, self._sample_rate)
            frame.pts = self._samples_written
            self._samples_written += planes.shape[1]
            self._fifo.write(frame)
            frame_size = self._stream.codec_context.frame_size or _MP3_FRAME_SAMPLES
            while True:
                out = self._fifo.read(frame_size)
                if out is None:
                    break
                for pkt in self._stream.encode(out):
                    self._container.mux(pkt)
        except Exception as e:
            # Never raise into the live audio path — disable and log once.
            self._failed = True
            logger.error("recording_encode_failed", error=str(e), exc_info=True)

    def _flush(self) -> None:
        tail = self._fifo.read()
        if tail is not None:
            for pkt in self._stream.encode(tail):
                self._container.mux(pkt)
        for pkt in self._stream.encode(None):
            self._container.mux(pkt)


def build_manifest(
    *,
    session_id: str,
    started_at: float | None,
    meta: dict[str, Any] | None,
    transcript: list[Any],
    turns: list[Any],
    recorder: CallRecorder,
) -> dict[str, Any]:
    """Manifest JSON for the platform UI: timestamped transcript + latency chips."""
    t0 = started_at
    entries = []
    for e in transcript:
        d = e.model_dump()
        if t0 is not None:
            d["offset_ms"] = max(0, round((e.timestamp - t0) * 1000))
        entries.append(d)
    return {
        "session_id": session_id,
        "started_at": t0,
        "ended_at": time.time(),
        "meta": {**(recorder.meta or {}), **(meta or {})},
        "transcript": entries,
        "turns": [m.model_dump() for m in turns],
        "audio": {
            "file": recorder.audio_path.name,
            "format": "mp3",
            "layout": recorder.layout,
            "sample_rate": recorder.sample_rate,
            "bitrate_kbps": recorder.bitrate_kbps,
            "duration_secs": round(recorder.duration_secs, 3),
        },
    }
