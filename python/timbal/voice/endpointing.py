"""Local VAD endpointing — cut STT commit latency with Silero + Smart Turn.

The cascaded pipeline's latency floor is the STT provider's commit debounce:
ElevenLabs Scribe VAD waits ~1.2s of silence before emitting a committed
transcript, and only then does turn detection run. LiveKit and Pipecat instead
run a local VAD on raw PCM and score their EOU model ~200ms after speech
stops — their entire responsiveness game happens in that 300–600ms budget.

:class:`VadEndpointer` closes that gap without touching the turn-taking
machinery:

1. Silero VAD (:class:`~timbal.voice.vad.SileroVad`) runs on the mic PCM
   inside :meth:`~timbal.voice.VoiceSession._forward_audio`.
2. On speech-stop (:attr:`~VadEndpointer.STOP_SILENCE_SECS` of silence after
   real speech), Smart Turn scores the buffered audio window immediately.
3. The EOU probability maps to a **variable** extra wait
   (:func:`endpointing_delay`): confident-complete → commit almost
   immediately; unsure → wait longer; incomplete → the computed delay exceeds
   the provider's own debounce, so the provider commit (and the existing HOLD
   machinery) wins automatically.
4. When the delay elapses with no new speech, the endpointer force-commits the
   provider buffer via ``stt.commit()`` — the committed transcript then flows
   through the session's normal ``_handle_committed`` path. No parallel
   commit/dedup state machine.

The endpointer is a *fast path* only: disabling it (or the ``timbal[voice]``
extra being absent) restores exactly today's provider-debounce behaviour.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from .vad import SileroVad

logger = structlog.get_logger("timbal.voice.endpointing")

# Must match timbal.voice.vad.FRAME_SECS without importing it (that module
# needs numpy/onnxruntime; this one must import bare so VoiceSession can
# reference VadEndpointer in type hints and degrade at start()).
_FRAME_SECS = 512 / 16_000


def endpointing_delay(
    p: float,
    *,
    min_delay: float = 0.0,
    max_delay: float = 3.0,
    curve: float = 2.0,
) -> float:
    """Map EOU probability to extra silence to wait before force-committing.

    ``delay = min + (max - min) * (1 - p) ** curve`` — smooth, monotonic
    decreasing in ``p``. With the defaults (fired ~0.3s after speech stop:
    0.2s VAD silence + ~0.1s Smart Turn inference):

    ========  =========  ==========================================
    p         delay      total silence before commit
    ========  =========  ==========================================
    0.95      ~0.01s     ~0.3s   (confident: LiveKit-class response)
    0.7       ~0.27s     ~0.6s
    0.5       0.75s      ~1.05s  (unsure: still beats the provider)
    <0.4      >1.1s      provider debounce (~1.2s) commits first —
                         incomplete utterances fall through to the
                         existing HOLD machinery untouched
    ========  =========  ==========================================
    """
    p = min(1.0, max(0.0, p))
    return min_delay + (max_delay - min_delay) * (1.0 - p) ** curve


class VadEndpointer:
    """Silero speech-stop → Smart Turn score → variable delay → ``stt.commit()``.

    Wire-up (done by :class:`~timbal.voice.VoiceSession`):

    * :meth:`bind` supplies the session callbacks — ``score`` (audio EOU
      probability for the buffered mic window, e.g.
      :meth:`~timbal.voice.LocalAudioTurnDetector.score_recent_audio`),
      ``commit`` (force the STT provider to finalize), and ``should_commit``
      (session gating: not closed, real transcribed speech since the last
      commit).
    * :meth:`push` receives every mic PCM chunk (inline; Silero is <1ms/frame).
    * :meth:`notify_committed` cancels any pending endpoint when a commit
      arrives through the provider path (ours or its own debounce).

    All timing knobs are class attributes overridable per instance via the
    constructor.
    """

    SPEECH_THRESHOLD = 0.5
    """Per-frame Silero probability at/above which a frame counts as speech."""
    STOP_SILENCE_SECS = 0.20
    """Silence after speech before the EOU model is scored (Pipecat's stop_secs)."""
    MIN_SPEECH_SECS = 0.10
    """Minimum accumulated speech for an utterance to be endpoint-eligible
    (filters clicks/echo blips that STT never transcribes). Accumulated over
    the whole utterance — Silero routinely dips below threshold between
    phonemes, so this must NOT require consecutive frames (a short "work."
    would otherwise never arm; LiveKit's equivalent is 0.05s)."""
    SPEECH_RESUME_SECS = 0.064
    """Consecutive speech (2 frames) that cancels a pending endpoint — a single
    noise-spike frame must not kill a valid pending commit."""
    MIN_DELAY_SECS = 0.0
    MAX_DELAY_SECS = 3.0
    DELAY_CURVE = 2.0
    """See :func:`endpointing_delay`."""
    MIN_COMMIT_INTERVAL_SECS = 2.0
    """Floor between force-commits (ElevenLabs throttles rapid commits)."""

    def __init__(
        self,
        vad: SileroVad | None = None,
        *,
        speech_threshold: float | None = None,
        stop_silence_secs: float | None = None,
        min_speech_secs: float | None = None,
        min_delay_secs: float | None = None,
        max_delay_secs: float | None = None,
        delay_curve: float | None = None,
        min_commit_interval_secs: float | None = None,
    ) -> None:
        self._vad = vad
        self.speech_threshold = speech_threshold if speech_threshold is not None else self.SPEECH_THRESHOLD
        self.stop_silence_secs = stop_silence_secs if stop_silence_secs is not None else self.STOP_SILENCE_SECS
        self.min_speech_secs = min_speech_secs if min_speech_secs is not None else self.MIN_SPEECH_SECS
        self.min_delay_secs = min_delay_secs if min_delay_secs is not None else self.MIN_DELAY_SECS
        self.max_delay_secs = max_delay_secs if max_delay_secs is not None else self.MAX_DELAY_SECS
        self.delay_curve = delay_curve if delay_curve is not None else self.DELAY_CURVE
        self.min_commit_interval_secs = (
            min_commit_interval_secs if min_commit_interval_secs is not None else self.MIN_COMMIT_INTERVAL_SECS
        )

        self._score: Callable[[], Awaitable[float | None]] | None = None
        self._commit: Callable[[], Awaitable[None]] | None = None
        self._should_commit: Callable[[], bool] | None = None

        self._started = False
        self._closed = False
        self._speech_run = 0.0
        self._silence_run = 0.0
        # Total speech in the current utterance. Deliberately NOT reset by a
        # single silence frame (unlike _speech_run): Silero dips between
        # phonemes and short words would never reach min_speech otherwise.
        self._utterance_speech = 0.0
        self._utterance_active = False
        self._pending: asyncio.Task[None] | None = None
        self._last_commit_sent_at = 0.0
        # Timestamps of recent speech frames (p >= threshold), pruned to
        # _SPEECH_HISTORY_SECS. Lets the session corroborate STT partials with
        # real mic energy (hallucination veto for barge-ins).
        self._recent_speech: deque[float] = deque()
        self._last_frame_at = 0.0

    _SPEECH_HISTORY_SECS = 3.0
    """How far back :meth:`speech_secs_in_window` can look."""

    def bind(
        self,
        *,
        score: Callable[[], Awaitable[float | None]],
        commit: Callable[[], Awaitable[None]],
        should_commit: Callable[[], bool],
    ) -> None:
        """Attach the session callbacks. Must run before :meth:`start`."""
        self._score = score
        self._commit = commit
        self._should_commit = should_commit

    async def start(self, *, sample_rate: int) -> None:
        """Load Silero (downloading/instantiating off the event loop) and arm.

        Raises ``ImportError`` when the ``timbal[voice]`` extra is missing —
        the session catches it and leaves endpointing off.
        """
        if self._score is None or self._commit is None or self._should_commit is None:
            raise RuntimeError("Call bind() before start().")
        if self._vad is None:
            from .vad import SileroVad  # requires timbal[voice]

            self._vad = SileroVad()
        await self._vad.start(sample_rate=sample_rate)
        self._started = True

    async def close(self) -> None:
        self._closed = True
        self._started = False
        self._cancel_pending()

    def push(self, chunk: bytes) -> None:
        """Feed mic PCM; runs the VAD state machine inline (called per chunk)."""
        if not self._started or self._closed:
            return
        try:
            probs = self._vad.process(chunk)
        except Exception as e:
            logger.warning("vad_process_failed", error=str(e))
            return
        now = time.monotonic()
        self._last_frame_at = now
        cutoff = now - self._SPEECH_HISTORY_SECS
        while self._recent_speech and self._recent_speech[0] < cutoff:
            self._recent_speech.popleft()
        for p in probs:
            if p >= self.speech_threshold:
                self._recent_speech.append(now)
                self._speech_run += _FRAME_SECS
                self._silence_run = 0.0
                # Real resumed speech invalidates a pending endpoint: the user
                # was not done. (Two-frame debounce; see SPEECH_RESUME_SECS.)
                if self._speech_run >= self.SPEECH_RESUME_SECS:
                    self._cancel_pending()
                self._utterance_speech += _FRAME_SECS
                if self._utterance_speech >= self.min_speech_secs:
                    self._utterance_active = True
            else:
                self._silence_run += _FRAME_SECS
                self._speech_run = 0.0
                if self._silence_run >= self.stop_silence_secs:
                    if self._utterance_active:
                        # Consume the utterance: one endpoint attempt per speech-stop.
                        logger.debug(
                            "vad_speech_stop",
                            utterance_secs=round(self._utterance_speech, 2),
                        )
                        self._utterance_active = False
                        self._spawn_pending()
                    # Real silence ends the utterance either way — sub-threshold
                    # blips that never reached min_speech must not accumulate
                    # across separate noises into a phantom utterance.
                    self._utterance_speech = 0.0

    def notify_committed(self) -> None:
        """A committed transcript arrived (ours or provider debounce) — any
        pending endpoint is now stale."""
        self._cancel_pending()

    def speech_secs_in_window(self, window_secs: float) -> float | None:
        """Total Silero speech seconds in the trailing ``window_secs``.

        Returns ``None`` when the VAD hasn't processed mic audio in the last
        second (not started, starved, or failing) — callers must treat that as
        "no evidence either way" and NOT veto on it, or a broken VAD would
        make the assistant uninterruptible.
        """
        now = time.monotonic()
        if not self._started or now - self._last_frame_at > 1.0:
            return None
        window_secs = min(window_secs, self._SPEECH_HISTORY_SECS)
        cutoff = now - window_secs
        return sum(_FRAME_SECS for t in self._recent_speech if t >= cutoff)

    # -- internal -----------------------------------------------------------

    def _cancel_pending(self) -> None:
        if self._pending is not None and not self._pending.done():
            self._pending.cancel()
        self._pending = None

    def _spawn_pending(self) -> None:
        if self._pending is not None and not self._pending.done():
            return
        self._pending = asyncio.create_task(self._run_endpoint())

    async def _run_endpoint(self) -> None:
        try:
            if not self._should_commit():
                logger.debug("vad_endpoint_skipped", reason="session_gate")
                return
            t0 = time.monotonic()
            p = await self._score()
            if p is None:
                # No audio EOU available / not enough buffered signal: never
                # force-commit blind — the provider debounce handles it.
                logger.debug("vad_endpoint_skipped", reason="no_eou_score")
                return
            delay = endpointing_delay(
                p,
                min_delay=self.min_delay_secs,
                max_delay=self.max_delay_secs,
                curve=self.delay_curve,
            )
            # INFO on purpose: fires once per speech-stop and is the whole
            # observable behaviour of the endpointing fast path.
            logger.info(
                "vad_eou_score",
                p=round(p, 3),
                delay_secs=round(delay, 3),
                score_ms=round((time.monotonic() - t0) * 1000, 1),
            )
            remaining = delay - (time.monotonic() - t0)
            if remaining > 0:
                await asyncio.sleep(remaining)
            if self._closed or not self._should_commit():
                logger.debug("vad_endpoint_skipped", reason="session_gate_post_delay")
                return
            now = time.monotonic()
            if now - self._last_commit_sent_at < self.min_commit_interval_secs:
                logger.debug("vad_endpoint_skipped", reason="commit_interval")
                return
            self._last_commit_sent_at = now
            logger.info("vad_endpoint_commit", p=round(p, 3))
            await self._commit()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("vad_endpoint_failed", error=str(e))
