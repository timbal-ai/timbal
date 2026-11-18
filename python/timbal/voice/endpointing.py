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
from typing import TYPE_CHECKING, Any

import structlog

from .predictive_eou import prediction_is_complete

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
    min_delay: float = 0.45,
    max_delay: float = 3.0,
    curve: float = 2.0,
) -> float:
    """Map EOU probability to extra silence to wait before force-committing.

    ``delay = min + (max - min) * (1 - p) ** curve`` — smooth, monotonic
    decreasing in ``p``. ``min_delay`` is a LiveKit-shaped floor: even a
    confident-complete score still waits before force-committing, so a brief
    thinking pause ("Uh, I don't know." → "tell me a story") can cancel the
    pending endpoint via resumed speech. Defaults (fired ~0.3s after speech
    stop: 0.2s VAD silence + ~0.1s Smart Turn inference):

    ========  =========  ==========================================
    p         delay      total silence before commit
    ========  =========  ==========================================
    0.95      0.45s      ~0.75s  (confident: LiveKit min_delay floor)
    0.7       ~0.68s     ~1.0s
    0.5       ~1.1s      ~1.4s   (unsure)
    <0.4      >1.4s      provider debounce (~1.2s) often commits first —
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
    MIN_DELAY_SECS = 0.45
    """LiveKit-shaped floor: never force-commit instantly on high-p scores."""
    MAX_DELAY_SECS = 3.0
    DELAY_CURVE = 2.0
    """See :func:`endpointing_delay`."""
    MIN_COMMIT_INTERVAL_SECS = 2.0
    """Floor between force-commits (ElevenLabs throttles rapid commits)."""
    TEXT_INCOMPLETE_DELAY_SECS = 1.2
    """When the latest STT partial looks incomplete (hedge / trailing-off),
    bump the delay to at least this — same short tier as the commit-path
    inverse HOLD. Gives the continuation time to cancel the pending endpoint."""
    TEXT_INCOMPLETE_THRESHOLD = 0.4
    """``P(complete)`` below this from the text scorer → treat as incomplete."""

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
        text_incomplete_delay_secs: float | None = None,
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
        self.text_incomplete_delay_secs = (
            text_incomplete_delay_secs if text_incomplete_delay_secs is not None else self.TEXT_INCOMPLETE_DELAY_SECS
        )

        self._score: Callable[[], Awaitable[float | None]] | None = None
        self._commit: Callable[[], Awaitable[None]] | None = None
        self._should_commit: Callable[[], bool] | None = None
        self._text_score: Callable[[], Awaitable[float | None]] | None = None

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
        # Opt-in during-speech prediction. None keeps the silence-then-score path.
        self._predictor: Any = None
        self._resume_pending = False
        self.last_predictive = False
        """True when the commit just sent used the short during-speech confirmation."""

    _SPEECH_HISTORY_SECS = 3.0
    """How far back :meth:`speech_secs_in_window` can look."""

    def bind(
        self,
        *,
        score: Callable[[], Awaitable[float | None]],
        commit: Callable[[], Awaitable[None]],
        should_commit: Callable[[], bool],
        text_score: Callable[[], Awaitable[float | None]] | None = None,
    ) -> None:
        """Attach the session callbacks. Must run before :meth:`start`.

        ``text_score`` is optional: when provided, returns ``P(complete)`` for
        the latest STT partial (or ``None``). Used to bump the delay when the
        transcript looks mid-thought even if the audio model is confident —
        same LiveKit min/max shape, text as the confidence signal. Swappable
        later for a DistilBERT-class text EOU without changing this wiring.
        """
        self._score = score
        self._commit = commit
        self._should_commit = should_commit
        self._text_score = text_score

    def attach_predictor(self, predictor: Any) -> None:
        """Use a during-speech :class:`~timbal.voice.predictive_eou.SpeechEouPredictor`.

        With one attached, a pause is scored as it starts and a confident
        agreement confirms at ``predictor.eager_silence_secs`` instead of
        waiting out :attr:`min_delay_secs`. A disagreement keeps this class's
        normal floor, anchored so the commit does not move earlier.
        """
        self._predictor = predictor

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
                leaving_pause = self._silence_run > 0.0
                self._speech_run += _FRAME_SECS
                self._silence_run = 0.0
                if leaving_pause:
                    self._resume_pending = True
                # Real resumed speech invalidates a pending endpoint: the user
                # was not done. (Two-frame debounce; see SPEECH_RESUME_SECS.)
                if self._speech_run >= self.SPEECH_RESUME_SECS:
                    self._cancel_pending()
                    if self._resume_pending:
                        self._resume_pending = False
                        if self._predictor is not None:
                            self._predictor.on_speech_resumed()
                self._utterance_speech += _FRAME_SECS
                if self._utterance_speech >= self.min_speech_secs:
                    self._utterance_active = True
                if self._predictor is not None and self._utterance_speech > 0.0:
                    self._predictor.note_speech()
            else:
                self._silence_run += _FRAME_SECS
                self._speech_run = 0.0
                if (
                    self._predictor is not None
                    and self._utterance_active
                    and self._silence_run >= self._predictor.predict_silence_secs
                ):
                    self._predictor.kick_audio()
                # Predictive mode arms at the short confirmation, while the
                # caller may still resume. The task itself decides whether
                # that pause is a turn; an unconfident score waits out the
                # normal floor. Without a predictor this is stop_silence.
                arm_at = self.stop_silence_secs
                if self._predictor is not None:
                    arm_at = min(arm_at, self._predictor.eager_silence_secs)
                if self._silence_run >= arm_at and self._utterance_active:
                    # Consume the utterance: one endpoint attempt per speech-stop.
                    logger.debug(
                        "vad_speech_stop",
                        utterance_secs=round(self._utterance_speech, 2),
                        predictive=self._predictor is not None and arm_at < self.stop_silence_secs,
                    )
                    self._utterance_active = False
                    self._spawn_pending()
                if self._silence_run >= self.stop_silence_secs:
                    # Real silence ends the utterance either way — sub-threshold
                    # blips that never reached min_speech must not accumulate
                    # across separate noises into a phantom utterance.
                    self._utterance_speech = 0.0

    def notify_committed(self) -> None:
        """A committed transcript arrived (ours or provider debounce) — any
        pending endpoint is now stale."""
        self._cancel_pending()

    def last_speech_at(self) -> float | None:
        """Monotonic time of the last frame Silero called speech, or ``None``.

        ``None`` when the VAD has no recent evidence (not started, starved, or
        no speech yet) — same caveat as :meth:`speech_secs_in_window`. The
        session uses this as the user's end-of-speech stamp for STT latency.
        """
        if not self._started or not self._recent_speech:
            return None
        if time.monotonic() - self._last_frame_at > 1.0:
            return None
        return self._recent_speech[-1]

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
            self.last_predictive = False
            if not self._should_commit():
                logger.debug("vad_endpoint_skipped", reason="session_gate")
                return
            t0 = time.monotonic()
            silence_at_arm = self._silence_run
            signals = None
            if self._predictor is not None:
                signals = await self._predictor.signals()
            early = signals is not None and prediction_is_complete(signals)
            p_text: float | None = None
            text_bumped = False
            if early:
                assert signals is not None and signals.p_audio is not None
                p = signals.p_audio
                p_text = signals.p_text
                driver = "predictive"
                delay = 0.0
            else:
                if signals is not None and signals.audio_fresh:
                    p = signals.p_audio
                else:
                    p = await self._score()
                if signals is not None and signals.text_fresh:
                    p_text = signals.p_text
                elif self._text_score is not None:
                    try:
                        p_text = await self._text_score()
                    except Exception as e:
                        logger.warning("vad_text_score_failed", error=str(e))
                        p_text = None
                driver = "audio"
                if p is None:
                    # No audio EOU (text-only detector, or not enough buffered
                    # signal). Size the delay from the text score instead of
                    # skipping: the mapping only needs a P(complete), and refusing
                    # to commit here leaves the turn waiting on the provider's own
                    # debounce — 7.5s on a spoken account number under Nova.
                    if p_text is None:
                        logger.debug("vad_endpoint_skipped", reason="no_eou_score")
                        return
                    p, driver = p_text, "text"
                delay = endpointing_delay(
                    p,
                    min_delay=self.min_delay_secs,
                    max_delay=self.max_delay_secs,
                    curve=self.delay_curve,
                )
                # Only a bump when text is the *second* opinion. With text driving,
                # the incomplete case is already priced into the delay above, and
                # applying the floor again would double-count it.
                if driver == "audio" and p_text is not None and p_text < self.TEXT_INCOMPLETE_THRESHOLD:
                    bumped = max(delay, self.text_incomplete_delay_secs)
                    text_bumped = bumped > delay
                    delay = bumped
            if early:
                assert self._predictor is not None
                heard = max(self._silence_run, silence_at_arm + (time.monotonic() - t0))
                remaining = self._predictor.eager_silence_secs - heard
            elif self._predictor is not None:
                # Armed before stop_silence. Count silence already heard, and
                # the wall time of the score when frames did not advance, so
                # an unconfident pause still commits at stop + delay.
                heard = max(self._silence_run, silence_at_arm + (time.monotonic() - t0))
                remaining = self.stop_silence_secs + delay - heard
            else:
                remaining = delay - (time.monotonic() - t0)
            # INFO on purpose: fires once per speech-stop and is the whole
            # observable behaviour of the endpointing fast path.
            logger.info(
                "vad_eou_score",
                p=round(p, 3),
                driver=driver,
                delay_secs=round(delay, 3),
                score_ms=round((time.monotonic() - t0) * 1000, 1),
                p_text=round(p_text, 3) if p_text is not None else None,
                text_incomplete_bump=text_bumped,
                predictive=early,
            )
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
            self.last_predictive = early
            logger.info("vad_endpoint_commit", p=round(p, 3), predictive=early)
            await self._commit()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("vad_endpoint_failed", error=str(e))
