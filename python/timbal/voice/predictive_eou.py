"""Shared Flux eager-threshold.

``flux_eager_threshold`` is the one place that chooses ``eager_eot_threshold``.
There is no 0.3 floor. The result stays strictly below ``eot_threshold``.
An explicit ``eager_eot_threshold`` is lowered the same way when it is not
already below that end-of-turn value. Omitted, the value is
``min(0.5, eot_threshold - 0.1)``.
"""

from __future__ import annotations

from typing import Any

# Sent as Flux ``eager_eot_threshold`` when a feature needs eager events and
# the caller did not set one. Must stay below the confirmation ``eot_threshold``
# (0.8 unless overridden). Deepgram rejects eager >= eot.
DEFAULT_EAGER_EOT_THRESHOLD = 0.5
DEFAULT_FLUX_EOT_THRESHOLD = 0.8


def flux_eager_threshold(stt_extra: dict[str, Any]) -> float:
    """Eager confidence for a Flux session, kept strictly under end-of-turn.

    An explicit ``eager_eot_threshold`` wins. Otherwise use
    :data:`DEFAULT_EAGER_EOT_THRESHOLD`, lowered when the caller set
    ``eot_threshold`` close to it. Deepgram requires eager < eot.
    """
    eot = DEFAULT_FLUX_EOT_THRESHOLD
    configured = stt_extra.get("eot_threshold")
    if configured is not None and not isinstance(configured, bool):
        try:
            eot = float(configured)
        except (TypeError, ValueError):
            eot = DEFAULT_FLUX_EOT_THRESHOLD
    raw = stt_extra.get("eager_eot_threshold")
    eager: float | None = None
    if raw is not None and not isinstance(raw, bool):
        try:
            eager = float(raw)
        except (TypeError, ValueError):
            eager = None
    if eager is None:
        # No 0.3 floor: that equals eot when eot is 0.3, and Deepgram rejects
        # eager_eot_threshold >= eot_threshold.
        eager = min(DEFAULT_EAGER_EOT_THRESHOLD, eot - 0.1)
    if eot > 0 and eager >= eot:
        eager = eot - 0.1 if eot - 0.1 > 0 else eot / 2
    if eager <= 0:
        eager = eot / 2 if eot > 0 else DEFAULT_EAGER_EOT_THRESHOLD
    return eager
