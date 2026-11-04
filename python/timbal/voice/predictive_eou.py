"""Shared Flux eager-threshold.

``flux_eager_threshold`` is the one place that chooses ``eager_eot_threshold``
for speculative generation and for during-speech prediction. An explicit
``stt_extra["eager_eot_threshold"]`` wins. Otherwise the value is
``min(0.5, max(0.3, eot_threshold - 0.1))``, which stays inside Deepgram's
0.3–0.9 range and strictly below ``eot_threshold``.

During-speech prediction adds the rest of this module (scoring while the
caller is still speaking). This function is the shared piece, copied to match
that implementation so the two features do not each invent a threshold.
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
    raw = stt_extra.get("eager_eot_threshold")
    if raw is not None and not isinstance(raw, bool):
        try:
            return float(raw)
        except (TypeError, ValueError):
            return DEFAULT_EAGER_EOT_THRESHOLD
    eot = DEFAULT_FLUX_EOT_THRESHOLD
    configured = stt_extra.get("eot_threshold")
    if configured is not None and not isinstance(configured, bool):
        try:
            eot = float(configured)
        except (TypeError, ValueError):
            eot = DEFAULT_FLUX_EOT_THRESHOLD
    return min(DEFAULT_EAGER_EOT_THRESHOLD, max(0.3, eot - 0.1))
