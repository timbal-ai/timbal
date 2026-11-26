"""The live voice session, as a context variable the core can read without importing ``timbal.voice``.

``timbal.voice.session`` sets it for everything a running call starts (its turns, their tools, tasks
spawned from either). Core code checks it first, so a run outside a call pays one ``ContextVar.get``.
"""

from __future__ import annotations

import contextvars
from typing import Any

CURRENT_VOICE_SESSION: contextvars.ContextVar[Any] = contextvars.ContextVar("timbal_voice_session", default=None)
