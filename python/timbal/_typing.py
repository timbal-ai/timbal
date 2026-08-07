"""Shared TypeVars for static typing of Runnable / collector / event payloads.

These exist only for the type checker. Runtime behaviour is unchanged — Generic
bases add ``__class_getitem__`` but no validators or extra fields.
"""

from typing import Any

from typing_extensions import TypeVar

PayloadT = TypeVar("PayloadT", default=Any)
"""Payload inside ``OutputEvent.output`` / ``TimbalCollector`` / ``Runnable``."""

CollectT = TypeVar("CollectT")
"""Result type returned by ``BaseCollector.collect()`` / ``.result()``."""
