"""Minimal pydantic-free model machinery for hot-path framework objects.

Events, ``Span`` and ``RunStatus`` are constructed on every runnable call.
Pydantic's validation, mutable-default deepcopies and ``__getattr__``-routed
private attributes made them a measurable share of per-run overhead, so they
are plain ``__slots__`` classes built on this base instead. The pydantic API
surface consumers rely on (``model_dump()``, ``model_dump_json()``, field
equality) is preserved.
"""

import json
from typing import Any

from pydantic import BaseModel


def dump_value(value: Any, mode: str = "python") -> Any:
    """Recursively convert a value the way pydantic's model_dump would.

    Nested SlotModel/BaseModel instances become dicts in both modes. In
    ``json`` mode, values that json.dumps can't handle are stringified.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, SlotModel):
        return value.model_dump(mode=mode)
    if isinstance(value, BaseModel):
        return value.model_dump(mode=mode)
    if isinstance(value, dict):
        return {k: dump_value(v, mode) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [dump_value(v, mode) for v in value]
    if mode == "json":
        return str(value)
    return value


class SlotModel:
    """Base for plain ``__slots__`` classes with a pydantic-compatible surface.

    Subclasses declare ``__slots__`` and list their public fields (in dump
    order) in ``_FIELDS``. ``type``-style class-level constants may appear in
    ``_FIELDS`` without a slot.
    """

    __slots__ = ()

    # Marker for timbal.utils.serialization.dump() — routes through model_dump().
    __timbal_serializable__ = True

    _FIELDS: tuple[str, ...] = ()

    def model_dump(self, mode: str = "python", **_kwargs: Any) -> dict[str, Any]:
        return {f: dump_value(getattr(self, f, None), mode) for f in self._FIELDS}

    def model_dump_json(self, **_kwargs: Any) -> str:
        return json.dumps(self.model_dump(mode="json"))

    def __eq__(self, other: Any) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return all(getattr(self, f, None) == getattr(other, f, None) for f in self._FIELDS)

    __hash__ = None  # Mutable, like pydantic models — unhashable.

    def __repr__(self) -> str:
        fields = ", ".join(f"{f}={getattr(self, f, None)!r}" for f in self._FIELDS)
        return f"{type(self).__name__}({fields})"
