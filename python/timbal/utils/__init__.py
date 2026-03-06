# Lazy imports (PEP 562) — submodules are only loaded when their exports are
# first accessed, avoiding the cost of importing the entire dependency chain
# (pydantic, structlog, asyncio, …) at package import time.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .import_spec import ImportSpec
    from .model import create_model_from_handler, issubclass_safe, resolve_default
    from .net import is_port_in_use
    from .schema import SupportedStringFormats, SupportedTypes, assert_never, is_list, transform_schema
    from .serialization import coerce_to_dict, dump, safe_is_nan, sync_to_async_gen

__all__ = [
    # import_spec
    "ImportSpec",
    # schema
    "SupportedTypes",
    "SupportedStringFormats",
    "is_list",
    "assert_never",
    "transform_schema",
    # model
    "create_model_from_handler",
    "issubclass_safe",
    "resolve_default",
    # serialization
    "dump",
    "safe_is_nan",
    "coerce_to_dict",
    "sync_to_async_gen",
    # net
    "is_port_in_use",
]

_LAZY_IMPORTS = {
    # import_spec
    "ImportSpec": ".import_spec",
    # schema
    "SupportedTypes": ".schema",
    "SupportedStringFormats": ".schema",
    "is_list": ".schema",
    "assert_never": ".schema",
    "transform_schema": ".schema",
    # model
    "create_model_from_handler": ".model",
    "issubclass_safe": ".model",
    "resolve_default": ".model",
    # serialization
    "dump": ".serialization",
    "safe_is_nan": ".serialization",
    "coerce_to_dict": ".serialization",
    "sync_to_async_gen": ".serialization",
    # net
    "is_port_in_use": ".net",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache to bypass __getattr__ on subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
