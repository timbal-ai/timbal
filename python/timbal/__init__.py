# pyright: reportUnsupportedDunderAll=false

try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0.dev0"

__all__ = ["Agent", "Tool", "Workflow"]

_LAZY_IMPORTS = {
    "Agent": ".core",
    "Tool": ".core",
    "Workflow": ".core",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        mod = importlib.import_module(_LAZY_IMPORTS[name], __name__)
        val = getattr(mod, name)
        globals()[name] = val  # cache to bypass __getattr__ on subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
