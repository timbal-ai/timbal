# ruff: noqa: F401
import warnings

# Filter SWIG-related deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type swigvarlink.*")

try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0.dev0"

# Lazy imports - only load when accessed
__all__ = ["Agent", "Tool", "Workflow", "MCPToolSet"]

def __getattr__(name: str):
    """Lazy import mechanism for top-level exports.
    
    This defers imports until they're actually accessed, preventing
    unnecessary module loading when only importing the package.
    """
    if name == "Agent":
        from .core import Agent
        return Agent
    elif name == "Tool":
        from .core import Tool
        return Tool
    elif name == "Workflow":
        from .core import Workflow
        return Workflow
    elif name == "MCPToolSet":
        from .core import MCPToolSet
        return MCPToolSet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
