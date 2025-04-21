import warnings

# Filter SWIG-related deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type swigvarlink.*")


from . import errors, state, steps, types
from .core import Agent, Flow

try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0.dev0"


__all__ = [
    "__version__",
    "Agent",
    "Flow",
    "errors",
    "state",
    "steps",
    "types",
]
