# ruff: noqa: F401
import warnings

from .core import Agent, Tool, Workflow

# Filter SWIG-related deprecation warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyPacked.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type SwigPyObject.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*builtin type swigvarlink.*")


try:
    from ._version import __version__  # type: ignore
except ImportError:
    __version__ = "0.0.0.dev0"
