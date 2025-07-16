from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from .base import BaseStep

RunnableLike = BaseStep | Callable[..., Any]


class RemoteConfig(BaseModel):
    """Configuration for remote app execution."""

    org_id: str
    """The organization ID."""
    app_id: str
    """The application ID."""
    version_id: str | None = None
    """The version ID.
    If none, the default version will be used.
    """
