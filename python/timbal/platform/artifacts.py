"""Run artifacts on the Timbal platform.

Blobs an app run writes for later turns of its conversation to read back: offloaded tool
results, compaction transcripts, attachment text and persisted files. The platform keeps
them in a private store, one namespace per (project, user): a caller reads and writes only
its own, the same owner binding as the run's trace.

Available when the run has an app subject (``platform_config.subject.app_id``), the same
condition the platform tracing provider needs.
"""

from typing import Any
from urllib.parse import quote

from ..errors import PlatformError
from ..state import get_run_context

__all__ = [
    "artifact_exists",
    "artifacts_available",
    "get_artifact",
    "put_artifact",
]


def _base_path() -> str | None:
    run_context = get_run_context()
    config = run_context.platform_config if run_context is not None else None
    subject = config.subject if config is not None else None
    if subject is None or not subject.app_id:
        return None
    return f"orgs/{subject.org_id}/apps/{subject.app_id}/artifacts"


def artifacts_available() -> bool:
    """Whether the current run can store artifacts on the platform."""
    return _base_path() is not None


def _path(key: str) -> str:
    base = _base_path()
    if base is None:
        raise RuntimeError("Run artifacts need a platform config with an app subject.")
    return f"{base}/{'/'.join(quote(segment, safe='') for segment in key.split('/'))}"


async def put_artifact(key: str, data: bytes, content_type: str = "application/octet-stream") -> dict[str, Any]:
    """Store ``data`` under ``key``. Returns ``{"handle": key, "url": <stable unsigned URL>}``."""
    from .utils import _request

    res = await _request("PUT", _path(key), headers={"Content-Type": content_type}, content=data)
    return res.json()


async def get_artifact(key: str) -> bytes:
    """The artifact's bytes. Raises ``FileNotFoundError`` when it doesn't exist."""
    from .utils import _request

    try:
        res = await _request("GET", _path(key))
    except PlatformError as e:
        if e.status_code == 404:
            raise FileNotFoundError(f"No artifact found for key {key!r}.") from e
        raise
    return res.content


async def artifact_exists(key: str) -> bool:
    from .utils import _request

    try:
        await _request("HEAD", _path(key))
    except PlatformError as e:
        if e.status_code == 404:
            return False
        raise
    return True
