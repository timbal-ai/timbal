"""Push call recordings to the platform (option C of the recording handoff).

Activated per session by ``TIMBAL_VOICE_RECORDING_UPLOAD=platform``. The
contract, agreed with the platform team (final, general-sessions model):

    PUT {host}/orgs/{org}/projects/{project}/sessions/{session_id}
    Authorization: Bearer {token}
    multipart: "manifest" (application/json) + "audio" (audio/mpeg)

Platform-side constraints we satisfy by construction: the path session_id
equals ``manifest.session_id`` (both derive from the recording filename),
session ids are 32 hex chars (within their ``[A-Za-z0-9_-]{1,128}``), and
32 kbps MP3 keeps multi-hour calls far below the 100 MB cap.

Identity and auth come from :func:`resolve_platform_config` — the same
resolution every other platform call uses (env ``TIMBAL_API_HOST`` /
``TIMBAL_API_TOKEN``/``TIMBAL_API_KEY`` / ``TIMBAL_ORG_ID`` /
``TIMBAL_PROJECT_ID``, with ~/.timbal file fallback) — and the request goes
through :func:`timbal.platform.utils._request` with a recording-specific
retry policy:

* 2xx → the platform upserted (idempotent by session_id) → delete both files.
* 4xx (except 429) → permanent (auth/validation): keep files, log once,
  no retry (``_request`` raises immediately).
* 5xx / 429 / network → exponential backoff (1s base, ×5, cap 5 min),
  ~1h total budget. Files stay on disk the whole time — the platform
  sweeper (or nothing, on ephemeral ECS disk) is the backstop.

Uploads are fire-and-forget background tasks: they never block session
teardown, and a crash mid-upload leaves both files intact because deletion
only ever happens after a 2xx.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import structlog

from ..errors import PlatformError
from ..platform.utils import _request
from ..state.config_loader import resolve_platform_config

logger = structlog.get_logger("timbal.server.recording_upload")

# Keep strong refs so fire-and-forget upload tasks aren't garbage-collected.
_upload_tasks: set[asyncio.Task[Any]] = set()


async def drain_upload_tasks() -> None:
    """Block until every in-flight recording upload has finished.

    Single-session boxes call this before exiting: the process is the only
    thing holding the recording, so the exit must wait for the platform PUT.
    Bounded by the upload's own retry budget (~1h), which sits under the
    platform's hard session cap.
    """
    while _upload_tasks:
        await asyncio.gather(*list(_upload_tasks), return_exceptions=True)


def _recording_backoff(attempt: int) -> float:
    """1, 5, 25, 125, then 300s flat — the agreed recording retry curve."""
    return min(1.0 * 5.0**attempt, 300.0)


# 14 retries on that curve ≈ 53 min of waits (+ request time) ≈ the ~1h budget.
_MAX_RETRIES = 14

# Multipart bodies up to ~100 MB (platform cap) on slow links: no write/read
# deadline, keep only the connect/pool timeout.
_UPLOAD_TIMEOUT = httpx.Timeout(30.0, read=None, write=None)


async def upload_recording(
    audio_path: Path,
    manifest_path: Path,
    *,
    path: str,
    max_retries: int = _MAX_RETRIES,
    backoff: Callable[[int], float] = _recording_backoff,
) -> bool:
    """PUT both files to the platform; delete them on success. Returns success.

    ``path`` is the platform API path (relative to the platform host, which
    ``_request`` resolves together with auth headers).
    """
    session_id = audio_path.stem
    try:
        files = {
            "manifest": (manifest_path.name, manifest_path.read_bytes(), "application/json"),
            "audio": (audio_path.name, audio_path.read_bytes(), "audio/mpeg"),
        }
    except OSError as e:
        logger.error("recording_upload_files_missing", session_id=session_id, error=str(e))
        return False

    try:
        response = await _request(
            "PUT",
            path,
            files=files,
            max_retries=max_retries,
            backoff=backoff,
            timeout=_UPLOAD_TIMEOUT,
        )
    except PlatformError as e:
        if e.status_code is not None and 400 <= e.status_code < 500 and e.status_code != 429:
            # Permanent (auth/validation): keep the files, don't hammer.
            logger.error("recording_upload_rejected", session_id=session_id, status=e.status_code, path=path)
        else:
            logger.error(
                "recording_upload_failed",
                session_id=session_id,
                status=e.status_code,
                error=str(e),
                hint="files kept on disk for sweeper/manual ingest",
            )
        return False
    except Exception as e:  # network/timeout retries exhausted
        logger.error(
            "recording_upload_failed",
            session_id=session_id,
            error=str(e),
            hint="files kept on disk for sweeper/manual ingest",
        )
        return False

    audio_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    try:
        body = response.json()  # {"session_id", "created", "item_count"}
    except Exception:
        body = {}
    logger.info(
        "recording_upload_ok",
        session_id=session_id,
        created=body.get("created"),
        item_count=body.get("item_count"),
    )
    return True


def platform_recording_upload_hook() -> Callable[[Any], Awaitable[None]] | None:
    """Build the ``on_saved`` hook from platform config, or None (with a log).

    ``force_refresh``: serverless session boxes are CRIU-restored from warm
    snapshots with env arriving at restore time — a boot-time default
    resolution may have cached ``None``, and this hook runs per session.
    """
    config = resolve_platform_config(force_refresh=True)
    subject = config.subject if config is not None else None
    if config is None or subject is None or not subject.org_id or not subject.project_id:
        logger.warning(
            "recording_upload_misconfigured",
            has_platform_config=config is not None,
            has_org=bool(subject and subject.org_id),
            has_project=bool(subject and subject.project_id),
            hint=(
                "TIMBAL_VOICE_RECORDING_UPLOAD=platform needs TIMBAL_API_HOST, "
                "TIMBAL_API_TOKEN|TIMBAL_API_KEY, TIMBAL_ORG_ID and TIMBAL_PROJECT_ID"
            ),
        )
        return None
    path_prefix = f"orgs/{subject.org_id}/projects/{subject.project_id}/sessions"

    async def _hook(result: Any) -> None:
        """on_saved: schedule the upload and return immediately (never blocks teardown)."""
        if result.manifest_path is None:
            logger.warning("recording_upload_skipped", reason="no_manifest", path=str(result.audio_path))
            return
        task = asyncio.create_task(
            upload_recording(
                result.audio_path,
                result.manifest_path,
                path=f"{path_prefix}/{result.audio_path.stem}",
            )
        )
        _upload_tasks.add(task)
        task.add_done_callback(_upload_tasks.discard)

    return _hook
