"""Durable chat sessions on the Timbal platform (monolith#567: the ``SessionRuntime`` side table).

The contract it speaks (``rfcs/session-runtime.md``), app-scoped like runs, under
``orgs/{org}/apps/{app}/session_runtime``:

- ``GET /{session_id}`` → ``{runtime, runtime_version, write_id, lease_owner, lease_until}`` | 404.
- ``PUT /{session_id}`` with ``If-Match: "<version>"`` (``If-None-Match: *`` to create), body
  ``{runtime, has_work, closed, root_run_id, write_id}`` → ``{runtime_version}``; 412 when the version moved.
- ``DELETE /{session_id}`` → 204 (also its inbox).
- ``GET ?with_work=true`` → ``{session_ids}`` (open sessions; ``with_work``: ``has_work`` or inbox items).
- ``POST /{session_id}/lease`` ``{owner, ttl_s}`` → ``{lease_owner, lease_until}``; 409 while another
  owner's lease runs. ``DELETE /{session_id}/lease?owner=`` → 204.
- ``POST /{session_id}/inbox`` ``{kind, body}`` → 201 ``{id}``; 404 for no session.
- ``GET /{session_id}/inbox?owner=&wait_ms=`` → ``{items}`` (long poll); 409 when ``owner`` does not hold the lease.
- ``POST /{session_id}/inbox/ack`` ``{owner, ids}`` → 204.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import TYPE_CHECKING, Any

from uuid_extensions import uuid7

from ..errors import PlatformError
from .chat_store import (
    INBOX_KINDS,
    ChatSessionStore,
    SessionLeased,
    SessionSnapshot,
    SnapshotConflict,
)

if TYPE_CHECKING:
    from .config import PlatformConfig

__all__ = ["PlatformChatSessionStore"]


class PlatformChatSessionStore(ChatSessionStore):
    """Snapshots, leases and the inbox of durable chat sessions, kept by the platform.

    Credentials: ``platform_config``, else the environment (``TIMBAL_API_KEY``, ``TIMBAL_ORG_ID``,
    ``TIMBAL_APP_ID``). Every worker of an app that uses it can serve any of its sessions: one holds a
    session's lease, the others queue work for it in the inbox.
    """

    supports_inbox = True

    def __init__(self, platform_config: PlatformConfig | None = None) -> None:
        self.platform_config = platform_config
        # session id -> the write_id of its last save that failed without an answer (it may have landed)
        self._uncertain: dict[str, str] = {}

    def _config(self) -> PlatformConfig:
        cfg = self.platform_config
        if cfg is None:
            from .config_loader import resolve_platform_config

            cfg = resolve_platform_config(None)
        if cfg is None or cfg.subject is None or not cfg.subject.app_id:
            raise RuntimeError(
                "PlatformChatSessionStore needs platform credentials and an app: set TIMBAL_API_KEY, "
                "TIMBAL_ORG_ID and TIMBAL_APP_ID, or pass platform_config=."
            )
        return cfg

    async def _call(self, method: str, path: str = "", **kwargs: Any) -> Any:
        """One request as the store's own principal, whatever run is current in the calling task."""
        from ..platform.utils import _request
        from . import get_run_context, set_run_context
        from .context import RunContext
        from .tracing.providers.platform import _pooled

        cfg = self._config()
        caller = RunContext()
        caller.platform_config = cfg
        previous = get_run_context()
        set_run_context(caller)
        try:
            with _pooled():
                base = f"orgs/{cfg.subject.org_id}/apps/{cfg.subject.app_id}/session_runtime"
                return await _request(method=method, path=f"{base}/{path}" if path else base, **kwargs)
        finally:
            set_run_context(previous)

    async def load(self, session_id: str) -> SessionSnapshot | None:
        try:
            res = await self._call("GET", session_id)
        except PlatformError as e:
            if e.status_code == 404:
                return None
            raise
        data = res.json()
        snapshot = SessionSnapshot.from_json(_dumps(data["runtime"]))
        snapshot.version = int(data["runtime_version"])
        return snapshot

    async def save(self, snapshot: SessionSnapshot, *, expect_version: int | None) -> int:
        runtime = asdict(snapshot)
        runtime.pop("version", None)
        sid = snapshot.session_id
        write_id = uuid7(as_type="hex")
        body = {
            "runtime": runtime,
            "has_work": snapshot.has_work(),
            "closed": snapshot.closed,
            "root_run_id": runtime.get("root_run_id"),
            "write_id": write_id,
        }
        content = _dumps(body).encode()
        try:
            version = await self._put(sid, content, expect_version)
        except SnapshotConflict:
            # The last save failed without an answer and may have landed (a timeout after commit). If the stored
            # row is that very write, nobody else wrote since: this save goes on top of it.
            uncertain = self._uncertain.pop(sid, None)
            if uncertain is None:
                raise
            stored = (await self._call("GET", sid)).json()
            if stored.get("write_id") != uncertain:
                raise
            version = await self._put(sid, content, int(stored["runtime_version"]))
        except BaseException:
            self._uncertain[sid] = write_id
            raise
        self._uncertain.pop(sid, None)
        return version

    async def _put(self, session_id: str, content: bytes, expect_version: int | None) -> int:
        headers = {"Content-Type": "application/json"}
        headers.update({"If-None-Match": "*"} if expect_version is None else {"If-Match": f'"{expect_version}"'})
        try:
            # Not retried here: a retry of a save that landed would read as a conflict (see save).
            res = await self._call("PUT", session_id, headers=headers, content=content, max_retries=0)
        except PlatformError as e:
            if e.status_code in (409, 412):
                raise SnapshotConflict(f"session {session_id}: not at version {expect_version} on the platform") from e
            raise
        return int(res.json()["runtime_version"])

    async def delete(self, session_id: str) -> None:
        try:
            await self._call("DELETE", session_id)
        except PlatformError as e:
            if e.status_code != 404:
                raise

    async def list_ids(self, *, with_work: bool = False) -> list[str]:
        res = await self._call("GET", params={"with_work": "true"} if with_work else None)
        return list(res.json().get("session_ids") or [])

    async def acquire_lease(self, session_id: str, owner: str) -> None:
        try:
            await self._call("POST", f"{session_id}/lease", json={"owner": owner, "ttl_s": self.lease_ttl_s})
        except PlatformError as e:
            if e.status_code == 404:
                return  # nothing stored yet: the first save takes it
            if e.status_code == 409:
                raise SessionLeased(f"session {session_id} is held by another owner on the platform") from e
            raise

    async def release_lease(self, session_id: str, owner: str) -> None:
        try:
            await self._call("DELETE", f"{session_id}/lease", params={"owner": owner})
        except PlatformError as e:
            if e.status_code not in (404, 409):
                raise

    async def post_inbox(self, session_id: str, kind: str, body: dict[str, Any]) -> str:
        if kind not in INBOX_KINDS:
            raise ValueError(f"kind must be one of {list(INBOX_KINDS)}, got {kind!r}")
        try:
            res = await self._call("POST", f"{session_id}/inbox", json={"kind": kind, "body": body}, max_retries=0)
        except PlatformError as e:
            if e.status_code == 404:
                raise LookupError(f"No stored chat session {session_id!r}.") from e
            raise
        return str(res.json()["id"])

    async def pull_inbox(self, session_id: str, owner: str, *, wait_s: float = 0.0) -> list[dict[str, Any]]:
        import httpx

        try:
            res = await self._call(
                "GET",
                f"{session_id}/inbox",
                params={"owner": owner, "wait_ms": int(wait_s * 1000)},
                timeout=httpx.Timeout(10.0, read=wait_s + 15.0),
            )
        except PlatformError as e:
            if e.status_code == 404:
                raise LookupError(f"No stored chat session {session_id!r}.") from e
            if e.status_code == 409:
                raise SessionLeased(f"session {session_id} is not leased to {owner} on the platform") from e
            raise
        return list(res.json().get("items") or [])

    async def ack_inbox(self, session_id: str, owner: str, ids: list[str]) -> None:
        if ids:
            await self._call("POST", f"{session_id}/inbox/ack", json={"owner": owner, "ids": list(ids)})


def _dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
