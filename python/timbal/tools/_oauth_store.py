"""Refresh-token handling shared by OAuth tools whose provider rotates refresh tokens.

Some providers (Notion, Qlik Cloud) issue a new refresh token on every refresh and may stop
accepting the old one. Keeping the newest one only in memory breaks the next process: it starts
again from the configured, already spent token. This module keeps it durable and spends it once:

- an optional **token file** (JSON, written atomically, owner-only) holds the newest token pair,
  so restarts and every process sharing the file use the current one;
- an **asyncio lock** per credential set (per event loop) and an **OS file lock** on
  ``<token_file>.lock`` serialize exchanges inside a process and across processes;
- if the provider refuses the file's refresh token and a different one is configured (the user
  signed in again), that one is tried.

Access tokens with a known lifetime are reused until shortly before they expire. Without one
(``expires_in`` absent) they are reused until the provider rejects them; the caller then asks for
a forced refresh, passing the rejected token so that concurrent callers refresh only once.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import time
import weakref
from collections.abc import Awaitable, Callable, Iterator
from typing import Any

import structlog

logger = structlog.get_logger("timbal.tools.oauth_store")

#: Seconds before expiry at which a cached access token is renewed.
TOKEN_SKEW = 60

# key -> (access token, expires at or None when unknown, newest refresh token)
_TOKEN_CACHE: dict[str, tuple[str, float | None, str | None]] = {}
_LOCKS: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]] = weakref.WeakKeyDictionary()


def cache_key(*parts: str) -> str:
    """A stable key for one credential set; secrets are hashed, never stored as keys."""
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def clear() -> None:
    """Forget cached tokens and locks (tests)."""
    _TOKEN_CACHE.clear()
    _LOCKS.clear()


def async_lock(key: str) -> asyncio.Lock:
    """One lock per credential set and event loop (an asyncio.Lock must not cross loops)."""
    locks = _LOCKS.setdefault(asyncio.get_running_loop(), {})
    lock = locks.get(key)
    if lock is None:
        lock = locks[key] = asyncio.Lock()
    return lock


def expires_at_from(body: dict[str, Any]) -> float | None:
    """Absolute expiry from a token response's ``expires_in``, or None when it has none."""
    raw = body.get("expires_in")
    if raw in (None, ""):
        return None
    try:
        return time.time() + max(60, int(raw))
    except (TypeError, ValueError):
        return None


def is_fresh(expires_at: float | None, now: float | None = None) -> bool:
    return expires_at is None or expires_at - (now or time.time()) > TOKEN_SKEW


def read_token_file(path: str | os.PathLike[str]) -> dict[str, Any] | None:
    """The token state saved by ``write_token_file``, or None when the file is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("refresh_token") else None


def write_token_file(path: str | os.PathLike[str], data: dict[str, Any]) -> None:
    """Save token state atomically, readable by the owner only (it holds a live refresh token)."""
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(target)}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


@contextlib.contextmanager
def file_lock(path: str) -> Iterator[None]:
    """Exclusive OS lock on ``<path>.lock`` while a token file is read, exchanged and rewritten."""
    lock_path = f"{path}.lock"
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.name == "nt":
            import msvcrt

            from ..state.tracing.providers.jsonl import _is_windows_lock_contention_error

            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_LOCK, 1)  # retries for ~10 s, then raises
                    break
                except OSError as exc:
                    # Only "held by another process" is worth waiting out; anything else
                    # (a filesystem without locks, a bad fd) would loop forever.
                    if not _is_windows_lock_contention_error(exc):
                        raise
            try:
                yield
            finally:
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


Exchange = Callable[[str], Awaitable[dict[str, Any]]]


async def refreshed_access_token(
    *,
    key: str,
    owner: dict[str, str],
    refresh_token: str | None,
    token_file: str | os.PathLike[str] | None,
    exchange: Exchange,
    is_refused: Callable[[Exception], bool],
    rejected: str | None = None,
) -> str:
    """An access token for this credential set, refreshing (and persisting the rotation) when needed.

    ``owner`` identifies whom a token file belongs to (e.g. ``{"client_id": ...}``); a file saved
    for someone else is ignored. ``exchange(refresh_token)`` performs the provider's refresh grant
    and returns its JSON. ``is_refused(exc)`` says whether an exchange error means "this refresh
    token is no good" (worth trying an older candidate) rather than a client or network fault.
    ``rejected`` is an access token the provider just refused: it is never returned again.
    """
    if token_file is None and not refresh_token:
        raise ValueError("A refresh token or a token file is required.")
    path = os.path.abspath(os.fspath(token_file)) if token_file is not None else None
    async with async_lock(key):
        if path is None:
            return await _refresh_locked(key, owner, refresh_token, None, exchange, is_refused, rejected)
        lock = file_lock(path)
        # Waiting for another process blocks, so wait in a thread. If this call is cancelled
        # meanwhile, the thread still gets the lock later: release it as soon as it does.
        entering = asyncio.ensure_future(asyncio.to_thread(lock.__enter__))
        try:
            await asyncio.shield(entering)
        except asyncio.CancelledError:
            entering.add_done_callback(lambda f: None if f.exception() else lock.__exit__(None, None, None))
            raise
        try:
            return await _refresh_locked(key, owner, refresh_token, path, exchange, is_refused, rejected)
        finally:
            lock.__exit__(None, None, None)


async def _refresh_locked(
    key: str,
    owner: dict[str, str],
    refresh_token: str | None,
    token_file: str | None,
    exchange: Exchange,
    is_refused: Callable[[Exception], bool],
    rejected: str | None,
) -> str:
    now = time.time()
    cached = _TOKEN_CACHE.get(key)
    if cached and cached[0] != rejected and is_fresh(cached[1], now):
        return cached[0]

    stored = read_token_file(token_file) if token_file else None
    if stored and any(stored.get(k) not in (None, v) for k, v in owner.items()):
        stored = None  # saved for another tenant or client
    if stored and isinstance(stored.get("access_token"), str) and stored["access_token"] != rejected:
        expires_at = stored.get("expires_at")
        expires_at = float(expires_at) if expires_at is not None else None
        if is_fresh(expires_at, now):
            _TOKEN_CACHE[key] = (stored["access_token"], expires_at, stored["refresh_token"])
            return stored["access_token"]

    # Newest first: the file, then what this process last saw, then the configured token.
    candidates = [t for t in ((stored or {}).get("refresh_token"), cached[2] if cached else None, refresh_token) if t]
    candidates = list(dict.fromkeys(candidates))
    if not candidates:
        raise ValueError(f"Token file {token_file!r} holds no refresh token; sign in again.")
    body: dict[str, Any] | None = None
    used = candidates[0]
    for i, candidate in enumerate(candidates):
        try:
            body = await exchange(candidate)
            used = candidate
            break
        except Exception as exc:
            # Only a refused token is worth retrying with an older candidate; a bad client
            # id/secret or an outage fails the same way for all of them.
            if not is_refused(exc) or i == len(candidates) - 1:
                raise
    assert body is not None

    newest = body.get("refresh_token") or used
    expires_at = expires_at_from(body)
    _TOKEN_CACHE[key] = (body["access_token"], expires_at, newest)
    if token_file:
        try:
            write_token_file(
                token_file,
                {**owner, "refresh_token": newest, "access_token": body["access_token"], "expires_at": expires_at},
            )
        except OSError as exc:
            # The old refresh token is already spent: failing the call would lose the new one too.
            # Keep it in memory for this process and say why the next one may fail.
            logger.warning("oauth_token_file_write_failed", path=token_file, error=str(exc))
    return body["access_token"]
