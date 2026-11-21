"""Tests for the shared rotating-refresh-token store (token file, locks, fallbacks)."""

import asyncio
import errno
import os
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from timbal.tools import _oauth_store as store


class Refused(Exception):
    pass


@pytest.fixture(autouse=True)
def _clean() -> None:
    store.clear()


def _rotating(valid: set[str] | None = None, *, delay: float = 0.0, expires_in: int | None = 3600):
    """An exchange() that rotates: each refresh token works once and yields the next one."""
    alive = set(valid) if valid is not None else {"rt1"}
    spent: list[str] = []
    guard = threading.Lock()

    async def exchange(token: str) -> dict[str, Any]:
        with guard:
            spent.append(token)
            ok = token in alive
            alive.discard(token)
            n = len(spent)
            alive.add(f"rotated-{n}")
        if delay:
            await asyncio.sleep(delay)
        if not ok:
            raise Refused(token)
        body: dict[str, Any] = {"access_token": f"at-{n}", "refresh_token": f"rotated-{n}"}
        if expires_in is not None:
            body["expires_in"] = expires_in
        return body

    return exchange, spent


def _get(exchange, refresh_token: str | None = "rt1", token_file: Any = None, rejected: str | None = None):
    return store.refreshed_access_token(
        key=store.cache_key("t", str(token_file or refresh_token)),
        owner={"client_id": "cid"},
        refresh_token=refresh_token,
        token_file=token_file,
        exchange=exchange,
        is_refused=lambda exc: isinstance(exc, Refused),
        rejected=rejected,
    )


@pytest.mark.asyncio
async def test_rotation_survives_a_restart_via_token_file(tmp_path: Any) -> None:
    exchange, spent = _rotating(expires_in=None)
    token_file = tmp_path / "t.json"
    assert await _get(exchange, token_file=token_file) == "at-1"
    assert store.read_token_file(token_file)["refresh_token"] == "rotated-1"
    store.clear()  # a new process: the configured token (rt1) is spent; the file has the newest
    assert await _get(exchange, token_file=token_file, rejected="at-1") == "at-2"
    assert spent == ["rt1", "rotated-1"]


@pytest.mark.asyncio
async def test_token_without_expiry_is_reused_until_rejected(tmp_path: Any) -> None:
    exchange, spent = _rotating(expires_in=None)
    token_file = tmp_path / "t.json"
    assert await _get(exchange, token_file=token_file) == "at-1"
    assert await _get(exchange, token_file=token_file) == "at-1"
    assert spent == ["rt1"]
    assert await _get(exchange, token_file=token_file, rejected="at-1") == "at-2"


@pytest.mark.asyncio
async def test_rejected_token_refreshed_once_by_concurrent_callers(tmp_path: Any) -> None:
    exchange, spent = _rotating(expires_in=None, delay=0.05)
    token_file = tmp_path / "t.json"
    await _get(exchange, token_file=token_file)
    # Five calls see the same 401 at once: one refresh, the others reuse its result.
    tokens = await asyncio.gather(*(_get(exchange, token_file=token_file, rejected="at-1") for _ in range(5)))
    assert tokens == ["at-2"] * 5
    assert spent == ["rt1", "rotated-1"]


@pytest.mark.asyncio
async def test_concurrent_first_calls_spend_the_token_once(tmp_path: Any) -> None:
    exchange, spent = _rotating(delay=0.05)
    tokens = await asyncio.gather(*(_get(exchange, token_file=tmp_path / "t.json") for _ in range(5)))
    assert tokens == ["at-1"] * 5
    assert spent == ["rt1"]


@pytest.mark.asyncio
async def test_refused_file_token_falls_back_to_a_new_sign_in(tmp_path: Any) -> None:
    exchange, spent = _rotating({"fresh"})
    token_file = tmp_path / "t.json"
    store.write_token_file(token_file, {"client_id": "cid", "refresh_token": "revoked"})
    assert await _get(exchange, refresh_token="fresh", token_file=token_file) == "at-2"
    assert spent == ["revoked", "fresh"]


@pytest.mark.asyncio
async def test_other_errors_are_not_retried_with_older_tokens(tmp_path: Any) -> None:
    calls: list[str] = []

    async def exchange(token: str) -> dict[str, Any]:
        calls.append(token)
        raise RuntimeError("bad client secret")

    token_file = tmp_path / "t.json"
    store.write_token_file(token_file, {"client_id": "cid", "refresh_token": "from-file"})
    with pytest.raises(RuntimeError):
        await _get(exchange, refresh_token="from-env", token_file=token_file)
    assert calls == ["from-file"]


@pytest.mark.asyncio
async def test_token_file_of_another_owner_is_ignored(tmp_path: Any) -> None:
    exchange, spent = _rotating()
    token_file = tmp_path / "t.json"
    store.write_token_file(token_file, {"client_id": "someone-else", "refresh_token": "theirs"})
    assert await _get(exchange, token_file=token_file) == "at-1"
    assert spent == ["rt1"]


@pytest.mark.asyncio
async def test_failed_write_keeps_the_new_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    exchange, spent = _rotating()

    def read_only(*_args: Any, **_kwargs: Any) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(store, "write_token_file", read_only)
    assert await _get(exchange, token_file=tmp_path / "t.json") == "at-1"
    assert await _get(exchange, token_file=tmp_path / "t.json") == "at-1"
    assert spent == ["rt1"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_token_file_is_owner_only(tmp_path: Any) -> None:
    token_file = tmp_path / "t.json"
    store.write_token_file(token_file, {"refresh_token": "x"})
    assert os.stat(token_file).st_mode & 0o777 == 0o600
    assert store.read_token_file(token_file) == {"refresh_token": "x"}
    assert store.read_token_file(tmp_path / "missing.json") is None


def test_processes_sharing_a_token_file_spend_it_once(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """Two event loops in two threads, with no shared cache, stand in for two processes."""

    class NoCache(dict):
        def __setitem__(self, key: str, value: Any) -> None:
            return None

    monkeypatch.setattr(store, "_TOKEN_CACHE", NoCache())
    exchange, spent = _rotating(delay=0.2)
    token_file = tmp_path / "t.json"
    store.write_token_file(token_file, {"client_id": "cid", "refresh_token": "rt1"})
    results: list[Any] = []

    def worker() -> None:
        try:
            results.append(asyncio.run(_get(exchange, refresh_token=None, token_file=token_file)))
        except Exception as exc:  # collected, asserted below
            results.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == ["at-1", "at-1"]
    assert spent == ["rt1"]


@pytest.mark.asyncio
async def test_needs_a_refresh_token_or_a_file() -> None:
    exchange, _ = _rotating()
    with pytest.raises(ValueError, match="refresh token or a token file"):
        await _get(exchange, refresh_token=None)


def _fake_msvcrt(monkeypatch: pytest.MonkeyPatch, failures: list[OSError]) -> list[int]:
    """Run ``file_lock``'s Windows branch on any OS: ``locking`` raises ``failures`` in turn, then succeeds."""
    calls: list[int] = []

    def locking(_fd: int, mode: int, _nbytes: int) -> None:
        calls.append(mode)
        if mode == 1 and failures:
            raise failures.pop(0)

    monkeypatch.setitem(sys.modules, "msvcrt", SimpleNamespace(LK_LOCK=1, LK_UNLCK=0, locking=locking))
    monkeypatch.setattr(store.os, "name", "nt")
    return calls


def test_windows_file_lock_waits_out_contention(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    held = OSError(errno.EACCES, "held by another process")
    calls = _fake_msvcrt(monkeypatch, [held, held])
    with store.file_lock(str(tmp_path / "token.json")):
        pass
    assert calls == [1, 1, 1, 0]


def test_windows_file_lock_raises_other_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    calls = _fake_msvcrt(monkeypatch, [OSError(errno.EINVAL, "locking not supported")])
    with pytest.raises(OSError, match="locking not supported"):
        with store.file_lock(str(tmp_path / "token.json")):
            pass
    assert calls == [1]
