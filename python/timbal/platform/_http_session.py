"""Run-scoped connection reuse for buffered platform requests.

Scopes own their pools; there is no process-global or cross-event-loop client.
Retiring a scope stops new borrowers but lets requests already in flight finish.
The last borrower then closes the pool. Detached tasks with an inherited retired
scope fall back to a short-lived client on their next request.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from contextvars import ContextVar
from http.cookiejar import CookieJar, DefaultCookiePolicy

import httpx


class _NoResponseCookies(DefaultCookiePolicy):
    def set_ok(self, cookie, request):  # noqa: ARG002 - CookiePolicy interface
        # The previous per-request clients did not carry response cookies
        # between calls. Preserve that isolation even when auth changes.
        return False


class _Session:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.pid = os.getpid()
        self.client: httpx.AsyncClient | None = None
        self.borrowers = 0
        self.retired = False

    def usable(self) -> bool:
        return not self.retired and self.loop is asyncio.get_running_loop() and self.pid == os.getpid()

    async def close_if_idle(self) -> None:
        if self.retired and self.borrowers == 0 and self.client is not None:
            client, self.client = self.client, None
            await client.aclose()


_SESSION: ContextVar[_Session | None] = ContextVar("timbal_platform_http_session", default=None)


@asynccontextmanager
async def platform_http_session():
    """Reuse connections within one runner invocation, creating clients lazily.

    Nested scopes reuse their live parent. Retiring a scope does not wait for
    detached work: requests in progress retain ownership until their own cleanup.
    """
    inherited = _SESSION.get()
    if inherited is not None and inherited.usable():
        yield
        return
    session = _Session()
    token = _SESSION.set(session)
    try:
        yield
    finally:
        _SESSION.reset(token)
        session.retired = True
        await session.close_if_idle()


@asynccontextmanager
async def request_client(timeout: httpx.Timeout):
    session = _SESSION.get()
    if session is None or not session.usable():
        async with httpx.AsyncClient(timeout=timeout) as client:
            yield client
        return
    # No await between eligibility, construction and borrowing. Sibling tasks
    # on this event loop cannot retire the session between those operations.
    if session.client is None:
        session.client = httpx.AsyncClient(
            timeout=timeout,
            cookies=CookieJar(policy=_NoResponseCookies()),
        )
    session.borrowers += 1
    try:
        yield session.client
    finally:
        session.borrowers -= 1
        await session.close_if_idle()
