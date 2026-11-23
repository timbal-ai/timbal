"""PlatformTracingProvider runs stored somewhere other than ``apps/{app_id}/runs`` (issue #309).

A subclass overrides ``_run_path`` (and its own ``_resolve_config``) and gets the whole save pipeline: span
deltas, gzip, snapshot ordering, keep-alive saves, reads. What a host advertises is learned per family of
run paths, not per host: two endpoints on one host need not accept the same formats, and a format one of
them refused must not be turned off for the other. Strict reads (``get_strict``: restores, recoveries,
approval claims) read the whole trace, never the ``select=resume`` projection.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

import pytest
from structlog.testing import capture_logs
from timbal import Agent, Tool
from timbal.core.test_model import TestModel
from timbal.errors import PlatformError
from timbal.state import set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers import platform as platform_module
from timbal.state.tracing.providers.platform import PlatformTracingProvider
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

from .test_platform_deltas import CONFIG, FakePlatform, awaited_saves, clean  # noqa: F401

HOST = CONFIG.host  # one host, two endpoints
APP_PREFIX = f"orgs/{CONFIG.subject.org_id}/apps/{CONFIG.subject.app_id}/runs/"
CHAT_PREFIX = "orgs/1/chat/runs/"

# The other endpoint's runs belong to no app: an org-only subject, or none at all.
ORG_ONLY = PlatformConfig(
    host=HOST,
    auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_chat"),
    subject=PlatformSubject(org_id="1"),
)
NO_SUBJECT = PlatformConfig(host=HOST, auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_chat"))


class ChatRuns(PlatformTracingProvider):
    """Stores runs at another endpoint of the same host, with credentials of its own: only these two hooks.

    Its config is not written onto the run: the run's own config serves the run's other platform calls."""

    config: PlatformConfig = ORG_ONLY

    @classmethod
    def _run_path(cls, platform_config, run_id):  # noqa: ARG003 - one org in these tests
        return f"{CHAT_PREFIX}{run_id}"

    @classmethod
    def _resolve_config(cls, run_context):  # noqa: ARG003
        return cls.config


class Endpoint(FakePlatform):
    """One route family of the fake host. ``select``: it implements (and advertises) the resume projection,
    the root and every span with a ``memory`` key. A route that does not know the parameter refuses it."""

    def __init__(self, *, select=False, **kwargs):
        super().__init__(**kwargs)
        self.select = select
        self.calls: list[dict] = []  # every request: method, run id, params, snapshot, content type

    def _headers(self):
        headers = super()._headers()
        if self.advertise and self.select:
            headers = {**headers, "X-Timbal-Select": "resume"}
        return headers

    async def request(self, method, path, headers=None, content=None, params=None, **kwargs):
        headers = headers or {}
        self.calls.append(
            {
                "method": method,
                "run": path.rsplit("/", 1)[-1],
                "params": params,
                "snapshot": headers.get("X-Timbal-Snapshot"),
                "type": headers.get("Content-Type"),
            }
        )
        if params and not self.select:
            raise PlatformError(f"unknown query parameters: {sorted(params)}", status_code=400)
        res = await super().request(method, path, headers=headers, content=content, **kwargs)
        if method != "GET" or not params:
            return res
        body = res.json()
        trace = [span for span in body.get("trace") or [] if span["parent_call_id"] is None or "memory" in span]
        projected = {**body, "trace": trace}
        return SimpleNamespace(raise_for_status=lambda: None, headers=res.headers, json=lambda: projected)


class Host:
    """The fake host: routes each request to its endpoint by path."""

    def __init__(self, apps: Endpoint, chat: Endpoint):
        self.apps, self.chat = apps, chat
        self.unrouted: list[str] = []

    async def request(self, method, path, **kwargs):
        if path.startswith(APP_PREFIX):
            return await self.apps.request(method, path, **kwargs)
        if path.startswith(CHAT_PREFIX):
            return await self.chat.request(method, path, **kwargs)
        self.unrouted.append(path)
        raise PlatformError(f"no route for {path}", status_code=404)


def tool_loop(provider, steps=3, step=None):
    def handler(messages):
        done = sum(1 for m in messages if m.role == "tool")
        if done >= steps:
            return "done"
        return Message(
            role="assistant", content=[ToolUseContent(id=f"c{done}", name="step", input={})], stop_reason="tool_use"
        )

    async def default_step() -> str:
        return "out " * 100

    return Agent(
        name="agent",
        model=TestModel(handler=handler),
        tools=[Tool(name="step", handler=step or default_step)],
        tracing_provider=provider,
        max_iter=steps + 5,
    )


async def run(agent, provider, config):
    ctx = RunContext(tracing_provider=provider)
    ctx.platform_config = config
    set_run_context(ctx)
    try:
        out = await agent(prompt="go").collect()
    finally:
        set_run_context(None)
    assert out.status.code == "success", out.error
    return ctx


async def read_back(provider, run_id, config, *, strict=False):
    PlatformTracingProvider._recent.clear()  # from the platform, not this process's cache
    reader = RunContext(parent_id=run_id, tracing_provider=provider)
    reader.platform_config = config
    return await (provider.get_strict(reader) if strict else provider.get(reader))


def app_provider():
    return PlatformTracingProvider.configured(platform_config=CONFIG)


class TestAnotherEndpoint:
    """The issue's "done when": a subclass that overrides only ``_run_path`` and ``_resolve_config``."""

    @pytest.mark.parametrize("config", [ORG_ONLY, NO_SUBJECT], ids=["org-only subject", "no subject"])
    async def test_stores_and_reads_its_runs_there_with_deltas_gzip_and_ordering(self, config):
        host = Host(apps=Endpoint(), chat=Endpoint())
        provider = ChatRuns.configured(config=config)
        with patch("timbal.platform.utils._request", host.request):
            ctx = await run(tool_loop(provider, steps=4), provider, config)
            stored = await read_back(provider, ctx.id, config)
        assert not host.unrouted and not host.apps.calls  # nothing went to an app's runs
        kinds = [patch_["kind"] for patch_ in host.chat.patches]
        assert kinds[0] == "full" and set(kinds[1:]) == {"delta"}  # learned from this endpoint's answers
        assert all(patch_["gzip"] for patch_ in host.chat.patches[1:])
        sent = [int(call["snapshot"]) for call in host.chat.calls if call["method"] == "PATCH"]
        assert sent == sorted(sent) and len(set(sent)) == len(sent)  # numbered in the order taken
        assert {span.call_id for span in stored.as_records()} == set(ctx._trace.keys())
        assert all("rev" not in patch_["body"] for patch_ in host.chat.patches)  # no subject, no rev

    async def test_keeps_a_long_run_alive_there(self):
        host = Host(apps=Endpoint(), chat=Endpoint())
        provider = ChatRuns.configured(keepalive_s=0.05)
        gate = asyncio.Event()

        async def slow_step() -> str:
            await gate.wait()
            return "out"

        agent = tool_loop(provider, steps=1, step=slow_step)
        with patch("timbal.platform.utils._request", host.request):
            ctx = RunContext(tracing_provider=provider)
            ctx.platform_config = ORG_ONLY
            set_run_context(ctx)
            running = asyncio.create_task(agent(prompt="go").collect())
            set_run_context(None)

            def headers_only():
                return [
                    p
                    for p in host.chat.patches
                    if p["kind"] == "delta"
                    and not p["body"]["spans"]
                    and set(p["body"]["root"])
                    == {
                        "call_id",
                        "path",
                        "parent_call_id",
                        "t0",
                    }
                ]

            async def until_kept_alive():
                while not headers_only():
                    await asyncio.sleep(0.01)

            try:
                await asyncio.wait_for(until_kept_alive(), 10)
            finally:
                gate.set()
                out = await asyncio.wait_for(running, 10)
        assert out.status.code == "success"
        assert not host.apps.calls and not host.unrouted

    async def test_app_runs_keep_their_path_and_their_errors(self):
        host = Host(apps=Endpoint(), chat=Endpoint())
        provider = app_provider()
        with patch("timbal.platform.utils._request", host.request):
            ctx = await run(tool_loop(provider), provider, CONFIG)
        assert host.apps.patches and not host.chat.calls and not host.unrouted
        assert PlatformTracingProvider._run_path(CONFIG, "r1") == f"{APP_PREFIX}r1"
        no_app = PlatformConfig(host=HOST, auth=CONFIG.auth, subject=PlatformSubject(org_id="1"))
        for config, error in ((no_app, ValueError), (NO_SUBJECT, RuntimeError)):
            reader = RunContext(parent_id=ctx.id, tracing_provider=provider)
            reader.platform_config = config
            with pytest.raises(error):
                await PlatformTracingProvider.get(reader)
            writer = RunContext(tracing_provider=provider)
            writer.platform_config = config
            with pytest.raises(error):
                await PlatformTracingProvider._store(writer)


class TestWhatAHostAdvertises:
    """Learned per family of run paths: the endpoint that answered, not every endpoint of its host."""

    async def test_an_endpoint_does_not_take_the_formats_another_one_advertised(self):
        host = Host(apps=Endpoint(), chat=Endpoint(deltas=False, gzip_ok=False, advertise=False))
        apps, chat = app_provider(), ChatRuns
        with patch("timbal.platform.utils._request", host.request):
            await run(tool_loop(apps), apps, CONFIG)  # learns deltas and gzip from the app endpoint
            await run(tool_loop(chat), chat, ORG_ONLY)
            before = len(host.apps.patches)
            later = await run(tool_loop(apps), apps, CONFIG)
        # The chat endpoint was never sent a format it did not advertise, so nothing was refused...
        attempts = [call["type"] for call in host.chat.calls if call["method"] == "PATCH"]
        assert host.chat.patches and attempts == ["application/json"] * len(host.chat.patches)
        assert not any(patch_["gzip"] for patch_ in host.chat.patches)
        # ...and the app endpoint keeps the formats it advertised.
        assert "delta" in [p["kind"] for p in host.apps.patches[before:]]
        assert later.id in host.apps.runs

    async def test_a_format_one_endpoint_refused_stays_on_for_the_other(self):
        host = Host(apps=Endpoint(), chat=Endpoint(deltas=False))
        host.chat.deltas = True  # advertises deltas at first, then refuses them (a proxy in between)
        apps, chat = app_provider(), ChatRuns
        with patch("timbal.platform.utils._request", host.request):
            first = await run(tool_loop(chat, steps=1), chat, ORG_ONLY)
            host.chat.deltas = False
            await run(tool_loop(chat), chat, ORG_ONLY)
            await run(tool_loop(apps), apps, CONFIG)
        assert first.id in host.chat.runs
        assert ChatRuns._family_caps(ORG_ONLY).delta_rejected
        assert not PlatformTracingProvider._family_caps(CONFIG).delta_rejected
        assert "delta" in [p["kind"] for p in host.apps.patches]

    async def test_a_read_asks_for_the_projection_only_where_it_was_advertised(self):
        host = Host(apps=Endpoint(select=True), chat=Endpoint())
        apps, chat = app_provider(), ChatRuns
        with patch("timbal.platform.utils._request", host.request):
            app_run = await run(tool_loop(apps), apps, CONFIG)
            chat_run = await run(tool_loop(chat), chat, ORG_ONLY)
            await read_back(apps, app_run.id, CONFIG)
            stored = await read_back(chat, chat_run.id, ORG_ONLY)  # the chat endpoint refuses `select`
        assert stored is not None and {s.call_id for s in stored.as_records()} == set(chat_run._trace.keys())
        assert [call["params"] for call in host.apps.calls if call["method"] == "GET"] == [{"select": "resume"}]
        assert [call["params"] for call in host.chat.calls if call["method"] == "GET"] == [None]

    async def test_takes_deltas_answers_for_the_providers_own_endpoint(self):
        host = Host(apps=Endpoint(), chat=Endpoint(deltas=False, advertise=False))
        apps = app_provider()
        with patch("timbal.platform.utils._request", host.request):
            await run(tool_loop(apps), apps, CONFIG)
        assert apps.takes_deltas() and PlatformTracingProvider.takes_deltas(CONFIG)
        assert not ChatRuns.takes_deltas(ORG_ONLY)
        assert not ChatRuns.takes_deltas(CONFIG)  # its own credentials win, as on its saves
        assert not ChatRuns.configured(config=ORG_ONLY).takes_deltas(ORG_ONLY)  # a configured copy: same endpoint

    async def test_takes_deltas_asks_the_providers_own_resolve_config(self):
        """A subclass that finds its credentials in ``_resolve_config`` is asked the same way its saves are."""
        host = Host(apps=Endpoint(deltas=False, advertise=False), chat=Endpoint())
        with patch("timbal.platform.utils._request", host.request):
            await run(tool_loop(ChatRuns), ChatRuns, None)
        assert ChatRuns.takes_deltas() and ChatRuns.takes_deltas(CONFIG)
        assert ChatRuns.configured(keepalive_s=1).takes_deltas()  # a configured copy: the same endpoint
        assert not PlatformTracingProvider.takes_deltas(CONFIG)  # the app's runs on that host never answered

    def test_configured_copies_share_their_endpoints_family(self):
        caps = PlatformTracingProvider._family_caps(CONFIG)
        assert PlatformTracingProvider.configured(platform_config=CONFIG)._family_caps(CONFIG) is caps
        assert PlatformTracingProvider._host_caps(HOST) is caps  # app runs: keyed by host, as before
        assert ChatRuns._family_caps(ORG_ONLY) is not caps
        assert ChatRuns.configured(keepalive_s=1)._family_caps(ORG_ONLY) is ChatRuns._family_caps(ORG_ONLY)

    @pytest.mark.parametrize("made_by", ["a factory", "configured()", "one _run_path for both"])
    def test_two_endpoints_never_share_what_one_of_them_refused(self, made_by):
        """However the classes are made: a refusal is for good, so sharing it would turn a format off for an
        endpoint that takes it. Telling them apart by class name merged classes a factory made, and every
        ``configured(_run_path=...)`` copy."""
        other = "orgs/1/threads/runs/"

        def factory(prefix):
            class Runs(PlatformTracingProvider):
                @classmethod
                def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                    return f"{prefix}{run_id}"

            return Runs

        class Mixed(PlatformTracingProvider):
            """App subjects to the app's runs, the rest to the chat endpoint."""

            @classmethod
            def _run_path(cls, platform_config, run_id):
                if platform_config.subject is not None and platform_config.subject.app_id:
                    return super()._run_path(platform_config, run_id)
                return f"{CHAT_PREFIX}{run_id}"

        if made_by == "a factory":
            pairs = [(factory(CHAT_PREFIX), ORG_ONLY), (factory(other), ORG_ONLY)]
        elif made_by == "configured()":
            pairs = [
                (PlatformTracingProvider.configured(_run_path=lambda _cfg, run_id: f"{CHAT_PREFIX}{run_id}"), ORG_ONLY),
                (PlatformTracingProvider.configured(_run_path=lambda _cfg, run_id: f"{other}{run_id}"), ORG_ONLY),
            ]
        else:
            pairs = [(Mixed, CONFIG), (Mixed, ORG_ONLY)]
        (first, first_config), (second, second_config) = pairs
        assert first._run_path(first_config, "r") != second._run_path(second_config, "r")
        first._family_caps(first_config).delta_rejected = True
        assert not second._family_caps(second_config).delta_rejected
        assert not PlatformTracingProvider._family_caps(CONFIG).delta_rejected  # nor app runs
        # And one endpoint is one family, run after run.
        assert first._family_caps(first_config) is first._family_caps(first_config)


class TestFamiliesOfRunPathsThatCheckTheRunId:
    """A ``_run_path`` may check the run id it is given (``UUID(run_id)``). Telling endpoints apart must not
    depend on its accepting an empty one: probed with an empty id, two ``configured()`` copies that keep their
    prefix in a class attribute fell back to one family, the shared function's."""

    class Prefixed(PlatformTracingProvider):
        prefix: str = CHAT_PREFIX

        @classmethod
        def _run_path(cls, platform_config, run_id):  # noqa: ARG003
            UUID(run_id)  # a run id, nothing else
            return f"{cls.prefix}{run_id}"

        @classmethod
        def _resolve_config(cls, run_context):  # noqa: ARG003
            return ORG_ONLY

    def test_two_prefixes_are_two_endpoints(self):
        chat = self.Prefixed.configured(prefix=CHAT_PREFIX)
        threads = self.Prefixed.configured(prefix="orgs/1/threads/runs/")
        chat._family_caps(ORG_ONLY).select_resume = True
        chat._family_caps(ORG_ONLY).delta_rejected = True
        assert not threads._family_caps(ORG_ONLY).select_resume
        assert not threads._family_caps(ORG_ONLY).delta_rejected

    def test_one_prefix_is_one_endpoint(self):
        first = self.Prefixed.configured(prefix=CHAT_PREFIX)
        second = self.Prefixed.configured(prefix=CHAT_PREFIX)
        assert first._family_caps(ORG_ONLY) is second._family_caps(ORG_ONLY)

    async def test_reads_ask_for_the_projection_only_where_it_was_advertised(self):
        threads_prefix = "orgs/1/threads/runs/"
        chat = self.Prefixed.configured(prefix=CHAT_PREFIX)
        threads = self.Prefixed.configured(prefix=threads_prefix)
        chat_endpoint, threads_endpoint = Endpoint(select=True), Endpoint()

        async def request(method, path, **kwargs):
            endpoint = chat_endpoint if path.startswith(CHAT_PREFIX) else threads_endpoint
            return await endpoint.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            chat_run = await run(tool_loop(chat), chat, None)
            threads_run = await run(tool_loop(threads), threads, None)
            await read_back(chat, chat_run.id, None)
            stored = await read_back(threads, threads_run.id, None)  # that endpoint refuses `select`
        assert {s.call_id for s in stored.as_records()} == set(threads_run._trace.keys())
        assert [c["params"] for c in threads_endpoint.calls if c["method"] == "GET"] == [None]

    def test_a_run_path_that_refuses_the_probe_is_told_apart_by_the_runs_it_stores(self):
        """A class that routes run ids from a registry, some to an app's runs and some to the chat endpoint:
        the probe id is in neither, so the family comes from the path each run is really stored at."""
        routes = {"run-a1": APP_PREFIX, "run-a2": APP_PREFIX, "run-c1": CHAT_PREFIX}

        class Registry(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{routes[run_id]}{run_id}"  # KeyError for any other id, the probe's too

        def caps(run_id):
            return Registry._family_caps(ORG_ONLY, Registry._run_path(ORG_ONLY, run_id), run_id)

        assert caps("run-a1") is caps("run-a2")  # one endpoint, one family
        caps("run-a1").select_resume = True
        assert not caps("run-c1").select_resume
        before = dict(PlatformTracingProvider._caps)
        assert not Registry.takes_deltas(ORG_ONLY)  # asked without a run: nothing to tell its endpoint by...
        assert PlatformTracingProvider._caps == before  # ...and nothing stored for it

    async def test_one_class_two_routes_reads_ask_for_the_projection_only_where_it_was_advertised(self):
        routes = {"run-app": APP_PREFIX, "run-chat": CHAT_PREFIX}

        class Registry(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{routes[run_id]}{run_id}"

        host = Host(apps=Endpoint(select=True), chat=Endpoint())  # the chat endpoint refuses `select`
        with patch("timbal.platform.utils._request", host.request):
            for _ in range(2):  # the first read learns that the app's runs take the projection
                await read_back(Registry, "run-app", None)
            await read_back(Registry, "run-chat", None)
        assert [c["params"] for c in host.apps.calls] == [None, {"select": "resume"}]
        assert [c["params"] for c in host.chat.calls] == [None]

    def test_a_short_run_id_is_taken_out_of_its_own_place_only(self):
        """Custom run ids can be short: ``"r"`` is in ``orgs`` and ``runs`` too. The run's own place is the last."""

        class Chat(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        assert Chat._caps_family(ORG_ONLY, Chat._run_path(ORG_ONLY, "r"), "r") == f"{CHAT_PREFIX}{{run_id}}"
        assert Chat._family_caps(ORG_ONLY, Chat._run_path(ORG_ONLY, "r"), "r") is Chat._family_caps(ORG_ONLY)

    def test_a_run_path_that_spells_the_run_id_as_a_uuid_is_one_family(self):
        class Dashed(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{UUID(run_id)}"  # 0000...-7000-8000-...: the run id, written another way

        one, other = RunContext(tracing_provider=None).id, RunContext(tracing_provider=None).id

        def caps(run_id):
            return Dashed._family_caps(ORG_ONLY, Dashed._run_path(ORG_ONLY, run_id), run_id)

        assert caps(one) is caps(other) is Dashed._family_caps(ORG_ONLY)  # and what takes_deltas() asks about
        caps(one).delta_rejected = True
        assert caps(other).delta_rejected  # a refusal is learned once for the endpoint, not once per run

    def test_a_run_path_the_run_id_cannot_be_found_in_keeps_only_what_it_refused(self):
        """Hashed, say: the endpoint cannot be told from the path. What it advertises is not kept (it could be
        another endpoint's); what it refused is, once for the class (a refusal kept too widely only sends plain
        saves). Not a family per run: each run would pay a refused save again and fill the cache."""

        class Hashed(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{sum(map(ord, run_id)) % 97}"

        def caps(run_id):
            return Hashed._family_caps(ORG_ONLY, Hashed._run_path(ORG_ONLY, run_id), run_id)

        advertised = {
            "Accept-Patch": f"application/json, {platform_module.DELTA_MEDIA_TYPE}",
            "Accept-Encoding": "gzip",
            "X-Timbal-Select": "resume",
        }
        caps("run-1").learn(advertised)
        assert not (caps("run-2").delta or caps("run-2").gzip or caps("run-2").select_resume)
        caps("run-1").gzip_rejected = True
        assert caps("run-2").gzip_rejected and caps("run-3") is caps("run-1")  # one entry for the class
        assert not [key for key in PlatformTracingProvider._caps if "{run_id}" in key]
        assert not Hashed.takes_deltas(ORG_ONLY)

    @pytest.mark.parametrize("forced", ["compress", "deltas"])
    async def test_a_forced_format_the_endpoint_refuses_is_refused_once_not_on_every_save(self, forced):
        """What an endpoint advertises is not shared when it cannot be told; what it refused is: a format
        forced on (``compress=True``, ``deltas=True``, ``TIMBAL_TRACE_GZIP``) and turned down would otherwise go
        out on every save, be refused, and be sent again."""

        class Hashed(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{sum(map(ord, run_id)) % 97}"

        provider = Hashed.configured(**{forced: True})
        endpoint = Endpoint(deltas=False, gzip_ok=False, advertise=False)

        async def request(method, path, **kwargs):
            return await endpoint.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            for _ in range(2):
                await run(tool_loop(provider, steps=3), provider, None)
        saves = [call for call in endpoint.calls if call["method"] == "PATCH"]
        assert len(saves) == len(endpoint.patches) + 1  # one refused, the rest whole and plain

    async def test_a_run_path_that_cannot_place_a_run_reads_it_as_not_stored(self):
        """``_run_path`` raises ``RunNotStored`` (a registry that does not know the run): no such run here, as a
        404 is. A restore must not fail and retry for good on it."""

        class Registry(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                if run_id != "known":
                    raise platform_module.RunNotStored(run_id)
                return f"{CHAT_PREFIX}known"

        endpoint = Endpoint()
        with patch("timbal.platform.utils._request", endpoint.request):
            assert await read_back(Registry, "unknown", None) is None
            assert await read_back(Registry, "unknown", None, strict=True) is None
        assert not endpoint.calls

    async def test_a_bug_in_run_path_is_not_read_as_a_run_not_stored(self):
        """Any other error, a ``KeyError`` included, is a bug or a misconfiguration: a strict read raises it. Read
        as "not stored", an HTTP resume would run without the tool policy of the run it resumes."""

        class Broken(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY
            settings: dict = {}

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{cls.settings['prefix']}{run_id}"  # the setting is missing

        endpoint = Endpoint()
        with patch("timbal.platform.utils._request", endpoint.request), pytest.raises(KeyError):
            await read_back(Broken, "run-1", None, strict=True)
        assert not endpoint.calls

    async def test_the_app_runs_logs_are_as_before(self):
        """A refused format is logged without a ``family`` for app runs, as before."""
        server = Endpoint(deltas=False)
        server.deltas = True  # advertises, then refuses
        apps = app_provider()
        with patch("timbal.platform.utils._request", server.request), capture_logs() as logs:
            first = await run(tool_loop(apps, steps=1), apps, CONFIG)
            server.deltas = False
            await run(tool_loop(apps), apps, CONFIG)
        refused = [entry for entry in logs if entry["event"] == "platform_trace_format_rejected"]
        assert first.id and refused and all("family" not in entry for entry in refused)


class TestTheCapabilityCache:
    """Process-wide: a route with an org, user or session id in it must not grow it without bound."""

    class PerOrg(PlatformTracingProvider):
        @classmethod
        def _run_path(cls, platform_config, run_id):
            return f"orgs/{platform_config.subject.org_id}/chat/runs/{run_id}"

    @staticmethod
    def org(number: int) -> PlatformConfig:
        return PlatformConfig(host=HOST, auth=ORG_ONLY.auth, subject=PlatformSubject(org_id=str(number)))

    def test_it_stays_bounded_and_keeps_app_runs(self):
        PlatformTracingProvider._family_caps(CONFIG).delta = True  # app runs on this host
        bound = platform_module._CAPS_FAMILIES_MAX
        for number in range(bound + 500):
            self.PerOrg._family_caps(self.org(number)).delta = True
        families = [key for key in PlatformTracingProvider._caps if " " in key]
        assert len(families) == bound
        assert self.PerOrg._family_caps(self.org(bound + 499)).delta  # the latest are kept
        assert PlatformTracingProvider._family_caps(CONFIG).delta  # app runs are keyed by host: never evicted

    def test_threads_filling_it_past_its_bound_do_not_break_it(self):
        """Saves run on event loops in more than one thread (a worker per thread, a flush at exit): inserting,
        evicting and refreshing families at once must neither raise nor lose the bound."""
        import threading

        bound = platform_module._CAPS_FAMILIES_MAX
        errors: list[BaseException] = []
        start = threading.Barrier(4)

        def fill(offset: int) -> None:
            start.wait()
            try:
                for number in range(bound * 2):
                    self.PerOrg._family_caps(self.org(offset * bound * 2 + number)).delta = True
                    self.PerOrg._family_caps(self.org(number % 50))  # refreshes, while others evict
            except BaseException as error:  # noqa: BLE001 - any of them is the failure
                errors.append(error)

        threads = [threading.Thread(target=fill, args=(offset,), daemon=True) for offset in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60)
        assert not any(thread.is_alive() for thread in threads), "a thread is stuck in the cache"
        assert not errors, errors[:1]
        assert len([key for key in PlatformTracingProvider._caps if " " in key]) <= bound

    def test_a_child_forked_while_a_thread_holds_its_locks_can_take_them(self):
        """A worker forked while another thread holds the capability cache's lock (or the snapshot counter's)
        inherits it held, with no thread left to let it go: its first save or read would block its loop."""
        import os
        import subprocess
        import sys
        import textwrap

        import timbal

        if sys.platform == "win32":
            pytest.skip("os.fork is POSIX only")
        script = textwrap.dedent(
            """
            import os, sys, threading, warnings
            from timbal.state.tracing.providers import platform

            warnings.simplefilter("ignore", DeprecationWarning)  # fork() with threads, on purpose

            held, release = threading.Event(), threading.Event()

            def hold():
                with platform._CAPS_LOCK, platform._snapshot_lock:
                    held.set()
                    release.wait()

            threading.Thread(target=hold, daemon=True).start()
            if not held.wait(10):
                sys.exit("the locks were never taken")
            pid = os.fork()
            if pid == 0:
                ok = platform._CAPS_LOCK.acquire(timeout=5) and platform._snapshot_lock.acquire(timeout=5)
                os._exit(0 if ok else 1)
            release.set()
            _, status = os.waitpid(pid, 0)
            sys.exit(os.waitstatus_to_exitcode(status))
            """
        )
        # The same timbal as this test's, whatever the interpreter has installed.
        source = os.path.dirname(os.path.dirname(timbal.__file__))
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([source, os.environ.get("PYTHONPATH", "")])}
        out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=60, env=env)
        assert out.returncode == 0, out.stderr[-2000:] or "the child could not take a lock it inherited held"

    def test_asking_about_deltas_stores_nothing(self):
        before = dict(PlatformTracingProvider._caps)
        for number in range(100):
            assert not self.PerOrg.takes_deltas(self.org(number))
        assert PlatformTracingProvider._caps == before


class TestCredentials:
    """Requests of a subclass with credentials of its own carry them; the run keeps its own config for its other
    platform calls (the model router, tools, files). Through the real request helper, on a mock transport."""

    async def test_each_endpoint_gets_its_own_credentials_and_the_run_keeps_its_config(self, monkeypatch):
        import httpx

        original = httpx.AsyncClient
        seen: list[tuple[str, str, str]] = []

        async def handle(request):
            seen.append((request.method, request.url.host + request.url.path, request.headers.get("authorization")))
            return httpx.Response(200, json={})

        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
        )
        chat = ChatRuns.configured(
            config=PlatformConfig(host="chat.test", auth=ORG_ONLY.auth, subject=ORG_ONLY.subject)
        )
        ctx = await run(tool_loop(chat, steps=1), chat, CONFIG)
        assert ctx.platform_config is CONFIG  # not replaced by the trace endpoint's
        chat_saves = [entry for entry in seen if entry[1].startswith(f"chat.test/{CHAT_PREFIX}")]
        assert chat_saves and len(chat_saves) == len(seen)
        assert {auth for _, _, auth in chat_saves} == {"Bearer t2_chat"}
        seen.clear()
        apps = app_provider()
        await run(tool_loop(apps, steps=1), apps, CONFIG)
        assert seen and {(path.split("/runs/")[0], auth) for _, path, auth in seen} == {
            (f"{HOST}/orgs/1/apps/2", "Bearer t2_test")
        }

    @pytest.mark.parametrize("subclass", [False, True], ids=["app runs", "another endpoint"])
    async def test_a_trace_only_config_never_becomes_the_runs(self, monkeypatch, subclass):
        """The run gets its config as it always does (here from the environment); the provider stores its
        traces with credentials of its own, given as ``trace_platform_config``. ``platform_config`` on a
        provider is also the run's (a run without one takes it), so it cannot be trace-only."""
        import httpx

        for name, value in {
            "TIMBAL_API_HOST": "router.test",
            "TIMBAL_API_KEY": "t2_router",
            "TIMBAL_ORG_ID": "1",
            "TIMBAL_APP_ID": "2",
        }.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
        monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
        monkeypatch.setattr("timbal.state.config_loader._file_config_cache", {})
        original = httpx.AsyncClient
        seen: list[tuple[str, str]] = []

        async def handle(request):
            seen.append((request.url.host + request.url.path, request.headers.get("authorization")))
            # The trace endpoint takes deltas (and echoes the snapshot, as a server that takes them does).
            headers = {"Accept-Patch": f"application/json, {platform_module.DELTA_MEDIA_TYPE}"}
            if "x-timbal-snapshot" in request.headers:
                headers["X-Timbal-Snapshot"] = request.headers["x-timbal-snapshot"]
            return httpx.Response(200, json={}, headers=headers)

        monkeypatch.setattr(
            httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(handle), **kwargs)
        )
        trace = PlatformConfig(
            host="trace.test",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t2_trace"),
            subject=PlatformSubject(org_id="1", app_id="2") if not subclass else PlatformSubject(org_id="1"),
        )

        class Chat(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        provider = (Chat if subclass else PlatformTracingProvider).configured(trace_platform_config=trace)
        ctx = RunContext(tracing_provider=provider)  # as a run starts: its config resolved from the environment
        assert ctx.platform_config.host == "router.test"
        set_run_context(ctx)
        try:
            out = await tool_loop(provider, steps=1)(prompt="go").collect()
        finally:
            set_run_context(None)
        assert out.status.code == "success", out.error
        assert ctx.platform_config.host == "router.test" and ctx.platform_config.auth.token.get_secret_value() == (
            "t2_router"
        )
        assert seen and {(path.split("/runs/")[0].split("/", 1)[0], auth) for path, auth in seen} == {
            ("trace.test", "Bearer t2_trace")
        }
        # Answers for the trace endpoint, which advertised deltas, not for the run's host, which never answered:
        # with the run's config passed too, as a save of that run would go (the trace config first).
        assert provider.takes_deltas() is True
        assert provider.takes_deltas(ctx.platform_config) is True
        assert PlatformTracingProvider.takes_deltas(ctx.platform_config) is False


class TestStrictReads:
    """``get`` (the next turn's memory) may read the projection; ``get_strict`` reads what restores, recoveries
    and approval claims need, which lives on tool spans too: the full trace."""

    async def test_get_strict_never_asks_for_the_projection(self):
        host = Host(apps=Endpoint(select=True), chat=Endpoint())
        apps = app_provider()
        with patch("timbal.platform.utils._request", host.request):
            ctx = await run(tool_loop(apps), apps, CONFIG)
            projected = await read_back(apps, ctx.id, CONFIG)
            whole = await read_back(apps, ctx.id, CONFIG, strict=True)
        assert [call["params"] for call in host.apps.calls if call["method"] == "GET"] == [{"select": "resume"}, None]
        assert {s.path for s in projected.as_records()} == {"agent"}
        assert {s.call_id for s in whole.as_records()} == set(ctx._trace.keys())

    async def test_a_session_restored_elsewhere_sees_the_approval_its_run_waits_for(self):
        """A run parked on a tool's approval, read back by another process: the pending approval lives on the
        tool's span, which the projection drops. The restored session must list it, and a stop must abandon it."""
        from timbal.core.chat_session import ChatSession

        host = Host(apps=Endpoint(select=True), chat=Endpoint())
        provider = app_provider()
        charged = []

        def handler(messages):
            if messages[-1].role == "tool":
                return "done"
            return Message(
                role="assistant", content=[ToolUseContent(id="pay", name="charge", input={})], stop_reason="tool_use"
            )

        agent = Agent(
            name="agent",
            model=TestModel(handler=handler),
            tools=[Tool(name="charge", handler=lambda: charged.append(1) or "charged", requires_approval=True)],
            tracing_provider=provider,
        )
        with patch("timbal.platform.utils._request", host.request):
            ctx = RunContext(tracing_provider=provider)
            ctx.platform_config = CONFIG
            set_run_context(ctx)
            out = await agent(prompt="pay").collect()
            set_run_context(None)
            assert out.status.reason == "approval_required"
            # Another process: nothing of this run in memory, and the host has advertised the projection.
            await read_back(provider, out.run_id, CONFIG)
            assert PlatformTracingProvider._family_caps(CONFIG).select_resume
            PlatformTracingProvider._recent.clear()
            session = ChatSession(agent, parent_id=out.run_id)
            try:
                await asyncio.wait_for(session.restore(), 10)
                assert session.parked
                (approval,) = session.state()["pending_approvals"]
                assert approval["path"] == "agent.charge"
                assert session.interrupt()
                finished = None
                async with asyncio.timeout(10):
                    async for item in session.events():
                        if isinstance(item, dict) and item.get("kind") == "turn_finished":
                            finished = item
                            break
            finally:
                await session.close()
        assert finished["status"]["code"] == "cancelled" and not session.parked
        assert charged == []
