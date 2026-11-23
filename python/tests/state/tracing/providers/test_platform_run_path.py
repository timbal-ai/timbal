"""PlatformTracingProvider runs stored somewhere other than ``apps/{app_id}/runs`` (issue #309).

A subclass overrides ``_run_path`` (and its own ``_resolve_config``) and gets the whole save pipeline: span
deltas, gzip, snapshot ordering, keep-alive saves, reads. What a host advertises is learned per endpoint, not
per host: two endpoints on one host need not accept the same formats, and a format one of them refused must
not be turned off for the other. The provider does not guess which paths share an endpoint: an app's runs
share the host's (as before), a subclass's runs each keep their own destination's, and a subclass that knows
its runs share one says so (``_run_path`` returns a ``RunPath`` with its ``family``). Strict reads (``get_strict``: restores, recoveries, a resume's
tool policy) read the whole trace, never the ``select=resume`` projection.
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
from timbal.state.tracing.providers.platform import PlatformTracingProvider, RunPath
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
    """Stores runs at another endpoint of the same host, with credentials of its own, and says that its runs
    share that endpoint (the ``RunPath`` it returns): the two hooks a subclass for another endpoint writes.

    Its config is not written onto the run: the run's own config serves the run's other platform calls."""

    config: PlatformConfig = ORG_ONLY

    @classmethod
    def _run_path(cls, platform_config, run_id):  # noqa: ARG003 - one org in these tests
        return RunPath(f"{CHAT_PREFIX}{run_id}", family=CHAT_PREFIX)  # every run goes to the one chat endpoint

    @classmethod
    def _resolve_config(cls, run_context):  # noqa: ARG003
        return cls.config


def chat_caps(provider=None, run_id="any-run"):
    """What the chat endpoint advertised and refused, as any run of ``ChatRuns`` (or a copy) sees it."""
    provider = provider or ChatRuns
    return provider._family_caps(ORG_ONLY, provider._run_path(ORG_ONLY, run_id))


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
        assert chat_caps().delta_rejected
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
        """A provider that finds its credentials in ``_resolve_config`` is asked the same way its saves are."""
        own = PlatformConfig(host="own.test", auth=CONFIG.auth, subject=CONFIG.subject)

        class OwnCredentials(PlatformTracingProvider):
            @classmethod
            def _resolve_config(cls, run_context):  # noqa: ARG003
                return own

        host = Host(apps=Endpoint(), chat=Endpoint())
        with patch("timbal.platform.utils._request", host.request):
            await run(tool_loop(OwnCredentials), OwnCredentials, CONFIG)
        assert OwnCredentials.takes_deltas() and OwnCredentials.takes_deltas(CONFIG)  # own.test learned deltas
        assert not PlatformTracingProvider.takes_deltas(CONFIG)  # the run's own host never answered

    def test_takes_deltas_has_no_destination_for_another_run_path_without_a_run(self):
        """Each run of another run path has its own destination (or its declared family): asked without a run,
        only a forced setting answers."""
        chat_caps().delta = True
        assert not ChatRuns.takes_deltas()
        assert ChatRuns.configured(deltas=True).takes_deltas()

    def test_configured_copies_share_their_endpoints_family(self):
        caps = PlatformTracingProvider._family_caps(CONFIG)
        assert PlatformTracingProvider.configured(platform_config=CONFIG)._family_caps(CONFIG) is caps
        assert PlatformTracingProvider._host_caps(HOST) is caps  # app runs: keyed by host, as before
        assert chat_caps() is not caps
        assert chat_caps(ChatRuns.configured(keepalive_s=1)) is chat_caps()

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

        def caps(provider, config, run_id="r"):
            return provider._family_caps(config, provider._run_path(config, run_id))

        assert first._run_path(first_config, "r") != second._run_path(second_config, "r")
        caps(first, first_config).delta_rejected = True
        assert not caps(second, second_config).delta_rejected
        assert not PlatformTracingProvider._family_caps(CONFIG).delta_rejected  # nor app runs
        # And one destination keeps what it learned, save after save.
        assert caps(first, first_config) is caps(first, first_config)


class TestFamilies:
    """Which runs share what an endpoint advertised and refused. The provider does not guess it from paths: a
    run of a subclass's ``_run_path`` keeps its own destination's (one endpoint, whatever routes there), and a
    subclass that knows its runs share an endpoint declares it (``RunPath(family=...)``). Guessing a template from the
    path took a version segment for a run id, merged factory-made classes, and missed other spellings of it."""

    class Prefixed(PlatformTracingProvider):
        prefix: str = CHAT_PREFIX

        @classmethod
        def _run_path(cls, platform_config, run_id):  # noqa: ARG003
            UUID(run_id)  # a run id, nothing else
            return f"{cls.prefix}{run_id}"

        @classmethod
        def _resolve_config(cls, run_context):  # noqa: ARG003
            return ORG_ONLY

    @staticmethod
    def caps(provider, run_id, config=ORG_ONLY):
        return provider._family_caps(config, provider._run_path(config, run_id))

    def test_each_destination_keeps_its_own(self):
        chat = self.Prefixed.configured(prefix=CHAT_PREFIX)
        threads = self.Prefixed.configured(prefix="orgs/1/threads/runs/")
        run_id = RunContext(tracing_provider=None).id
        self.caps(chat, run_id).select_resume = True
        self.caps(chat, run_id).delta_rejected = True
        assert self.caps(chat, run_id).select_resume and self.caps(chat, run_id).delta_rejected  # kept there
        assert not self.caps(threads, run_id).select_resume and not self.caps(threads, run_id).delta_rejected
        another = RunContext(tracing_provider=None).id
        assert not self.caps(chat, another).select_resume  # another run, another destination: its own

    def test_one_destination_is_one_entry_whatever_class_stores_there(self):
        run_id = RunContext(tracing_provider=None).id
        first = self.Prefixed.configured(prefix=CHAT_PREFIX)
        second = self.Prefixed.configured(prefix=CHAT_PREFIX)
        assert self.caps(first, run_id) is self.caps(second, run_id)

    def test_without_a_run_an_undeclared_subclass_has_no_endpoint_and_stores_nothing(self):
        before = dict(PlatformTracingProvider._caps)
        assert not self.Prefixed.takes_deltas(ORG_ONLY)
        assert PlatformTracingProvider._caps == before

    async def test_a_declared_family_learns_once_for_all_its_runs(self):
        """A family declared (``RunPath``): the second run's first save already goes as the endpoint takes it."""
        host = Host(apps=Endpoint(), chat=Endpoint())
        with patch("timbal.platform.utils._request", host.request):
            await run(tool_loop(ChatRuns, steps=1), ChatRuns, None)
            before = len(host.chat.patches)
            await run(tool_loop(ChatRuns, steps=1), ChatRuns, None)
        assert host.chat.patches[before]["kind"] == "delta"
        assert chat_caps().delta

    async def test_undeclared_runs_each_learn_from_their_own_answers(self):
        """Not declared: each run starts from nothing (its first save whole) and learns from its own answers."""
        undeclared = self.Prefixed.configured(prefix=CHAT_PREFIX)
        endpoint = Endpoint()

        async def request(method, path, **kwargs):
            return await endpoint.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            first = await run(tool_loop(undeclared, steps=2), undeclared, None)
            second = await run(tool_loop(undeclared, steps=2), undeclared, None)
        for ctx in (first, second):
            kinds = [
                p["kind"]
                for p, c in zip(endpoint.patches, [c for c in endpoint.calls if c["method"] == "PATCH"], strict=True)
                if c["run"] == ctx.id
            ]
            assert kinds[0] == "full" and set(kinds[1:]) == {"delta"}, kinds

    @pytest.mark.parametrize("made_by", ["configured(_run_path=...)", "a subclass", "multiple inheritance"])
    async def test_a_declared_family_is_not_inherited_by_a_class_that_moves_its_runs(self, made_by):
        """The declaration describes the ``_run_path`` it was written with. A copy or subclass that sends its
        runs elsewhere must not share it: the other endpoint refusing deltas would turn them off here."""
        threads_prefix = "orgs/1/threads/runs/"
        if made_by == "a subclass":

            class Moved(ChatRuns):
                @classmethod
                def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                    return f"{threads_prefix}{run_id}"

        elif made_by == "multiple inheritance":

            class ThreadsPath:
                @classmethod
                def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                    return f"{threads_prefix}{run_id}"

            class Moved(ThreadsPath, ChatRuns):  # its runs go to threads: ThreadsPath comes first
                pass

        else:
            Moved = ChatRuns.configured(_run_path=lambda _cfg, run_id: f"{threads_prefix}{run_id}")
        assert self.caps(Moved, "r1") is not chat_caps()
        assert chat_caps(ChatRuns.configured(keepalive_s=1)) is chat_caps()
        chat, threads = Endpoint(), Endpoint(deltas=False)  # threads refuses deltas

        async def request(method, path, **kwargs):
            return await (chat if path.startswith(CHAT_PREFIX) else threads).request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            await run(tool_loop(ChatRuns, steps=1), ChatRuns, None)  # learns: chat takes deltas
            threads.deltas = True  # advertises at first...
            await run(tool_loop(Moved, steps=1), Moved, None)
            threads.deltas = False  # ...then refuses
            await run(tool_loop(Moved, steps=2), Moved, None)
            before = len(chat.patches)
            await run(tool_loop(ChatRuns, steps=2), ChatRuns, None)
        assert {p["kind"] for p in chat.patches[before:]} == {"delta"}

    @pytest.mark.parametrize("family", ["", None, True, 1])
    def test_a_family_is_a_non_empty_string(self, family):
        """An empty one would be the app runs' entry; a non-string ``True`` would be every endpoint's."""
        with pytest.raises(TypeError):
            RunPath(f"{CHAT_PREFIX}r1", family=family)

    def test_a_family_names_a_prefix_of_its_path(self):
        """Tied to the route it declares: copies that keep their prefix in an attribute and declare one fixed
        name would otherwise share it across endpoints; a family that is not a prefix of the path is refused."""

        class Prefixed(PlatformTracingProvider):
            prefix: str = CHAT_PREFIX

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return RunPath(f"{cls.prefix}{run_id}", family="prefixed-runs")

        with pytest.raises(ValueError, match="prefix"):
            Prefixed.configured(prefix="orgs/1/threads/runs/")._run_path(ORG_ONLY, "r1")
        with pytest.raises(TypeError):
            RunPath(None, family="f")

    def test_a_run_path_copies_and_pickles_with_its_family(self):
        import copy
        import pickle

        path = RunPath(f"{CHAT_PREFIX}r1", family=CHAT_PREFIX)
        for again in (copy.copy(path), copy.deepcopy(path), pickle.loads(pickle.dumps(path))):
            assert again == path and isinstance(again, RunPath) and again.family == CHAT_PREFIX

    def test_a_declared_family_never_meets_a_destination_or_the_app_runs(self):
        """Keys apart: ``""`` the app runs', ``=`` a declared family, ``@`` a run's own destination. A family
        named like a destination is still a family."""

        class Plain(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        class Declared(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return RunPath(f"{CHAT_PREFIX}{run_id}", family=f"{CHAT_PREFIX}r")  # also run "r"'s destination

        destination = self.caps(Plain, "r")
        assert self.caps(Declared, "r1") is not destination
        assert destination is not PlatformTracingProvider._family_caps(CONFIG)
        assert self.caps(Declared, "r1") is not PlatformTracingProvider._family_caps(CONFIG)

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

    def test_an_opaque_destination_keeps_its_own(self):
        """Hashed, say: everything is kept for the concrete destination, which is one endpoint whatever routes
        there: what it advertised and what it refused."""

        class Hashed(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{sum(map(ord, run_id)) % 97}"

        def caps(run_id):
            return Hashed._family_caps(ORG_ONLY, Hashed._run_path(ORG_ONLY, run_id))

        assert Hashed._run_path(ORG_ONLY, "run-1") != Hashed._run_path(ORG_ONLY, "run-2")
        caps("run-1").learn({"Accept-Patch": f"application/json, {platform_module.DELTA_MEDIA_TYPE}"})
        caps("run-1").gzip_rejected = True
        assert caps("run-1").delta and caps("run-1").gzip_rejected  # kept for that destination
        assert not (caps("run-2").delta or caps("run-2").gzip_rejected)  # another destination learns its own
        assert not Hashed.takes_deltas(ORG_ONLY)

    @pytest.mark.parametrize("forced", ["compress", "deltas"])
    async def test_a_forced_format_a_destination_refuses_is_refused_once_there(self, forced):
        """A format forced on (``compress=True``, ``deltas=True``, ``TIMBAL_TRACE_GZIP``) and turned down: once
        per destination, not on every save. Two runs, two opaque destinations (in the order they come)."""
        destinations: dict[str, int] = {}

        class Opaque(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}slot-{destinations.setdefault(run_id, len(destinations))}"

        provider = Opaque.configured(**{forced: True})
        endpoint = Endpoint(deltas=False, gzip_ok=False, advertise=False)

        async def request(method, path, **kwargs):
            return await endpoint.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            for _ in range(2):
                await run(tool_loop(provider, steps=3), provider, None)
        saves = [call for call in endpoint.calls if call["method"] == "PATCH"]
        assert {call["run"] for call in saves} == {"slot-0", "slot-1"}
        assert len(saves) == len(endpoint.patches) + 2  # one refused per destination

    async def test_a_registry_with_versioned_routes_keeps_them_apart(self):
        """A registry that maps run ``1`` to ``v1/runs/99`` and run ``2`` to ``v2/runs/99``: two destinations,
        each with its own, so a projection one advertised is not sent to the other, which refuses it."""
        first, second = "1", "2"
        routes = {first: f"{CHAT_PREFIX}v1/runs/99", second: f"{CHAT_PREFIX}v2/runs/99"}

        class Registry(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return routes[run_id]

        assert Registry._family_caps(ORG_ONLY, routes[first]) is not Registry._family_caps(ORG_ONLY, routes[second])
        v1, v2 = Endpoint(select=True), Endpoint()  # v2 refuses `select`

        async def request(method, path, **kwargs):
            return await (v1 if "/v1/" in path else v2).request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            for _ in range(2):  # the first read learns that v1 takes the projection
                await read_back(Registry, first, None)
            await read_back(Registry, second, None)
        assert [c["params"] for c in v1.calls] == [None, {"select": "resume"}]
        assert [c["params"] for c in v2.calls] == [None]

    async def test_factory_made_classes_on_two_destinations_keep_their_refusals_apart(self):
        """Two classes from one factory (same module and qualname), opaque keys, two endpoints on one host:
        the second refusing deltas must not turn them off for the first."""

        def factory(prefix):
            class Runs(PlatformTracingProvider):
                trace_platform_config = ORG_ONLY
                deltas = True

                @classmethod
                def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                    return f"{prefix}{sum(map(ord, run_id)) % 7}"

            return Runs

        takes, refuses = factory(CHAT_PREFIX), factory("orgs/1/threads/runs/")
        chat, threads = Endpoint(), Endpoint(deltas=False)

        async def request(method, path, **kwargs):
            return await (chat if path.startswith(CHAT_PREFIX) else threads).request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            await run(tool_loop(refuses, steps=2), refuses, None)
            before = len(chat.patches)
            await run(tool_loop(takes, steps=2), takes, None)
        assert threads.patches and all(p["kind"] == "full" for p in threads.patches)
        assert {p["kind"] for p in chat.patches[before:]} == {"delta"}

    async def test_a_run_learns_from_its_own_answers_however_its_path_spells_the_id(self):
        """The run id given dashed, the path written with ``UUID(run_id).hex``: one destination for the run, so
        what its answers advertise is used on its next saves."""

        class Compact(PlatformTracingProvider):
            trace_platform_config = ORG_ONLY

            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{UUID(run_id).hex}"

        one = str(UUID(int=1, version=4))
        endpoint = Endpoint()

        async def request(method, path, **kwargs):
            return await endpoint.request(method, path, **kwargs)

        with patch("timbal.platform.utils._request", request):
            ctx = RunContext(id=one, tracing_provider=Compact)
            set_run_context(ctx)
            try:
                out = await tool_loop(Compact, steps=3)(prompt="go").collect()
            finally:
                set_run_context(None)
        assert out.status.code == "success", out.error
        kinds = [patch_["kind"] for patch_ in endpoint.patches]
        assert kinds[0] == "full" and "delta" in kinds[1:], kinds

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
            org = platform_config.subject.org_id
            return RunPath(f"orgs/{org}/chat/runs/{run_id}", family=f"orgs/{org}/chat/runs")  # declared per org

    @classmethod
    def per_org(cls, number: int):
        config = cls.org(number)
        return cls.PerOrg._family_caps(config, cls.PerOrg._run_path(config, "any-run"))

    @staticmethod
    def org(number: int) -> PlatformConfig:
        return PlatformConfig(host=HOST, auth=ORG_ONLY.auth, subject=PlatformSubject(org_id=str(number)))

    def test_it_stays_bounded_and_keeps_app_runs(self):
        PlatformTracingProvider._family_caps(CONFIG).delta = True  # app runs on this host
        bound = platform_module._CAPS_FAMILIES_MAX
        for number in range(bound + 500):
            self.per_org(number).delta = True
        assert len([key for key in PlatformTracingProvider._caps if " =" in key]) == bound
        assert self.per_org(bound + 499).delta  # the latest are kept
        assert PlatformTracingProvider._family_caps(CONFIG).delta  # app runs are keyed by host: never evicted

    def test_runs_of_undeclared_destinations_do_not_push_out_a_declared_family(self):
        """Undeclared runs each keep an entry of their own; a busy worker makes thousands. They are bounded on
        their own: a declared family (or an org's) is what saves of every run lean on."""

        class Undeclared(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        chat_caps().delta = True
        bound = platform_module._CAPS_FAMILIES_MAX
        for number in range(bound + 200):
            run_id = f"run-{number}"
            Undeclared._family_caps(ORG_ONLY, Undeclared._run_path(ORG_ONLY, run_id))
        assert chat_caps().delta
        assert len([key for key in PlatformTracingProvider._caps if " @" in key]) == bound

    def test_a_family_named_with_a_space_and_an_at_sign_is_kept_as_a_family(self):
        """An entry's kind is where it sits in the key, not a substring of the name: ``chat @v2`` is declared."""

        class Odd(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return RunPath(f"chat @v2/runs/{run_id}", family="chat @v2")

        class Undeclared(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        Odd._family_caps(ORG_ONLY, Odd._run_path(ORG_ONLY, "r")).delta = True
        for number in range(platform_module._CAPS_FAMILIES_MAX + 50):
            run_id = f"run-{number}"
            Undeclared._family_caps(ORG_ONLY, Undeclared._run_path(ORG_ONLY, run_id))
        assert Odd._family_caps(ORG_ONLY, Odd._run_path(ORG_ONLY, "r")).delta

    def test_declared_families_filling_their_bound_leave_room_for_runs(self):
        """The reverse: a worker serving more orgs than the bound still keeps what each run learns."""

        class Undeclared(PlatformTracingProvider):
            @classmethod
            def _run_path(cls, platform_config, run_id):  # noqa: ARG003
                return f"{CHAT_PREFIX}{run_id}"

        for number in range(platform_module._CAPS_FAMILIES_MAX + 50):
            self.per_org(number)
        own = Undeclared._family_caps(ORG_ONLY, Undeclared._run_path(ORG_ONLY, "run-1"))
        own.delta = True
        assert Undeclared._family_caps(ORG_ONLY, Undeclared._run_path(ORG_ONLY, "run-1")) is own

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
                    self.per_org(offset * bound * 2 + number).delta = True
                    self.per_org(number % 50)  # refreshes, while others evict
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
                return RunPath(f"{CHAT_PREFIX}{run_id}", family=CHAT_PREFIX)

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
        # App runs: answers for the trace endpoint, which advertised deltas, not for the run's host, which never
        # answered; with the run's config passed too, as a save of that run would go (the trace config first).
        # Another run path has no destination without a run (each run, or its declared family, has its own).
        assert provider.takes_deltas() is (not subclass)
        assert provider.takes_deltas(ctx.platform_config) is (not subclass)
        assert PlatformTracingProvider.takes_deltas(ctx.platform_config) is False
        if subclass:
            assert provider._family_caps(trace, provider._run_path(trace, "any-run")).delta


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
            try:
                out = await agent(prompt="pay").collect()
            finally:
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
