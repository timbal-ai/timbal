"""The ``tools/call`` ``_meta`` the client sends, against what the platform reads (monolith#565).

``platform_read`` below is a port of the platform's reader (monolith ``src/mcp/meta.rs``, #582 and #587), written
from that file and not from the client: it refuses (``Refused``) what the platform answers with a 400 for the
whole tool call. The client must never send anything it refuses, whatever the run's ids, app, revision or agent
names are: a value that would not pass is left out, and the call goes through unlinked.
"""

import json
import re
import string
import sys

import pytest
from timbal import Agent
from timbal.core.mcp import (
    PROVENANCE_APP_KEY,
    PROVENANCE_CALL_KEY,
    PROVENANCE_PARENT_KEY,
    PROVENANCE_PATH_KEY,
    PROVENANCE_REV_KEY,
    PROVENANCE_RUN_KEY,
    PROVENANCE_SESSION_KEY,
    MCPServer,
)
from timbal.core.test_model import TestModel
from timbal.state import set_call_id, set_parent_call_id, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.state.tracing.providers import InMemoryTracingProvider
from timbal.state.tracing.span import Span
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

TOKEN = "t3_proj_sk_0123456789abcdefghijklmnopqrstuvwxyzABCD"
JWT = "eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiI0MiJ9.c2lnbmF0dXJlLWJ5dGVz"


# -- the platform's reader, ported -------------------------------------------------------------------------------


class Refused(Exception):
    """The platform's 400 for the whole call."""


# Rust's ``char::is_whitespace`` (the Unicode White_Space property), for ``str::trim``.
_RUST_WHITESPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0                　")
_ID_CHARS = frozenset(string.ascii_letters + string.digits + "-_.:")
_ALNUM = frozenset(string.ascii_letters + string.digits)


def _trim(value: str) -> str:
    start, end = 0, len(value)
    while start < end and value[start] in _RUST_WHITESPACE:
        start += 1
    while end > start and value[end - 1] in _RUST_WHITESPACE:
        end -= 1
    return value[start:end]


def _is_control(char: str) -> bool:
    """Rust's ``char::is_control``: general category Cc."""
    code = ord(char)
    return code <= 0x1F or 0x7F <= code <= 0x9F


def _words(name: str) -> list[str]:
    words = []
    for part in re.split(r"[^A-Za-z0-9]", name):
        word = ""
        for i, char in enumerate(part):
            prev = part[i - 1] if i > 0 else None
            next_lower = i + 1 < len(part) and part[i + 1] in string.ascii_lowercase
            starts_word = (
                char in string.ascii_uppercase
                and word
                and prev is not None
                and (
                    prev in string.ascii_lowercase
                    or prev in string.digits
                    or (prev in string.ascii_uppercase and next_lower)
                )
            )
            if starts_word:
                words.append(word)
                word = ""
            word += char.lower()
        if word:
            words.append(word)
    return words


def _names_a_credential(key: str) -> bool:
    words = _words(re.split(r"[/.]", key)[-1])
    if words == ["progress", "token"]:
        return False

    def singular(word: str) -> str:
        return word[:-1] if word.endswith("s") else word

    secret = {"secret", "password", "passwd", "authorization", "credential", "cookie", "apikey"}
    qualifiers = {"access", "refresh", "id", "auth", "bearer", "session", "api", "turn", "oauth", "jwt"}
    credential_tokens = bool(words) and (
        words[-1] == "token" or (len(words) >= 2 and words[-1] == "tokens" and words[-2] in qualifiers)
    )
    return (
        any(singular(word) in secret for word in words)
        or credential_tokens
        or any(words[i] == "api" and singular(words[i + 1]) == "key" for i in range(len(words) - 1))
    )


def _parse_api_key(value: str) -> bool:
    """``parse_api_key(...).is_some()``: ``t``, a version Rust parses as a ``u8``, ``_``."""
    if not value.startswith("t") or "_" not in value[1:]:
        return False
    version = value[1:].split("_", 1)[0]
    digits = version[1:] if version.startswith("+") else version
    return bool(digits) and all(c in string.digits for c in digits) and int(digits) <= 255


def _looks_like_a_credential(value: str) -> bool:
    value = _trim(value)
    lower = "".join(c.lower() if c.isascii() else c for c in value)
    if lower.startswith("bearer ") or lower.startswith("basic "):
        return True
    tail = value.rsplit("_", 1)[-1]
    if len(tail) >= 32 and set(tail) <= _ALNUM and _parse_api_key(value):
        return True
    segments = value.split(".")
    return (
        len(segments) == 3
        and value.startswith("eyJ")
        and all(segment and set(segment) <= _ALNUM | {"-", "_", "="} for segment in segments)
    )


def _reject_credentials(path: str, value, *, key: bool = True) -> None:
    if key and _names_a_credential(path):
        raise Refused(f"credential name {path}")
    if isinstance(value, str) and _looks_like_a_credential(value):
        raise Refused(f"credential value under {path}")
    if isinstance(value, list):
        for i, item in enumerate(value):
            _reject_credentials(f"{path}[{i}]", item, key=False)
    if isinstance(value, dict):
        for inner, item in value.items():
            _reject_credentials(f"{path}.{inner}", item)


def _id(meta: dict, key: str):
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value and len(value.encode()) <= 128 and set(value) <= _ID_CHARS:
        return value
    raise Refused(key)


def _text(meta: dict, key: str, limit: int):
    value = meta.get(key)
    if value is None:
        return None
    if isinstance(value, str) and _trim(value) and len(value) <= limit and not any(map(_is_control, value)):
        return value
    raise Refused(key)


def _app_id(meta: dict):
    value = meta.get(PROVENANCE_APP_KEY)
    if value is None:
        return None
    parsed = None
    if isinstance(value, int) and not isinstance(value, bool) and -(2**63) <= value < 2**63:
        parsed = value  # serde_json's as_i64: a JSON integer in range (a float or a bool is not one)
    elif isinstance(value, str) and 0 < len(value) <= 18 and all(c in string.digits for c in value):
        parsed = int(value)
    if parsed is not None and parsed > 0:
        return parsed
    raise Refused(PROVENANCE_APP_KEY)


def platform_read(meta: dict) -> dict:
    """What the platform makes of a ``_meta`` sent over the wire, or ``Refused``."""
    try:
        # As it arrives: UTF-8 JSON (serde_json refuses a lone surrogate, which Python writes as an escape).
        meta = json.loads(json.dumps(meta, ensure_ascii=False).encode("utf-8"))
    except UnicodeEncodeError as error:
        raise Refused("not UTF-8") from error
    for key, value in meta.items():
        _reject_credentials(key, value)
    return {
        "run_id": _id(meta, PROVENANCE_RUN_KEY),
        "call_id": _id(meta, PROVENANCE_CALL_KEY),
        "parent_run_id": _id(meta, PROVENANCE_PARENT_KEY),
        "session_id": _id(meta, PROVENANCE_SESSION_KEY),
        "app_id": _app_id(meta),
        "rev": _text(meta, PROVENANCE_REV_KEY, 255),
        "path": _text(meta, PROVENANCE_PATH_KEY, 512),
    }


class TestTheOracle:
    """The port says what the platform's own tests say (``meta.rs``'s ``mod tests``)."""

    def test_it_reads_what_the_platform_reads(self):
        read = platform_read(
            {
                PROVENANCE_RUN_KEY: "06af0701755d7c1e8000220bfdb2062e",
                PROVENANCE_CALL_KEY: "06af0701-77aa-7c1e-8000-1b2c3d4e5f60",
                PROVENANCE_APP_KEY: "1829",
                PROVENANCE_REV_KEY: "t2_refactor_trace_saving",
                PROVENANCE_PATH_KEY: "Support Bot.memory_read",
                "progressToken": "eyJ-not-a-jwt",
                "acme.com/usage": {"input_tokens": 1, "MAX_TOKENS": 9, "HTTPTokensUsed": 1, "tokens": 9},
                "acme.com/secretary": "Ada",
            }
        )
        assert read["app_id"] == 1829 and read["path"] == "Support Bot.memory_read"

    @pytest.mark.parametrize(
        "meta",
        [
            {"timbal.ai/turn_token": "harmless-looking"},
            {"authorization": "x"},
            {"acme.com/clientSecret": "x"},
            {"acme.com/config": {"AWSSecretAccessKey": "x"}},
            {"acme.com/list": [{"ok": "x"}, {"clientSecret": "plain"}]},
            {"acme.com/accessTokens": "x"},
            {"acme.com/x-api-key": "x"},
            {"acme.com/ctx": JWT},
            {PROVENANCE_RUN_KEY: TOKEN},
            {"acme.com/ctx": "t2_0123456789abcdefghijklmnopqrstuv"},
            {"acme.com/ctx": {"inner": "Bearer abc.def"}},
            {PROVENANCE_APP_KEY: 0},
            {PROVENANCE_APP_KEY: 1.5},
            {PROVENANCE_APP_KEY: "1" * 19},
            {PROVENANCE_REV_KEY: "tab\there"},
            {PROVENANCE_PATH_KEY: "  "},
            {PROVENANCE_CALL_KEY: "has space"},
            {PROVENANCE_CALL_KEY: "x" * 129},
            {PROVENANCE_REV_KEY: "main\udcff"},
        ],
    )
    def test_it_refuses_what_the_platform_refuses(self, meta):
        with pytest.raises(Refused):
            platform_read(meta)


# -- the client ----------------------------------------------------------------------------------------------------


def fresh():
    set_run_context(None)
    set_call_id(None)
    set_parent_call_id(None)


def platform_context(**subject) -> RunContext:
    """A run on an app of the platform. Explicit config and provider: nothing reads ``~/.timbal`` or calls out."""
    context = RunContext(
        parent_id="06af0701755d7c1e8000220bfdb20000",
        tracing_provider=InMemoryTracingProvider,
        platform_config=PlatformConfig(
            host="api.example.test",
            auth=PlatformAuth(type=PlatformAuthType.BEARER, token=TOKEN),
            subject=PlatformSubject(**{"org_id": "1", "app_id": "1829", "rev": "main", **subject}),
        ),
    )
    context._session_id = "06af0701755d7c1e8000220bfdb2aaaa"
    return context


def meta_of(context: RunContext, path: str = "support.billing.memory_read", server=None) -> dict:
    """The ``_meta`` a call of the tool at ``path`` sends, run under ``context``."""
    call_id = "06af0701-77aa-7c1e-8000-1b2c3d4e5f60"
    context._trace[call_id] = Span(path=path, call_id=call_id, parent_call_id=None, t0=0)
    set_run_context(context)
    set_call_id(call_id)
    try:
        server = server or MCPServer(transport="http", service="api")
        return server._provenance_meta() or {}
    finally:
        fresh()


class TestPlatformClaims:
    def test_every_claim_is_read_and_none_is_a_credential(self):
        context = platform_context(rev="feature/ñ")
        meta = meta_of(context, path="Asistente Médico.billing.memory_read")
        assert platform_read(meta) == {
            "run_id": context.id,
            "call_id": "06af0701-77aa-7c1e-8000-1b2c3d4e5f60",
            "parent_run_id": "06af0701755d7c1e8000220bfdb20000",
            "session_id": "06af0701755d7c1e8000220bfdb2aaaa",
            "app_id": 1829,
            "rev": "feature/ñ",
            "path": "Asistente Médico.billing.memory_read",
        }
        assert meta[PROVENANCE_APP_KEY] == 1829  # a JSON number: the platform looks the run up on (app_id, run_id)
        assert TOKEN not in json.dumps(meta)
        assert "timbal.ai/tool_call_id" not in meta  # the platform reads call_id, not this

    def test_the_deployed_revision_is_read_where_the_runtime_sets_it(self, monkeypatch):
        """The platform's runtime sets ``TIMBAL_PROJECT_REV`` (the config loader reads ``TIMBAL_REV``): with no
        revision on the subject, the claim takes it from there, so the platform can attribute the call."""
        monkeypatch.setenv("TIMBAL_PROJECT_REV", "rev-42")
        assert platform_read(meta_of(platform_context(rev=None)))["rev"] == "rev-42"
        assert platform_read(meta_of(platform_context(rev="main")))["rev"] == "main"  # the subject's comes first
        monkeypatch.delenv("TIMBAL_PROJECT_REV")
        assert platform_read(meta_of(platform_context(rev=None))).get("rev") is None

    def test_the_per_run_connection_to_the_platform_sends_them(self):
        """``service="api"`` calls through a connection it opens per run: that one sends the claims."""
        server = MCPServer(transport="http", url="https://api.example.test/mcp", provenance=True)
        assert PROVENANCE_APP_KEY not in meta_of(platform_context(), server=server)  # a third party: never
        server._first_party = True
        assert meta_of(platform_context(), server=server)[PROVENANCE_APP_KEY] == 1829

    @pytest.mark.parametrize(
        ("change", "left_out"),
        [
            ({"id": "run with spaces"}, PROVENANCE_RUN_KEY),
            ({"id": "r" * 129}, PROVENANCE_RUN_KEY),
            ({"id": "run/1"}, PROVENANCE_RUN_KEY),
            ({"id": TOKEN}, PROVENANCE_RUN_KEY),
            ({"parent_id": JWT}, PROVENANCE_PARENT_KEY),
            ({"parent_id": "ñandú"}, PROVENANCE_PARENT_KEY),
            ({"_session_id": "t2_0123456789abcdefghijklmnopqrstuv"}, PROVENANCE_SESSION_KEY),
            ({"_session_id": "conv 42"}, PROVENANCE_SESSION_KEY),
            ({"_session_id": ""}, PROVENANCE_SESSION_KEY),
            ({"app_id": "0"}, PROVENANCE_APP_KEY),
            ({"app_id": "-4"}, PROVENANCE_APP_KEY),
            ({"app_id": "12a"}, PROVENANCE_APP_KEY),
            ({"app_id": "²"}, PROVENANCE_APP_KEY),  # a digit to str.isdigit(), and int() refuses it
            ({"app_id": "١٢"}, PROVENANCE_APP_KEY),  # Arabic-Indic digits: int() takes them, the platform does not
            ({"app_id": str(2**63)}, PROVENANCE_APP_KEY),
            ({"app_id": None}, PROVENANCE_APP_KEY),
            ({"app_id": "1" * 5000}, PROVENANCE_APP_KEY),  # past what int() parses from text: refused, not raised
            ({"rev": "tab\there"}, PROVENANCE_REV_KEY),
            ({"rev": "nl\n"}, PROVENANCE_REV_KEY),
            ({"rev": "x" * 256}, PROVENANCE_REV_KEY),
            ({"rev": "   "}, PROVENANCE_REV_KEY),
            ({"rev": "　"}, PROVENANCE_REV_KEY),  # blank to Rust's trim
            ({"rev": JWT}, PROVENANCE_REV_KEY),
            ({"rev": "Bearer main"}, PROVENANCE_REV_KEY),
            ({"path": "bot\x00.tool"}, PROVENANCE_PATH_KEY),
            ({"path": "bot\x85.tool"}, PROVENANCE_PATH_KEY),
            ({"path": "a" * 513}, PROVENANCE_PATH_KEY),
            ({"path": "Bearer bot.tool"}, PROVENANCE_PATH_KEY),  # an agent named "Bearer bot"
            ({"path": "eyJhbG.billing.tool"}, PROVENANCE_PATH_KEY),  # a JWT's shape
            ({"path": "basic support.tool"}, PROVENANCE_PATH_KEY),
            ({"rev": "main\udcff"}, PROVENANCE_REV_KEY),  # a lone surrogate (an env var read with surrogateescape)
            ({"path": "bot\ud800.tool"}, PROVENANCE_PATH_KEY),
        ],
    )
    def test_a_value_the_platform_would_refuse_is_left_out(self, change, left_out):
        """Not sent: the call goes through, unlinked, instead of failing on a 400. The other claims stay."""
        subject = {key: value for key, value in change.items() if key in ("app_id", "rev")}
        context = platform_context(**subject)
        if "id" in change:
            context.id = change["id"]
        if "parent_id" in change:
            context.parent_id = change["parent_id"]
        if "_session_id" in change:
            context._session_id = change["_session_id"]
        meta = meta_of(context, path=change.get("path", "support.billing.memory_read"))
        assert left_out not in meta
        platform_read(meta)  # nothing refused
        everything = {
            PROVENANCE_RUN_KEY,
            PROVENANCE_CALL_KEY,
            PROVENANCE_PARENT_KEY,
            PROVENANCE_SESSION_KEY,
            PROVENANCE_APP_KEY,
            PROVENANCE_REV_KEY,
            PROVENANCE_PATH_KEY,
        }
        assert set(meta) == everything - {left_out}

    @pytest.mark.parametrize(
        ("value", "kept"),
        [
            ("12", 12),
            ("000123", 123),
            (str(2**63 - 1), 2**63 - 1),  # 19 digits: sent as a number, which the platform reads as an i64
            ("0" * 5000 + "7", 7),
        ],
        ids=["digits", "leading zeros", "i64 max", "5000 leading zeros"],
    )
    def test_an_app_id_is_sent_as_a_number(self, value, kept):
        meta = meta_of(platform_context(app_id=value))
        assert meta[PROVENANCE_APP_KEY] == kept and platform_read(meta)["app_id"] == kept

    def test_free_text_the_platform_takes_is_sent(self):
        """Spaces and Unicode: agent names are free text. Branch names shaped like key prefixes are not keys."""
        for rev, path in (
            ("t2_refactor_trace_saving", "t3_sc.agent.tool"),
            ("feature/ñ", "Support Bot.memory_read"),
            ("x" * 255, "a" * 512),
        ):
            meta = meta_of(platform_context(rev=rev), path=path)
            assert (meta[PROVENANCE_REV_KEY], meta[PROVENANCE_PATH_KEY]) == (rev, path)
            platform_read(meta)

    def test_nothing_at_all(self):
        """A run whose every id would be refused, with no app: no ``_meta`` rather than an empty one."""
        context = RunContext(id="not an id", tracing_provider=InMemoryTracingProvider)
        set_run_context(context)
        try:
            assert MCPServer(transport="http", url="https://x.test/mcp", provenance=True)._provenance_meta() is None
        finally:
            fresh()


META_SERVER_SCRIPT = """
import json
from mcp.server.fastmcp import Context, FastMCP

mcp = FastMCP("meta-server")


@mcp.tool()
def echo_meta(ctx: Context) -> str:
    \"\"\"Return the request's _meta.\"\"\"
    meta = ctx.request_context.meta
    return json.dumps(meta.model_dump(exclude_none=True) if meta is not None else {})


mcp.run()
"""


class TestOverTheWire:
    async def test_the_whole_meta_a_call_sends_passes_the_platforms_reader(self, tmp_path):
        """A real ``tools/call`` from an agent: what reaches the server is read, and nothing in it is refused."""
        fresh()
        script = tmp_path / "meta_server.py"
        script.write_text(META_SERVER_SCRIPT)
        server = MCPServer(transport="stdio", command=sys.executable, args=[str(script)], provenance=True)
        server._first_party = True  # as the per-run connection service="api" opens to the platform
        context = platform_context(rev="feature/memory-v1")
        set_run_context(context)
        call = Message(
            role="assistant", content=[ToolUseContent(id="m1", name="echo_meta", input={})], stop_reason="tool_use"
        )
        agent = Agent(
            name="Asistente Médico",
            model=TestModel(responses=[call, "done"]),
            tools=[server],
            tracing_provider=InMemoryTracingProvider,
        )
        try:
            events = [event async for event in agent(prompt="go")]
        finally:
            await server.close()
            fresh()
        tool_out = next(e for e in events if getattr(e, "type", "") == "OUTPUT" and e.path.endswith(".echo_meta"))
        sent = json.loads(tool_out.output)
        assert platform_read(sent) == {
            "run_id": tool_out.run_id,
            "call_id": tool_out.call_id,
            "parent_run_id": "06af0701755d7c1e8000220bfdb20000",
            "session_id": "06af0701755d7c1e8000220bfdb2aaaa",
            "app_id": 1829,
            "rev": "feature/memory-v1",
            "path": "Asistente Médico.echo_meta",
        }
        assert TOKEN not in tool_out.output
