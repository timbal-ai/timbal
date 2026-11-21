"""Tests for Notion tools: OAuth + API key auth, HTTP behaviour, Markdown <-> blocks, tool wiring."""

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import SecretStr
from timbal.errors import CredentialNotAvailable
from timbal.platform.integrations import Integration
from timbal.tools import _oauth_store, notion
from timbal.tools.notion import (
    NotionAddComment,
    NotionAPIError,
    NotionAppendContent,
    NotionAuth,
    NotionCreatePage,
    NotionGetDataSource,
    NotionGetPage,
    NotionPartialWriteError,
    NotionQueryDataSource,
    NotionSearch,
    NotionTokenError,
    NotionUpdatePage,
    NotionWhoAmI,
    _resolve_auth,
    markdown_to_blocks,
    normalize_id,
    property_value,
    rich_text,
)

PAGE = "1a2b3c4d-5e6f-47a8-b9c0-d1e2f3a4b5c6"
DB = "2b3c4d5e-6f7a-48b9-c0d1-e2f3a4b5c6d7"
DS = "3c4d5e6f-7a8b-49c0-d1e2-f3a4b5c6d7e8"


def _tool(**kwargs: object) -> SimpleNamespace:
    defaults = {
        "integration": None,
        "auth_method": None,
        "api_key": None,
        "token": None,
        "refresh_token": None,
        "token_file": None,
        "client_id": None,
        "client_secret": None,
        "notion_version": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _integration(creds: dict[str, Any]) -> MagicMock:
    integration = MagicMock(spec=Integration)
    integration.resolve = AsyncMock(return_value=creds)
    return integration


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "NOTION_API_KEY",
        "NOTION_ACCESS_TOKEN",
        "NOTION_REFRESH_TOKEN",
        "NOTION_TOKEN_FILE",
        "NOTION_CLIENT_ID",
        "NOTION_CLIENT_SECRET",
        "NOTION_AUTH_METHOD",
        "NOTION_VERSION",
    ):
        monkeypatch.delenv(var, raising=False)
    _oauth_store.clear()

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(notion.asyncio, "sleep", no_sleep)


def _mock_http(monkeypatch: pytest.MonkeyPatch, handler: Any) -> list[httpx.Request]:
    """Route every httpx.AsyncClient the module opens through `handler`; returns the requests seen."""
    seen: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: real_client(transport=httpx.MockTransport(record), **kwargs)
    )
    return seen


def _body(request: httpx.Request) -> Any:
    return json.loads(request.content) if request.content else None


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_api_key_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "ntn_internal")
    assert await _resolve_auth(_tool()) == NotionAuth("ntn_internal", "api_key", False)


@pytest.mark.asyncio
async def test_api_key_from_integration() -> None:
    auth = await _resolve_auth(_tool(integration=_integration({"api_key": "ntn_platform"})))
    assert auth.method == "api_key"
    assert auth.token == "ntn_platform"


@pytest.mark.asyncio
async def test_oauth_token_from_integration() -> None:
    auth = await _resolve_auth(_tool(integration=_integration({"access_token": "oauth-user"})))
    assert auth == NotionAuth("oauth-user", "oauth", True)  # replaced by resolving the integration again


@pytest.mark.asyncio
async def test_oauth_before_api_key_when_unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "ntn_internal")
    assert (await _resolve_auth(_tool(token=SecretStr("oauth-user")))).method == "oauth"


@pytest.mark.asyncio
async def test_pinned_api_key_ignores_oauth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "ntn_internal")
    auth = await _resolve_auth(_tool(auth_method="api_key", token=SecretStr("oauth-user")))
    assert auth.token == "ntn_internal"


@pytest.mark.asyncio
async def test_pinned_oauth_never_falls_back_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NOTION_API_KEY", "ntn_internal")
    monkeypatch.setenv("NOTION_AUTH_METHOD", "oauth")
    with pytest.raises(CredentialNotAvailable):
        await _resolve_auth(_tool())


@pytest.mark.asyncio
async def test_invalid_auth_method() -> None:
    with pytest.raises(ValueError, match="'oauth' or 'api_key'"):
        await _resolve_auth(_tool(auth_method="password", api_key=SecretStr("k")))


@pytest.mark.asyncio
async def test_refresh_token_needs_client_and_reports_can_refresh(monkeypatch: pytest.MonkeyPatch) -> None:
    refresh = AsyncMock(return_value="fresh-access")
    monkeypatch.setattr("timbal.tools.notion.get_token_from_refresh_token", refresh)
    tool = _tool(refresh_token=SecretStr("rt"), client_id="cid", client_secret=SecretStr("secret"))
    assert await _resolve_auth(tool) == NotionAuth("fresh-access", "oauth", True)
    refresh.assert_awaited_once_with("cid", "secret", "rt", token_file=None, rejected=None)


@pytest.mark.asyncio
async def test_missing_token_file_says_to_sign_in(tmp_path: Any) -> None:
    tool = _tool(auth_method="oauth", token_file=str(tmp_path / "nope.json"), client_id="c", client_secret="s")
    with pytest.raises(CredentialNotAvailable, match="sign in first"):
        await _resolve_auth(tool)


def _no_http(monkeypatch: pytest.MonkeyPatch) -> list[httpx.Request]:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request to {request.url}")

    return _mock_http(monkeypatch, refuse)


def _deployer_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """The deployer's own Notion sign-in, as examples/notion/.env.example sets it up."""
    token_file = tmp_path / "deployer.json"
    _oauth_store.write_token_file(
        token_file, {"provider": "notion", "client_id": "cid", "refresh_token": "deployer-rt", "access_token": "dep"}
    )
    monkeypatch.setenv("NOTION_ACCESS_TOKEN", "deployer-oauth")
    monkeypatch.setenv("NOTION_REFRESH_TOKEN", "deployer-rt")
    monkeypatch.setenv("NOTION_TOKEN_FILE", str(token_file))
    monkeypatch.setenv("NOTION_CLIENT_ID", "cid")
    monkeypatch.setenv("NOTION_CLIENT_SECRET", "secret")
    monkeypatch.setenv("NOTION_API_KEY", "ntn_deployer")


@pytest.mark.asyncio
async def test_integration_api_key_beats_the_deployer_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    _deployer_env(monkeypatch, tmp_path)
    auth = await _resolve_auth(_tool(integration=_integration({"api_key": "ntn_user"})))
    assert auth == NotionAuth("ntn_user", "api_key", False)


@pytest.mark.asyncio
async def test_integration_without_credentials_ignores_the_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    seen = _no_http(monkeypatch)
    _deployer_env(monkeypatch, tmp_path)
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await _resolve_auth(_tool(integration=_integration({})))
    assert exc_info.value.env_vars == []
    assert seen == []


@pytest.mark.asyncio
async def test_integration_refresh_token_is_never_exchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    # The platform stores this refresh token; Notion rotates it, so spending it here kills that copy.
    seen = _no_http(monkeypatch)
    creds = {"refresh_token": "platform-rt", "client_id": "cid", "client_secret": "secret"}
    with pytest.raises(CredentialNotAvailable, match="must refresh it and return `token`"):
        await _resolve_auth(_tool(integration=_integration(creds)))
    assert seen == []


@pytest.mark.asyncio
async def test_rejected_integration_token_is_resolved_again(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path != "/v1/oauth/token", "an integration's token must not be refreshed here"
        if request.headers["Authorization"] == "Bearer old":
            return httpx.Response(401, json={"code": "unauthorized", "message": "API token is invalid."})
        return httpx.Response(200, json={"results": [], "has_more": False})

    seen = _mock_http(monkeypatch, handler)
    _deployer_env(monkeypatch, tmp_path)
    platform = {"refresh_token": "platform-rt", "client_id": "cid", "client_secret": "secret"}
    integration = _integration({})
    integration.resolve = AsyncMock(side_effect=[{**platform, "access_token": "old"}, {**platform, "token": "new"}])
    out = await NotionSearch(integration=integration)().collect()
    assert out.status.code == "success", out.error
    assert [r.headers["Authorization"] for r in seen] == ["Bearer old", "Bearer new"]


@pytest.mark.asyncio
async def test_rejected_integration_token_reports_the_401(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(401, json={"code": "unauthorized"}))
    _deployer_env(monkeypatch, tmp_path)
    creds = {"access_token": "old", "refresh_token": "platform-rt", "client_id": "cid", "client_secret": "secret"}
    with pytest.raises(NotionAPIError, match="revoked or expired") as exc_info:
        await notion._notion_request(_tool(integration=_integration(creds)), "GET", "/users/me")
    assert exc_info.value.status == 401
    assert [r.url.path for r in seen] == ["/v1/users/me"]


@pytest.mark.asyncio
async def test_missing_credentials() -> None:
    with pytest.raises(CredentialNotAvailable) as exc_info:
        await _resolve_auth(_tool())
    assert exc_info.value.provider_name == "Notion"
    assert "NOTION_API_KEY" in exc_info.value.env_vars


# ---------------------------------------------------------------------------
# OAuth token endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_code_exchange_uses_basic_auth_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(
        monkeypatch, lambda _: httpx.Response(200, json={"access_token": "a", "refresh_token": "r", "bot_id": "b"})
    )
    body = await notion.exchange_code("cid", "secret", "the-code", "http://localhost:8766/callback")
    assert body["refresh_token"] == "r"
    request = seen[0]
    assert str(request.url) == "https://api.notion.com/v1/oauth/token"
    assert request.headers["Authorization"] == "Basic " + __import__("base64").b64encode(b"cid:secret").decode()
    assert _body(request) == {
        "grant_type": "authorization_code",
        "code": "the-code",
        "redirect_uri": "http://localhost:8766/callback",
    }


@pytest.mark.asyncio
async def test_rotated_refresh_token_is_written_to_the_token_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    responses = iter(
        [
            {"access_token": "a1", "refresh_token": "r2"},
            {"access_token": "a2", "refresh_token": "r3"},
        ]
    )
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json=next(responses)))
    token_file = tmp_path / "notion.json"
    assert await notion.get_token_from_refresh_token("cid", "s", "r1", token_file=token_file) == "a1"
    saved = json.loads(token_file.read_text())
    assert (saved["refresh_token"], saved["client_id"], saved["provider"]) == ("r2", "cid", "notion")
    _oauth_store.clear()  # restart; the rejected access token forces the next refresh
    assert await notion.get_token_from_refresh_token("cid", "s", "r1", token_file=token_file, rejected="a1") == "a2"
    assert [_body(r)["refresh_token"] for r in seen] == ["r1", "r2"]


@pytest.mark.asyncio
async def test_invalid_grant_explains_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(400, json={"error": "invalid_grant"}))
    with pytest.raises(NotionTokenError, match="already used.*NOTION_TOKEN_FILE") as exc_info:
        await notion.get_token_from_refresh_token("cid", "s", "spent")
    assert exc_info.value.code == "invalid_grant"


@pytest.mark.asyncio
async def test_invalid_client_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(401, json={"error": "invalid_client"}))
    with pytest.raises(NotionTokenError, match="client_id or client_secret is wrong"):
        await notion.exchange_code("cid", "bad", "code", None)


# ---------------------------------------------------------------------------
# HTTP behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_headers_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"results": [], "has_more": False}))
    await NotionSearch(api_key=SecretStr("ntn_x"))(query="roadmap").collect()
    request = seen[0]
    assert request.headers["Authorization"] == "Bearer ntn_x"
    assert request.headers["Notion-Version"] == notion.NOTION_VERSION
    seen.clear()
    await NotionSearch(api_key=SecretStr("ntn_x"), notion_version="2025-09-03")(query="x").collect()
    assert seen[0].headers["Notion-Version"] == "2025-09-03"


@pytest.mark.asyncio
async def test_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "1"}, json={"code": "rate_limited"}),
            httpx.Response(529, json={"code": "service_unavailable"}),
            httpx.Response(200, json={"results": [], "has_more": False}),
        ]
    )
    seen = _mock_http(monkeypatch, lambda _: next(responses))
    out = await NotionSearch(api_key=SecretStr("k"))().collect()
    assert out.status.code == "success", out.error
    assert len(seen) == 3


@pytest.mark.asyncio
async def test_blocked_requests_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(
        monkeypatch,
        lambda _: httpx.Response(
            429,
            json={"code": "rate_limited", "additional_data": {"rate_limit_reason": "public_api_request_blocked"}},
        ),
    )
    out = await NotionSearch(api_key=SecretStr("k"))().collect()
    assert out.status.code == "error"
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_rejected_oauth_token_is_refreshed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/oauth/token":
            return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "r2"})
        if request.headers["Authorization"] == "Bearer stale":
            return httpx.Response(401, json={"code": "unauthorized", "message": "API token is invalid."})
        return httpx.Response(200, json={"results": [], "has_more": False})

    seen = _mock_http(monkeypatch, handler)
    tool = NotionSearch(token=SecretStr("stale"), refresh_token=SecretStr("r1"), client_id="c", client_secret="s")
    out = await tool().collect()
    assert out.status.code == "success", out.error
    assert [r.url.path for r in seen] == ["/v1/search", "/v1/oauth/token", "/v1/search"]
    assert seen[2].headers["Authorization"] == "Bearer fresh"


@pytest.mark.asyncio
async def test_not_shared_page_explains_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(
        monkeypatch,
        lambda _: httpx.Response(404, json={"object": "error", "code": "object_not_found", "message": "Not found."}),
    )
    out = await NotionGetPage(api_key=SecretStr("k"))(page_id=PAGE).collect()
    assert out.status.code == "error"
    assert "Connections" in out.error["message"]


@pytest.mark.asyncio
async def test_bad_api_key_explained(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(401, json={"code": "unauthorized"}))
    with pytest.raises(NotionAPIError, match="NOTION_API_KEY") as exc_info:
        await notion._notion_request(_tool(api_key=SecretStr("k")), "GET", "/users/me")
    assert exc_info.value.status == 401


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ref",
    [
        PAGE,
        PAGE.replace("-", ""),
        f"https://www.notion.so/acme/Roadmap-{PAGE.replace('-', '')}?pvs=4",
        f"https://www.notion.so/{PAGE.replace('-', '')}#heading",
        # The title's last word is hex or digits too: it must not merge into the ID.
        f"https://www.notion.so/acme/Roadmap-2025-{PAGE.replace('-', '')}",
        f"https://www.notion.so/acme/My-Cafe-{PAGE.replace('-', '')}",
        f"https://www.notion.so/acme/Sprint-12-{PAGE.replace('-', '')}?pvs=4",
        f"https://www.notion.so/acme/Deadbeef-{PAGE.replace('-', '')}#top",
    ],
)
def test_normalize_id(ref: str) -> None:
    assert normalize_id(ref) == PAGE


def test_peek_link_resolves_to_the_row_not_the_database() -> None:
    db, view, row = DB.replace("-", ""), "9" * 32, PAGE.replace("-", "")
    assert normalize_id(f"https://www.notion.so/acme/{db}?v={view}&p={row}&pm=s") == PAGE
    assert normalize_id(f"https://www.notion.so/acme/Tasks-{db}?v={view}") == DB  # a view is not the row


def test_normalize_id_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        normalize_id("not a notion id")


def test_markdown_to_blocks() -> None:
    blocks = markdown_to_blocks(
        "# Title\nSome **bold** text\nsecond line\n\n- item\n1. first\n- [x] done\n> quote\n---\n```py\nprint(1)\n```"
    )
    assert [b["type"] for b in blocks] == [
        "heading_1",
        "paragraph",
        "bulleted_list_item",
        "numbered_list_item",
        "to_do",
        "quote",
        "divider",
        "code",
    ]
    assert blocks[1]["paragraph"]["rich_text"][1] == {
        "type": "text",
        "text": {"content": "bold"},
        "annotations": {"bold": True},
    }
    assert blocks[1]["paragraph"]["rich_text"][-1]["text"]["content"] == " text\nsecond line"
    assert blocks[4]["to_do"]["checked"] is True
    assert blocks[7]["code"]["language"] == "python"


def test_rich_text_respects_notion_limits() -> None:
    long = rich_text("x" * 4500)
    assert [len(p["text"]["content"]) for p in long] == [2000, 2000, 500]
    many = rich_text(" ".join(f"**b{i}**" for i in range(150)))
    assert len(many) <= 100


def _text(parts: list[dict[str, Any]]) -> str:
    return "".join(p["text"]["content"] for p in parts)


def _within_limits(blocks: list[dict[str, Any]]) -> None:
    for block in blocks:
        parts = block[block["type"]].get("rich_text", [])
        assert len(parts) <= 100
        assert all(len(p["text"]["content"]) <= 2000 for p in parts)


def test_rich_text_keeps_all_the_text_past_100_segments() -> None:
    line = " ".join(f"**b{i}**" for i in range(150))
    parts = rich_text(line)
    assert len(parts) <= 100
    assert _text(parts) == line.replace("**", "")  # formatting gives way, the text does not
    assert parts[0]["annotations"] == {"bold": True}


def test_rich_text_too_long_for_one_value_raises() -> None:
    with pytest.raises(ValueError, match="too long for one Notion value"):
        rich_text("x" * 200_001)


def test_long_paragraph_continues_in_further_blocks() -> None:
    text = "y" * 250_000
    blocks = markdown_to_blocks(text)
    _within_limits(blocks)
    assert [b["type"] for b in blocks] == ["paragraph", "paragraph"]
    assert "".join(_text(b["paragraph"]["rich_text"]) for b in blocks) == text


def test_heavily_formatted_line_keeps_its_formatting_across_blocks() -> None:
    line = " ".join(f"**b{i}**" for i in range(150))
    blocks = markdown_to_blocks(f"- {line}")
    _within_limits(blocks)
    # 150 bold runs and the 149 spaces between them: 299 segments, three blocks.
    assert [b["type"] for b in blocks] == ["bulleted_list_item", "paragraph", "paragraph"]
    parts = [p for b in blocks for p in b[b["type"]]["rich_text"]]
    assert _text(parts) == line.replace("**", "")
    assert sum(1 for p in parts if p.get("annotations") == {"bold": True}) == 150


def test_long_code_fence_continues_in_further_code_blocks() -> None:
    code = "z" * 250_000
    blocks = markdown_to_blocks(f"```python\n{code}\n```")
    _within_limits(blocks)
    assert [(b["type"], b["code"]["language"]) for b in blocks] == [("code", "python"), ("code", "python")]
    assert "".join(_text(b["code"]["rich_text"]) for b in blocks) == code


def test_to_do_keeps_checked_only_on_its_first_block() -> None:
    blocks = markdown_to_blocks("- [x] " + "w" * 210_000)
    assert [b["type"] for b in blocks] == ["to_do", "paragraph"]
    assert blocks[0]["to_do"]["checked"] is True


def test_only_web_links_are_linked() -> None:
    parts = rich_text("[below](#notes), [spec](./spec.md) and [site](https://example.com/a)")
    links = {p["text"]["content"]: (p["text"].get("link") or {}).get("url") for p in parts}
    assert links["below"] is None
    assert links["spec"] is None
    assert links["site"] == "https://example.com/a"


def test_links_and_italic() -> None:
    parts = rich_text("see [docs](https://example.com) and *this*")
    assert parts[1]["text"]["link"] == {"url": "https://example.com"}
    assert parts[3]["annotations"] == {"italic": True}


def test_property_values() -> None:
    assert property_value({"type": "title", "title": [{"plain_text": "Launch"}]}) == "Launch"
    assert property_value({"type": "status", "status": {"name": "Done"}}) == "Done"
    assert property_value({"type": "multi_select", "multi_select": [{"name": "a"}, {"name": "b"}]}) == ["a", "b"]
    assert property_value({"type": "date", "date": {"start": "2026-01-01", "end": None}}) == "2026-01-01"
    assert property_value({"type": "formula", "formula": {"type": "number", "number": 3}}) == 3
    assert property_value({"type": "unique_id", "unique_id": {"prefix": "TASK", "number": 7}}) == "TASK-7"
    assert property_value({"type": "people", "people": [{"id": "u1", "name": "Ada"}]}) == ["Ada"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_whoami(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "id": "bot1",
                "name": "Timbal",
                "type": "bot",
                "bot": {"owner": {"type": "workspace", "workspace": True}, "workspace_name": "Acme"},
            },
        ),
    )
    out = await NotionWhoAmI(api_key=SecretStr("k"))().collect()
    assert out.output == {
        "id": "bot1",
        "name": "Timbal",
        "type": "bot",
        "workspace_name": "Acme",
        "workspace_id": None,
        "owner": "workspace",
        "auth_method": "api_key",
    }


@pytest.mark.asyncio
async def test_search(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(
        monkeypatch,
        lambda _: httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "page",
                        "id": PAGE,
                        "url": "https://www.notion.so/x",
                        "parent": {"type": "workspace", "workspace": True},
                        "properties": {"Name": {"type": "title", "title": [{"plain_text": "Roadmap"}]}},
                    },
                    {"object": "data_source", "id": DS, "title": [{"plain_text": "Tasks"}], "parent": {}},
                ],
                "has_more": True,
                "next_cursor": "c2",
            },
        ),
    )
    out = await NotionSearch(api_key=SecretStr("k"))(query=" road ", object_type="page", page_size=500).collect()
    assert _body(seen[0]) == {"page_size": 100, "query": "road", "filter": {"property": "object", "value": "page"}}
    assert [r["title"] for r in out.output["results"]] == ["Roadmap", "Tasks"]
    assert out.output["next_cursor"] == "c2"


@pytest.mark.asyncio
async def test_get_page_reads_content_as_markdown(monkeypatch: pytest.MonkeyPatch) -> None:
    def text(kind: str, content: str, **extra: Any) -> dict[str, Any]:
        return {"type": kind, kind: {"rich_text": [{"plain_text": content}], **extra}, "has_children": False}

    children = {
        PAGE: [
            text("heading_2", "Plan"),
            {**text("bulleted_list_item", "parent"), "id": "b1", "has_children": True},
            text("numbered_list_item", "one"),
            text("numbered_list_item", "two"),
            text("to_do", "ship", checked=True),
            {"type": "code", "code": {"rich_text": [{"plain_text": "x = 1"}], "language": "python"}},
            {"type": "child_page", "id": "c1", "child_page": {"title": "Notes"}},
        ],
        "b1": [text("bulleted_list_item", "child")],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/pages/{PAGE}":
            return httpx.Response(
                200,
                json={
                    "object": "page",
                    "id": PAGE,
                    "properties": {"title": {"type": "title", "title": [{"plain_text": "Q4"}]}},
                },
            )
        block_id = request.url.path.split("/")[3]
        return httpx.Response(200, json={"results": children[block_id], "has_more": False})

    _mock_http(monkeypatch, handler)
    out = await NotionGetPage(api_key=SecretStr("k"))(page_id=PAGE).collect()
    assert out.status.code == "success", out.error
    assert out.output["title"] == "Q4"
    assert out.output["content"] == (
        "## Plan\n- parent\n  - child\n1. one\n2. two\n- [x] ship\n```python\nx = 1\n```\n[page: Notes] (c1)"
    )
    assert out.output["truncated"] is False


@pytest.mark.asyncio
async def test_get_page_stops_at_max_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/v1/pages/"):
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        blocks = [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": str(i)}]}} for i in range(10)]
        return httpx.Response(200, json={"results": blocks, "has_more": True, "next_cursor": "more"})

    seen = _mock_http(monkeypatch, handler)
    out = await NotionGetPage(api_key=SecretStr("k"))(page_id=PAGE, max_blocks=5).collect()
    assert out.output["content"] == "0\n1\n2\n3\n4"
    assert out.output["truncated"] is True
    assert len(seen) == 2


def _db_handler(extra: Any = None):
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == f"/v1/databases/{DB}":
            return httpx.Response(
                200, json={"object": "database", "id": DB, "data_sources": [{"id": DS, "name": "Tasks"}]}
            )
        if path == f"/v1/data_sources/{DS}" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "object": "data_source",
                    "id": DS,
                    "title": [{"plain_text": "Tasks"}],
                    "parent": {"type": "database_id", "database_id": DB},
                    "properties": {
                        "Task": {"type": "title", "title": {}},
                        "Status": {
                            "type": "status",
                            "status": {"options": [{"name": "Done"}], "groups": [{"name": "Complete"}]},
                        },
                        "Tags": {"type": "multi_select", "multi_select": {"options": [{"name": "ops"}]}},
                    },
                },
            )
        return extra(request)

    return handler


@pytest.mark.asyncio
async def test_get_data_source_from_a_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, _db_handler())
    url = f"https://www.notion.so/acme/{DB.replace('-', '')}?v=123"
    out = await NotionGetDataSource(api_key=SecretStr("k"))(database_id=url).collect()
    assert out.status.code == "success", out.error
    assert out.output["data_source_id"] == DS
    assert out.output["database_id"] == DB
    assert out.output["properties"]["Status"] == {"type": "status", "options": ["Done"], "groups": ["Complete"]}
    assert out.output["properties"]["Tags"] == {"type": "multi_select", "options": ["ops"]}


@pytest.mark.asyncio
async def test_data_source_id_falls_back_from_databases(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/databases/{DS}":
            return httpx.Response(404, json={"code": "object_not_found"})
        return _db_handler()(request)

    _mock_http(monkeypatch, handler)
    out = await NotionGetDataSource(api_key=SecretStr("k"))(database_id=DS).collect()
    assert out.output["data_source_id"] == DS


@pytest.mark.asyncio
async def test_database_with_several_sources_asks_which(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(
        monkeypatch,
        lambda _: httpx.Response(200, json={"data_sources": [{"id": "a", "name": "A"}, {"id": "b", "name": "B"}]}),
    )
    out = await NotionGetDataSource(api_key=SecretStr("k"))(database_id=DB).collect()
    assert out.status.code == "error"
    assert "2 data sources" in out.error["message"]


@pytest.mark.asyncio
async def test_query_data_source(monkeypatch: pytest.MonkeyPatch) -> None:
    def rows(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/v1/data_sources/{DS}/query"
        assert request.method == "POST"
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "object": "page",
                        "id": PAGE,
                        "properties": {
                            "Task": {"type": "title", "title": [{"plain_text": "Ship"}]},
                            "Status": {"type": "status", "status": {"name": "Done"}},
                        },
                    }
                ],
                "has_more": False,
            },
        )

    seen = _mock_http(monkeypatch, _db_handler(rows))
    flt = {"property": "Status", "status": {"equals": "Done"}}
    out = await NotionQueryDataSource(api_key=SecretStr("k"))(database_id=DB, filter=flt, page_size=10).collect()
    assert out.status.code == "success", out.error
    assert _body(seen[-1]) == {"page_size": 10, "filter": flt}
    assert out.output["rows"][0]["properties"] == {"Task": "Ship", "Status": "Done"}
    assert out.output["next_cursor"] is None


@pytest.mark.asyncio
async def test_create_row_uses_the_title_property_and_data_source_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def create(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": "page", "id": PAGE, "url": "u", "properties": {}})

    seen = _mock_http(monkeypatch, _db_handler(create))
    out = await NotionCreatePage(api_key=SecretStr("k"))(
        parent=DB, parent_type="database", title="New task", properties={"Status": {"status": {"name": "Done"}}}
    ).collect()
    assert out.status.code == "success", out.error
    body = _body(seen[-1])
    assert body["parent"] == {"type": "data_source_id", "data_source_id": DS}
    assert body["properties"]["Task"]["title"][0]["text"]["content"] == "New task"
    assert body["properties"]["Status"] == {"status": {"name": "Done"}}


@pytest.mark.asyncio
async def test_create_page_appends_past_100_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        return httpx.Response(200, json={"results": []})

    seen = _mock_http(monkeypatch, handler)
    content = "\n".join(f"- item {i}" for i in range(250))
    out = await NotionCreatePage(api_key=SecretStr("k"))(parent=PAGE, title="Big", content=content).collect()
    assert out.output["blocks_written"] == 250
    assert [r.method for r in seen] == ["POST", "PATCH", "PATCH"]
    assert len(_body(seen[0])["children"]) == 100
    assert [len(_body(r)["children"]) for r in seen[1:]] == [100, 50]
    assert _body(seen[0])["parent"] == {"type": "page_id", "page_id": PAGE}


def _fail_patch(n: int) -> Any:
    """A handler where page creation works and the n-th PATCH (1-based) is refused."""
    patches = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal patches
        if request.method == "POST":
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        patches += 1
        if patches == n:
            return httpx.Response(400, json={"code": "validation_error", "message": "Body failed validation."})
        return httpx.Response(200, json={"results": []})

    return handler


@pytest.mark.asyncio
async def test_create_page_failing_partway_says_the_page_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, _fail_patch(2))
    content = "\n".join(f"- item {i}" for i in range(350))
    out = await NotionCreatePage(api_key=SecretStr("k"))(parent=PAGE, title="Big", content=content).collect()
    assert out.status.code == "error"
    assert [r.method for r in seen] == ["POST", "PATCH", "PATCH"]
    message = out.error["message"]
    assert f"created page {PAGE}" in message
    assert "200 of 350" in message
    assert "do not create it again" in message
    assert 'begins "item 200"' in message
    assert "may still have been applied" not in message  # a 400 was refused, not maybe applied


@pytest.mark.asyncio
async def test_append_failing_partway_says_what_landed(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, _fail_patch(2))
    blocks = markdown_to_blocks("\n".join(f"- item {i}" for i in range(250)))
    with pytest.raises(NotionPartialWriteError, match="Appended 100 of 250") as exc_info:
        await notion._append_blocks(_tool(api_key=SecretStr("k")), PAGE, blocks)
    assert (exc_info.value.page_id, exc_info.value.blocks_written, exc_info.value.blocks_total) == (PAGE, 100, 250)
    assert exc_info.value.status == 400


@pytest.mark.asyncio
async def test_append_failing_first_keeps_the_plain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, _fail_patch(1))
    with pytest.raises(NotionAPIError) as exc_info:
        await notion._append_blocks(_tool(api_key=SecretStr("k")), PAGE, markdown_to_blocks("- one\n- two"))
    assert not isinstance(exc_info.value, NotionPartialWriteError)


@pytest.mark.asyncio
async def test_update_page_title_and_trash(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"properties": {"Task": {"type": "title", "title": []}}})
        return httpx.Response(200, json={"object": "page", "id": PAGE, "in_trash": True, "properties": {}})

    seen = _mock_http(monkeypatch, handler)
    out = await NotionUpdatePage(api_key=SecretStr("k"))(page_id=PAGE, title="Renamed", in_trash=True).collect()
    assert out.status.code == "success", out.error
    body = _body(seen[-1])
    assert body["in_trash"] is True
    assert body["properties"]["Task"]["title"][0]["text"]["content"] == "Renamed"


@pytest.mark.asyncio
async def test_update_page_needs_something() -> None:
    out = await NotionUpdatePage(api_key=SecretStr("k"))(page_id=PAGE).collect()
    assert out.status.code == "error"


@pytest.mark.asyncio
async def test_append_content(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"results": []}))
    out = await NotionAppendContent(api_key=SecretStr("k"))(page_id=PAGE, content="## Update\nAll good").collect()
    assert out.output == {"id": PAGE, "blocks_written": 2}
    assert seen[0].url.path == f"/v1/blocks/{PAGE}/children"
    assert seen[0].method == "PATCH"


@pytest.mark.asyncio
async def test_add_comment_needs_exactly_one_target(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"id": "c1", "discussion_id": "d1"}))
    out = await NotionAddComment(api_key=SecretStr("k"))(text="Looks good", page_id=PAGE).collect()
    assert out.output == {"id": "c1", "discussion_id": "d1"}
    assert _body(seen[0])["parent"] == {"page_id": PAGE}
    bad = await NotionAddComment(api_key=SecretStr("k"))(text="x", page_id=PAGE, discussion_id="d1").collect()
    assert bad.status.code == "error"


def test_config_exposes_both_auth_methods() -> None:
    config = NotionSearch().get_config()
    for key in ("integration", "auth_method", "api_key", "token", "refresh_token", "token_file", "client_id"):
        assert key in config


def test_no_tool_parameter_uses_a_reserved_runnable_kwarg() -> None:
    """Runnable.__call__ pops these before validation, so a parameter with such a name never arrives."""
    import timbal.tools.notion as module

    reserved = {"parent_id", "resume", "run_in_background"}
    for name in dir(module):
        cls = getattr(module, name)
        if isinstance(cls, type) and issubclass(cls, module._NotionTool) and cls is not module._NotionTool:
            fields = set(cls().params_model.model_fields)
            assert not fields & reserved, (name, fields & reserved)


def _request_sizes(seen: list[httpx.Request]) -> list[int]:
    return [len(r.content) for r in seen if r.method in ("POST", "PATCH")]


@pytest.mark.asyncio
async def test_append_batches_stay_under_the_request_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"results": []}))
    paragraphs = [chr(ord("a") + i) * 150_000 for i in range(5)]
    out = await NotionAppendContent(api_key=SecretStr("k"))(page_id=PAGE, content="\n\n".join(paragraphs)).collect()
    assert out.status.code == "success", out.error
    assert out.output["blocks_written"] == 5
    assert len(seen) > 1 and max(_request_sizes(seen)) < 500_000
    sent = [
        "".join(p["text"]["content"] for p in b["paragraph"]["rich_text"]) for r in seen for b in _body(r)["children"]
    ]
    assert sent == paragraphs  # every block, in order


@pytest.mark.asyncio
async def test_create_with_huge_non_ascii_text_stays_under_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        return httpx.Response(200, json={"results": []})

    seen = _mock_http(monkeypatch, handler)
    text = "日本語" * 60_000  # 180,000 characters, about 1 MB as escaped JSON
    out = await NotionCreatePage(api_key=SecretStr("k"))(parent=PAGE, title="Big", content=text).collect()
    assert out.status.code == "success", out.error
    assert max(_request_sizes(seen)) < 500_000
    blocks = [b for r in seen for b in (_body(r).get("children") or [])]
    assert "".join(p["text"]["content"] for b in blocks for p in b["paragraph"]["rich_text"]) == text


def _nested_page_handler(tokens: list[str] | None = None) -> Any:
    children = {
        PAGE: [{"id": "b1", "type": "toggle", "toggle": {"rich_text": [{"plain_text": "t"}]}, "has_children": True}],
        "b1": [{"id": "b2", "type": "toggle", "toggle": {"rich_text": [{"plain_text": "u"}]}, "has_children": True}],
        "b2": [{"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "deep"}]}, "has_children": False}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if tokens is not None:
            tokens.append(request.headers["Authorization"])
            if request.headers["Authorization"] == "Bearer old":
                return httpx.Response(401, json={"code": "unauthorized"})
        if request.url.path == f"/v1/pages/{PAGE}":
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        return httpx.Response(200, json={"results": children[request.url.path.split("/")[3]], "has_more": False})

    return handler


@pytest.mark.asyncio
async def test_credentials_are_resolved_once_per_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, _nested_page_handler())
    integration = _integration({"token": "user"})
    tool = NotionGetPage(integration=integration)
    out = await tool(page_id=PAGE).collect()
    assert out.status.code == "success", out.error
    assert out.output["content"].endswith("deep")
    assert len(seen) == 4  # the page and three levels of children
    assert integration.resolve.await_count == 1
    await tool(page_id=PAGE).collect()
    assert integration.resolve.await_count == 2  # a new call resolves again


@pytest.mark.asyncio
async def test_a_token_replaced_mid_call_is_used_for_the_rest_of_it(monkeypatch: pytest.MonkeyPatch) -> None:
    tokens: list[str] = []
    _mock_http(monkeypatch, _nested_page_handler(tokens))
    integration = _integration({})
    integration.resolve = AsyncMock(side_effect=[{"token": "old"}, {"token": "new"}])
    out = await NotionGetPage(integration=integration)(page_id=PAGE).collect()
    assert out.status.code == "success", out.error
    assert tokens == ["Bearer old"] + ["Bearer new"] * 4
    assert integration.resolve.await_count == 2


@pytest.mark.asyncio
async def test_peek_link_as_a_database_keeps_the_database(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == f"/v1/databases/{DB}":
            return httpx.Response(200, json={"object": "database", "data_sources": [{"id": DS, "name": "Tasks"}]})
        if request.url.path == f"/v1/data_sources/{DS}":
            return httpx.Response(200, json={"object": "data_source", "id": DS, "properties": {}})
        return httpx.Response(404, json={"code": "object_not_found"})

    seen = _mock_http(monkeypatch, handler)
    peek = f"https://www.notion.so/acme/Tasks-{DB.replace('-', '')}?v={'9' * 32}&p={PAGE.replace('-', '')}&pm=s"
    out = await NotionGetDataSource(api_key=SecretStr("k"))(database_id=peek).collect()
    assert out.status.code == "success", out.error
    assert seen[0].url.path == f"/v1/databases/{DB}"


def test_web_url_rules() -> None:
    assert notion._is_web_url("http://x")
    assert notion._is_web_url("HTTPS://example.com/a?b=c")
    assert not notion._is_web_url("https://")
    assert not notion._is_web_url("javascript:alert(1)")
    assert not notion._is_web_url("mailto:a@example.com")
    assert not notion._is_web_url("https://example.com/" + "a" * 2000)  # over Notion's 2000-character limit


@pytest.mark.asyncio
async def test_create_with_large_properties_appends_all_content(monkeypatch: pytest.MonkeyPatch) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, json={"object": "page", "id": PAGE, "properties": {}})
        return httpx.Response(200, json={"results": []})

    seen = _mock_http(monkeypatch, handler)
    title = "t" * 199_000  # the properties alone take about half a request
    content = "p" * 199_000
    out = await NotionCreatePage(api_key=SecretStr("k"))(parent=PAGE, title=title, content=content).collect()
    assert out.status.code == "success", out.error
    assert "children" not in _body(seen[0])
    assert max(_request_sizes(seen)) < 500_000
    appended = [b for r in seen[1:] for b in _body(r)["children"]]
    assert "".join(p["text"]["content"] for b in appended for p in b["paragraph"]["rich_text"]) == content


@pytest.mark.asyncio
async def test_create_whose_first_append_fails_still_says_the_page_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    # Properties that fill most of a request leave no room for content in the create, so every
    # block is appended afterwards. If the first append fails, the page exists all the same.
    seen = _mock_http(monkeypatch, _fail_patch(1))
    out = await NotionCreatePage(api_key=SecretStr("k"))(
        parent=PAGE, title="t" * 199_000, content="p" * 199_000
    ).collect()
    assert out.status.code == "error"
    assert [r.method for r in seen] == ["POST", "PATCH"]
    assert "children" not in _body(seen[0])
    message = out.error["message"]
    assert f"created page {PAGE}" in message
    assert "wrote none of its" in message
    assert "do not create it again" in message


@pytest.mark.asyncio
async def test_partial_write_counts_with_size_based_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _mock_http(monkeypatch, _fail_patch(2))
    paragraphs = [chr(ord("a") + i) * 150_000 for i in range(7)]
    out = await NotionCreatePage(api_key=SecretStr("k"))(parent=PAGE, content="\n\n".join(paragraphs)).collect()
    assert out.status.code == "error"
    sent = [len(_body(r).get("children") or []) for r in seen]
    assert sent == [2, 2, 2]  # create, one append that landed, one that failed
    assert "wrote only 4 of 7" in out.error["message"]
    assert f'begins "{"e" * 80}"' in out.error["message"]


@pytest.mark.asyncio
async def test_whoami_resolves_credentials_once(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_http(monkeypatch, lambda _: httpx.Response(200, json={"id": "bot", "bot": {}}))
    integration = _integration({"token": "user"})
    out = await NotionWhoAmI(integration=integration)().collect()
    assert out.status.code == "success", out.error
    assert out.output["auth_method"] == "oauth"
    assert integration.resolve.await_count == 1
