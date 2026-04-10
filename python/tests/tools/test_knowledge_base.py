"""Tests for KnowledgeBaseQuery tool."""

import os
from unittest.mock import AsyncMock

import pytest
from timbal import Agent
from timbal.core.test_model import TestModel
from timbal.core.tool import Tool
from timbal.state import get_run_context
from timbal.state.config_loader import resolve_platform_config
from timbal.tools.knowledge_base import KnowledgeBaseQuery
from timbal.types.content import TextContent, ToolUseContent
from timbal.types.message import Message


def _skip_if_kb_integration_env_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timbal.state.config_loader._cached_default_config", None)
    monkeypatch.setattr("timbal.state.config_loader._default_config_resolved", False)
    org = os.environ.get("TIMBAL_ORG_ID")
    kb = os.environ.get("TIMBAL_KB_ID")
    token = os.environ.get("TIMBAL_API_KEY") or os.environ.get("TIMBAL_API_TOKEN")
    if not (org and kb and token):
        pytest.skip(
            "KB integration: set TIMBAL_ORG_ID, TIMBAL_KB_ID, and TIMBAL_API_KEY (or TIMBAL_API_TOKEN). "
            "Optional: TIMBAL_KB_INTEGRATION_SQL (default SELECT 1), TIMBAL_API_HOST if not in ~/.timbal."
        )
    cfg = resolve_platform_config()
    if cfg is None or not getattr(cfg, "host", None):
        pytest.skip(
            "KB integration: platform host not resolved — set TIMBAL_API_HOST or configure ~/.timbal (base_url + api_key)."
        )


@pytest.mark.asyncio
async def test_kb_query_merges_constructor_and_call_args(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def capture_query(
        sql: str,
        params: list | None = None,
        *,
        org_id: str | None = None,
        kb_id: str | None = None,
        legacy: bool = False,
        explain: bool | None = None,
    ):
        calls.append(
            {
                "sql": sql,
                "params": params,
                "org_id": org_id,
                "kb_id": kb_id,
                "legacy": legacy,
                "explain": explain,
            }
        )
        return {"rows": []}

    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", capture_query)

    tool = KnowledgeBaseQuery(kb_id="kb_ctor", org_id="org_ctor", legacy=False)
    assert tool.name == "knowledge_base_kb_ctor_query"
    out = await tool(sql="SELECT 1", params=[1], explain=True).collect()
    assert out.output == {"rows": []}
    assert calls == [
        {
            "sql": "SELECT 1",
            "params": [1],
            "org_id": "org_ctor",
            "kb_id": "kb_ctor",
            "legacy": False,
            "explain": True,
        }
    ]

    calls.clear()
    out2 = await tool(sql="SELECT 2", explain=False).collect()
    assert out2.output == {"rows": []}
    assert calls == [
        {
            "sql": "SELECT 2",
            "params": None,
            "org_id": "org_ctor",
            "kb_id": "kb_ctor",
            "legacy": False,
            "explain": False,
        }
    ]


@pytest.mark.asyncio
async def test_kb_query_legacy_ignores_explain(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    async def capture_query(
        sql: str,
        _params: list | None = None,
        *,
        org_id: str | None = None,
        kb_id: str | None = None,
        legacy: bool = False,
        explain: bool | None = None,
    ):
        calls.append(
            {"sql": sql, "legacy": legacy, "explain": explain, "org_id": org_id, "kb_id": kb_id}
        )
        return {"rows": []}

    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", capture_query)

    tool = KnowledgeBaseQuery(kb_id="k", org_id="o", legacy=True)
    await tool(sql="SELECT 1", explain=True).collect()
    assert calls == [{"sql": "SELECT 1", "legacy": True, "explain": None, "org_id": "o", "kb_id": "k"}]


@pytest.mark.asyncio
async def test_kb_query_custom_tool_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", AsyncMock(return_value={"rows": [1]}))
    tool = KnowledgeBaseQuery(name="docs_kb", kb_id="kb_docs")
    assert tool.name == "docs_kb"
    out = await tool(sql="SELECT 1").collect()
    assert out.output == {"rows": [1]}


def test_kb_query_auto_name_and_description_multi_kb() -> None:
    docs = KnowledgeBaseQuery(kb_id="product-docs", about="Product specs and API reference")
    hr = KnowledgeBaseQuery(kb_id="hr_policies")
    assert docs.name == "knowledge_base_product_docs_query"
    assert hr.name == "knowledge_base_hr_policies_query"
    assert docs.name != hr.name
    assert "product-docs" not in (docs.description or "").lower()
    assert "Product specs" in (docs.description or "")
    assert "hr_policies" not in (hr.description or "").lower()

    Agent(
        name="multi_kb_agent",
        model=TestModel(responses=["ok"]),
        tools=[docs, hr],
    )


def test_kb_query_duplicate_kb_id_needs_explicit_names() -> None:
    with pytest.raises(ValueError, match="already exists"):
        Agent(
            name="bad",
            model=TestModel(responses=["ok"]),
            tools=[
                KnowledgeBaseQuery(kb_id="same"),
                KnowledgeBaseQuery(kb_id="same"),
            ],
        )

    # Disambiguate with explicit tool names (e.g. different default params / views).
    Agent(
        name="ok",
        model=TestModel(responses=["ok"]),
        tools=[
            KnowledgeBaseQuery(name="kb_same_primary", kb_id="same"),
            KnowledgeBaseQuery(name="kb_same_archive", kb_id="same"),
        ],
    )


def test_kb_query_default_name_without_kb_id() -> None:
    t = KnowledgeBaseQuery()
    assert t.name == "knowledge_base_query"


def test_kb_query_explicit_description_not_auto_generated() -> None:
    t = KnowledgeBaseQuery(kb_id="mykb", description="Only this text — no kb_id echoed.")
    assert t.description == "Only this text — no kb_id echoed."


def test_kb_query_auto_description_hides_org_and_kb_ids() -> None:
    t = KnowledgeBaseQuery(kb_id="k2-secret-id", org_id="org_42")
    d = t.description or ""
    assert "org_42" not in d
    assert "k2-secret-id" not in d
    assert "legacy" not in d.lower()


def test_kb_query_tool_schema_exposes_only_sql_params_explain() -> None:
    t = KnowledgeBaseQuery(kb_id="x", org_id="y")
    fields = set(t.params_model.model_fields.keys())
    assert fields == {"sql", "params", "explain"}


def test_kb_query_whitespace_about_omitted_from_description() -> None:
    t = KnowledgeBaseQuery(kb_id="k", about="   \t  ")
    assert "When to use" not in (t.description or "")


def test_kb_query_numeric_leading_kb_id_slug() -> None:
    t = KnowledgeBaseQuery(kb_id="42docs")
    assert t.name == "knowledge_base_kb_42docs_query"


def test_kb_query_punctuation_only_kb_id_uses_kb_slug() -> None:
    t = KnowledgeBaseQuery(kb_id="---")
    assert t.name == "knowledge_base_kb_query"


def test_kb_query_empty_string_kb_id_treated_as_unset() -> None:
    """``kb_id=""`` is falsy: same as omitting (env / platform default)."""
    t = KnowledgeBaseQuery(kb_id="")
    assert t.name == "knowledge_base_query"


def test_kb_query_long_kb_id_truncates_tool_name() -> None:
    long_id = "a" * 80
    t = KnowledgeBaseQuery(kb_id=long_id)
    assert len(t.name) <= 64
    assert t.name.startswith("knowledge_base_")
    assert t.name.endswith("_query")


def test_kb_query_duplicate_env_default_tools_collide() -> None:
    with pytest.raises(ValueError, match="already exists"):
        Agent(
            name="two_defaults",
            model=TestModel(responses=["ok"]),
            tools=[KnowledgeBaseQuery(), KnowledgeBaseQuery()],
        )


@pytest.mark.asyncio
async def test_kb_query_success_records_default_tool_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", AsyncMock(return_value={"rows": []}))
    tool = KnowledgeBaseQuery(kb_id="acme")
    assert tool.name == "knowledge_base_acme_query"
    out = await tool(sql="SELECT 1").collect()
    assert out.usage.get("knowledge_base_acme_query:requests") == 1


@pytest.mark.asyncio
async def test_kb_query_explain_omitted_passes_none_to_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    explains: list = []

    async def cap(_sql: str, _params: list | None = None, **kwargs: object):
        explains.append(kwargs.get("explain"))
        return {"rows": []}

    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", cap)
    tool = KnowledgeBaseQuery(kb_id="k", legacy=False)
    await tool(sql="SELECT 1").collect()
    assert explains == [None]


def test_kb_query_get_config_includes_kb_fields() -> None:
    t = KnowledgeBaseQuery(kb_id="kid", org_id="oid", legacy=True, about="hint")
    cfg = t.get_config()
    assert cfg["kb_id"]["value"] == "kid"
    assert cfg["org_id"]["value"] == "oid"
    assert cfg["legacy"]["value"] is True
    assert cfg["about"]["value"] == "hint"


@pytest.mark.asyncio
async def test_kb_query_agent_test_model_tool_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent + TestModel invokes ``KnowledgeBaseQuery`` end-to-end (platform call mocked)."""

    async def fake_kb(sql: str, _params: list | None = None, **_kwargs: object):
        return {"rows": [{"echo_sql": sql}], "ok": True}

    monkeypatch.setattr("timbal.tools.knowledge_base.kb_query", fake_kb)

    kb_tool = KnowledgeBaseQuery(kb_id="fixture_kb")
    tool_name = kb_tool.name
    assert tool_name == "knowledge_base_fixture_kb_query"

    model = TestModel(
        responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="t_kb", name=tool_name, input={"sql": "SELECT 1"})],
                stop_reason="tool_use",
            ),
            Message(role="assistant", content=[TextContent(text="done")], stop_reason="end_turn"),
        ]
    )
    agent_name = "kb_agent_fixture"
    agent = Agent(name=agent_name, model=model, tools=[kb_tool])
    prompt = Message.validate({"role": "user", "content": "Query the knowledge base with SELECT 1."})
    output = await agent(prompt=prompt).collect()

    assert output.status.code == "success"
    assert output.usage.get(f"{tool_name}:requests") == 1

    ctx = get_run_context()
    assert ctx is not None
    root = ctx.root_span()
    assert root is not None
    assert root.path == agent_name
    tool_spans = ctx._trace.get_path(f"{agent_name}.{tool_name}")
    assert len(tool_spans) == 1
    assert isinstance(tool_spans[0].runnable, Tool)
    assert tool_spans[0].runnable.name == tool_name
    assert tool_spans[0].usage.get(f"{tool_name}:requests") == 1
    assert tool_spans[0].parent_call_id == root.call_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_kb_query_agent_live_k2_real_kb_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Live K2 query: requires env credentials and network. Run with ``pytest -m integration``."""
    _skip_if_kb_integration_env_not_set(monkeypatch)

    real_kb_id = os.environ["TIMBAL_KB_ID"]
    kb_tool = KnowledgeBaseQuery(kb_id=real_kb_id)
    tool_name = kb_tool.name
    sql = os.environ.get("TIMBAL_KB_INTEGRATION_SQL", "SELECT 1")

    model = TestModel(
        responses=[
            Message(
                role="assistant",
                content=[ToolUseContent(id="t_live", name=tool_name, input={"sql": sql})],
                stop_reason="tool_use",
            ),
            Message(role="assistant", content=[TextContent(text="done")], stop_reason="end_turn"),
        ]
    )
    agent_name = "kb_agent_live_integration"
    agent = Agent(name=agent_name, model=model, tools=[kb_tool])
    prompt = Message.validate({"role": "user", "content": "Run the configured probe SQL on the knowledge base."})
    output = await agent(prompt=prompt).collect()

    assert output.status.code == "success"
    assert output.usage.get(f"{tool_name}:requests") == 1
    assert output.output is not None
