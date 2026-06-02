import os
from pathlib import Path

import pytest
from timbal import Agent
from timbal.types import File
from timbal.types.events import OutputEvent

pytestmark = pytest.mark.integration

_MODEL_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
}


def _provider_api_key_env(model: str) -> str | None:
    provider = model.split("/", 1)[0]
    return _MODEL_ENV_KEYS.get(provider)


def _llm_credentials_available(model: str) -> bool:
    env_key = _provider_api_key_env(model)
    if env_key and os.getenv(env_key):
        return True
    try:
        from timbal.state.config_loader import resolve_platform_config

        cfg = resolve_platform_config()
    except Exception:
        return False
    return cfg is not None and cfg.auth is not None


def _assert_agent_answer(res: OutputEvent, expected: str) -> None:
    assert res.status.code == "success", res.error
    assert res.error is None, res.error
    assert res.output is not None, res.error
    assert expected in res.output.content[0].text


@pytest.fixture(
    params=[
        "openai/gpt-4o-mini",
        "openai/gpt-4o-mini-responses",
        "google/gemini-3.1-flash-lite",
        "anthropic/claude-sonnet-4-6",
        # ? Add more tests for other models.
    ]
)
def model(request):
    model_name = request.param
    if not _llm_credentials_available(model_name):
        env_key = _provider_api_key_env(model_name) or "provider API key"
        pytest.skip(f"{env_key} not set and no platform credentials available for {model_name}")
    if model_name.startswith("openai"):
        if model_name.endswith("-responses"):
            # Responses API is now the default, so remove any disable flag
            os.environ.pop("TIMBAL_DISABLE_OPENAI_RESPONSES_API", None)
            return model_name.replace("-responses", "")
        else:
            # Disable responses API for non-responses tests
            os.environ["TIMBAL_DISABLE_OPENAI_RESPONSES_API"] = "true"
    return model_name


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.png",
        "https://content.timbal.ai/tests/test.png",
    ]
)
def png(request):
    return request.param


@pytest.mark.asyncio
async def test_png(model, png) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(png), "What's Bob's score?"]

    res = await agent(prompt=prompt).collect()
    _assert_agent_answer(res, "87")


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.pdf",
        "https://content.timbal.ai/tests/test.pdf",
    ]
)
def pdf(request):
    return request.param


@pytest.mark.asyncio
async def test_pdf(model, pdf) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(pdf), "What's Bob's score?"]

    res = await agent(prompt=prompt).collect()
    _assert_agent_answer(res, "87.2")


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.md",
        "https://content.timbal.ai/tests/test.md",
    ]
)
def md(request):
    return request.param


@pytest.mark.asyncio
async def test_md(model, md) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(md), "What's Alice's age?"]

    res = await agent(prompt=prompt).collect()
    _assert_agent_answer(res, "28")


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.csv",
        "https://content.timbal.ai/tests/test.csv",
    ]
)
def csv(request):
    return request.param


@pytest.mark.asyncio
async def test_csv(model, csv) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(csv), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert res.status.code == "success", res.error
    assert res.output is not None, res.error
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.tsv",
        "https://content.timbal.ai/tests/test.tsv",
    ]
)
def tsv(request):
    return request.param


@pytest.mark.asyncio
async def test_tsv(model, tsv) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(tsv), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert res.status.code == "success", res.error
    assert res.output is not None, res.error
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.jsonl",
        "https://content.timbal.ai/tests/test.jsonl",
    ]
)
def jsonl(request):
    return request.param


@pytest.mark.asyncio
async def test_jsonl(model, jsonl) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(jsonl), "What's Alice's score?"]

    res = await agent(prompt=prompt).collect()
    _assert_agent_answer(res, "95.5")


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.json",
        "https://content.timbal.ai/tests/test.json",
    ]
)
def json(request):
    return request.param


@pytest.mark.asyncio
async def test_json(model, json) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(json), "Is Bob still active?"]

    res = await agent(prompt=prompt).collect()
    assert res.status.code == "success", res.error
    assert res.output is not None, res.error
    assert "no" in res.output.content[0].text.lower()


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.xlsx",
        "https://content.timbal.ai/tests/test.xlsx",
    ]
)
def xlsx(request):
    return request.param


@pytest.mark.asyncio
async def test_xlsx(model, xlsx) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(xlsx), "What's Alice's score?"]

    res = await agent(prompt=prompt).collect()
    _assert_agent_answer(res, "95.5")


@pytest.fixture(
    params=[
        Path(__file__).parent.parent / "fixtures" / "test.docx",
        "https://content.timbal.ai/tests/test.docx",
    ]
)
def docx(request):
    return request.param


@pytest.mark.asyncio
async def test_docx(model, docx) -> None:
    agent = Agent(name="agent", model=model, max_tokens=10000)

    prompt = [File.validate(docx), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert res.status.code == "success", res.error
    assert res.output is not None, res.error
    assert "bob johnson" in res.output.content[0].text.lower()
