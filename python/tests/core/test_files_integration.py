import os
from pathlib import Path

import pytest
from timbal import Agent
from timbal.types import File


@pytest.fixture(params=[
    "openai/gpt-4o-mini",
    "openai/gpt-4o-mini-responses",
    "google/gemini-2.5-flash-lite",
    "anthropic/claude-sonnet-4-0",
    # ? Add more tests for other models.
])
def model(request):
    if request.param.startswith("openai"):
        if request.param.endswith("-responses"):
            # Responses API is now the default, so remove any disable flag
            os.environ.pop("TIMBAL_DISABLE_OPENAI_RESPONSES_API", None)
            return request.param.replace("-responses", "")
        else:
            # Disable responses API for non-responses tests
            os.environ["TIMBAL_DISABLE_OPENAI_RESPONSES_API"] = "true"
    return request.param


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.png",
    "https://content.timbal.ai/tests/test.png",
])
def png(request):
    return request.param


@pytest.mark.asyncio
async def test_png(model, png) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(png), "What's Bob's score?"]

    res = await agent(prompt=prompt).collect()
    print(res)
    assert "87" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.pdf",
    "https://content.timbal.ai/tests/test.pdf",
])
def pdf(request):
    return request.param


@pytest.mark.asyncio
async def test_pdf(model, pdf) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(pdf), "What's Bob's score?"]

    res = await agent(prompt=prompt).collect()
    print(res)
    assert "87.2" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.md",
    "https://content.timbal.ai/tests/test.md",
])
def md(request):
    return request.param


@pytest.mark.asyncio
async def test_md(model, md) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(md), "What's Alice's age?"]

    res = await agent(prompt=prompt).collect()
    assert "28" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.csv",
    "https://content.timbal.ai/tests/test.csv",
])
def csv(request):
    return request.param

@pytest.mark.asyncio
async def test_csv(model, csv) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(csv), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.tsv",
    "https://content.timbal.ai/tests/test.tsv",
])
def tsv(request):
    return request.param

@pytest.mark.asyncio
async def test_tsv(model, tsv) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(tsv), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.jsonl",
    "https://content.timbal.ai/tests/test.jsonl",
])
def jsonl(request):
    return request.param


@pytest.mark.asyncio
async def test_jsonl(model, jsonl) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(jsonl), "What's Alice's score?"]

    res = await agent(prompt=prompt).collect()
    assert "95.5" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.json",
    "https://content.timbal.ai/tests/test.json",
])
def json(request):
    return request.param


@pytest.mark.asyncio
async def test_json(model, json) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(json), "Is Bob still active?"]

    res = await agent(prompt=prompt).collect()
    assert "no" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.xlsx",
    "https://content.timbal.ai/tests/test.xlsx",
])
def xlsx(request):
    return request.param


@pytest.mark.asyncio
async def test_xlsx(model, xlsx) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(xlsx), "What's Alice's score?"]

    res = await agent(prompt=prompt).collect()
    assert "95.5" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "fixtures" / "test.docx",
    "https://content.timbal.ai/tests/test.docx",
])
def docx(request):
    return request.param


@pytest.mark.asyncio
async def test_docx(model, docx) -> None:
    agent = Agent(name="agent", model=model, model_params={"max_tokens": 10000})

    prompt = [File.validate(docx), "What's Bob's full name?"]

    res = await agent(prompt=prompt).collect()
    assert "bob johnson" in res.output.content[0].text.lower()
