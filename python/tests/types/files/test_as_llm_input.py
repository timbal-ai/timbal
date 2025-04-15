from pathlib import Path

import pytest

from timbal.types import File
from timbal import Agent


@pytest.fixture(params=[
    "gpt-4o-mini",
    "gemini-2.0-flash-lite",
    "claude-3-5-sonnet-20241022",
    # ? Add more tests for other models.
])
def model(request):
    return request.param


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.pdf",
    "https://content.timbal.ai/tests/test.pdf",
])
def pdf(request):
    return request.param


@pytest.mark.asyncio
async def test_pdf(model, pdf) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(pdf), "What's Bob's score?"]

    res = await agent.complete(prompt=prompt)
    assert "87.2" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.md",
    "https://content.timbal.ai/tests/test.md",
])
def md(request):
    return request.param


@pytest.mark.asyncio
async def test_md(model, md) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(md), "What's Alice's age?"]

    res = await agent.complete(prompt=prompt)
    assert "28" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.csv",
    "https://content.timbal.ai/tests/test.csv",
])
def csv(request):
    return request.param

@pytest.mark.asyncio
async def test_csv(model, csv) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(csv), "What's Bob's full name?"]

    res = await agent.complete(prompt=prompt)
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.tsv",
    "https://content.timbal.ai/tests/test.tsv",
])
def tsv(request):
    return request.param

@pytest.mark.asyncio
async def test_tsv(model, tsv) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(tsv), "What's Bob's full name?"]

    res = await agent.complete(prompt=prompt)
    assert "bob johnson" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.jsonl",
    "https://content.timbal.ai/tests/test.jsonl",
])
def jsonl(request):
    return request.param


@pytest.mark.asyncio
async def test_jsonl(model, jsonl) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(jsonl), "What's Alice's score?"]

    res = await agent.complete(prompt=prompt)
    assert "95.5" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.json",
    "https://content.timbal.ai/tests/test.json",
])
def json(request):
    return request.param


@pytest.mark.asyncio
async def test_json(model, json) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(json), "Is Bob still active?"]

    res = await agent.complete(prompt=prompt)
    assert "no" in res.output.content[0].text.lower()


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.xlsx",
    "https://content.timbal.ai/tests/test.xlsx",
])
def xlsx(request):
    return request.param


@pytest.mark.asyncio
async def test_xlsx(model, xlsx) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(xlsx), "What's Alice's score?"]

    res = await agent.complete(prompt=prompt)
    assert "95.5" in res.output.content[0].text


@pytest.fixture(params=[
    Path(__file__).parent / "examples" / "test.docx",
    "https://content.timbal.ai/tests/test.docx",
])
def docx(request):
    return request.param


@pytest.mark.asyncio
async def test_docx(model, docx) -> None:
    agent = Agent(model=model, max_tokens=2048)

    prompt = [File.validate(docx), "What's Bob's full name?"]

    res = await agent.complete(prompt=prompt)
    assert "bob johnson" in res.output.content[0].text.lower()
