import time
from typing import Any

import pytest

from timbal.core_v2.context import get_run_context
from timbal.core_v2.tool import Tool
from timbal.core_v2.handlers import llm_router


def sync_handler(x: Any) -> Any:
    run_context = get_run_context()
    print("Run context: ", run_context)
    return x


def sync_gen_handler(x: Any) -> Any:
    for i in range(x):
        time.sleep(1)
        run_context = get_run_context()
        print("Run context: ", run_context)
        yield i


async def async_handler(x: Any) -> Any:
    run_context = get_run_context()
    print("Run context: ", run_context)
    return x


def test_missing_handler():
    with pytest.raises(ValueError):
        Tool()


def test_invalid_handler():
    handler = object()
    with pytest.raises(ValueError):
        Tool(handler=handler)


def test_lambda_without_name():
    with pytest.raises(ValueError):
        Tool(handler=lambda x: x)


def test_lambda_with_name():
    tool = Tool(
        name="identity",
        handler=lambda x: x,
    )

    assert tool.params_model_schema["title"] == "IdentityParams"


def test_fn_without_name():
    def identity(x: Any) -> Any:
        return x

    tool = Tool(handler=identity)

    assert tool.params_model_schema["title"] == "IdentityParams"


def test_fn_with_name():
    def identity(x: Any) -> Any:
        return x

    tool = Tool(
        name="my_tool",
        handler=identity,
    )

    assert tool.params_model_schema["title"] == "MyToolParams"


@pytest.mark.asyncio
async def test_sync():
    tool = Tool(handler=sync_handler)

    async for event in tool(x="hello"):
        pass


@pytest.mark.asyncio
async def test_sync_fixed():
    tool = Tool(
        handler=sync_handler, 
        fixed_params={"x": "hello"},
    )

    async for event in tool():
        pass


@pytest.mark.asyncio
async def test_async():
    tool = Tool(handler=async_handler)

    async for event in tool(x="hello"):
        pass

    

import asyncio
import time

@pytest.mark.asyncio
async def test_sync_gen():
    tool = Tool(handler=sync_gen_handler)
    start = time.time()

    async def run_tool():
        async for event in tool(x=5):
            pass

    # Launch two runs concurrently
    await asyncio.gather(run_tool(), run_tool())
    elapsed = time.time() - start

    print(f"Elapsed: {elapsed:.2f}s")
    # Assert that the runs were concurrent (should be just over 5s, not 10s)
    assert elapsed < 8, "Tool runs did not execute concurrently!"


@pytest.mark.asyncio
async def test_llm():
    tool = Tool(handler=llm_router)

    async for event in tool(model="gpt-4o-mini", messages=["hello"]):
        pass
