import asyncio
import time
from collections.abc import AsyncGenerator, Generator

import pytest
from timbal import Agent, Tool
from timbal.types.message import Message

# ==============================================================================
# Test Handler Functions
# ==============================================================================

def sync_handler(x: str) -> str:
    """Simple synchronous handler for testing."""
    return f"sync:{x}"


def sync_gen_handler(count: int) -> Generator[int, None, None]:
    """Synchronous generator handler for testing."""
    for i in range(count):
        yield i


async def async_handler(x: str) -> str:
    """Simple asynchronous handler for testing."""
    await asyncio.sleep(0.01)  # Small delay to test async behavior
    return f"async:{x}"


async def async_gen_handler(count: int) -> AsyncGenerator[int, None]:
    """Asynchronous generator handler for testing."""
    for i in range(count):
        await asyncio.sleep(0.01)
        yield i


def error_handler(message: str) -> str:
    """Handler that raises an exception for testing error handling."""
    raise ValueError(f"Test error: {message}")


async def async_error_handler(message: str) -> str:
    """Async handler that raises an exception for testing error handling."""
    await asyncio.sleep(0.01)
    raise ValueError(f"Test async error: {message}")


# ==============================================================================
# Test Tools
# ==============================================================================

@pytest.fixture
def simple_tool():
    """Create a simple tool for testing."""
    return Tool(name="simple", handler=sync_handler)


@pytest.fixture
def async_tool():
    """Create an async tool for testing."""
    return Tool(name="async", handler=async_handler)


@pytest.fixture
def gen_tool():
    """Create a generator tool for testing."""
    return Tool(name="gen", handler=sync_gen_handler)


@pytest.fixture
def async_gen_tool():
    """Create an async generator tool for testing."""
    return Tool(name="async_gen", handler=async_gen_handler)


@pytest.fixture
def error_tool():
    """Create a tool that raises errors for testing."""
    return Tool(name="error", handler=error_handler)


# ==============================================================================
# Test Agents
# ==============================================================================

@pytest.fixture
def simple_agent():
    """Create a simple agent for testing."""
    def get_time() -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S")
    
    return Agent(
        name="simple_agent",
        model="openai/gpt-4o-mini",
        tools=[get_time]
    )


@pytest.fixture
def multi_tool_agent():
    """Create an agent with multiple tools for testing."""
    def add(a: int, b: int) -> int:
        return a + b
    
    def multiply(a: int, b: int) -> int:
        return a * b
    
    def greet(name: str) -> str:
        return f"Hello, {name}!"
    
    return Agent(
        name="multi_tool_agent",
        model="openai/gpt-4o-mini",
        tools=[add, multiply, greet]
    )


@pytest.fixture
def math_agent():
    """Create an agent specialized for math operations."""
    def calculate(expression: str) -> str:
        """Safely evaluate simple math expressions."""
        try:
            # Only allow basic math operations for safety
            allowed_chars = set('0123456789+-*/().= ')
            if all(c in allowed_chars for c in expression):
                result = eval(expression)
                return f"{expression} = {result}"
            else:
                return "Invalid expression - only basic math allowed"
        except Exception as e:
            return f"Error: {str(e)}"
    
    return Agent(
        name="math_agent",
        model="openai/gpt-4o-mini",
        tools=[calculate],
        system_prompt="You are a helpful math assistant. Use the calculate tool for any math operations."
    )


# ==============================================================================
# Parametrized Test Data
# ==============================================================================

TOOL_HANDLERS = [
    ("sync", sync_handler),
    ("async", async_handler),
    ("sync_gen", sync_gen_handler),
    ("async_gen", async_gen_handler),
]

ERROR_SCENARIOS = [
    ("sync_error", error_handler),
    ("async_error", async_error_handler),
]