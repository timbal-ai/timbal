from timbal import Agent


def get_greeting(name: str) -> str:
    """Get a greeting for someone."""
    return f"Hello, {name}!"

def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

agent_fixture = Agent(
    name="test_agent",
    model="openai/gpt-4o-mini",
    tools=[get_greeting, add_numbers]
)