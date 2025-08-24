"""
Sample agent for testing eval_v2 functionality.
This simulates a real agent file that users would create.
"""
import time

from timbal import Agent


def get_current_time() -> str:
    """Get the current time in a readable format."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


def greet_person(name: str) -> str:
    """Greet a person by name."""
    return f"Hello, {name}! Nice to meet you."


def calculate_expression(expression: str) -> str:
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


# Create the agent that will be loaded by the eval system
agent = Agent(
    name="sample_agent",
    model="openai/gpt-4o-mini",
    tools=[get_current_time, add_numbers, greet_person, calculate_expression],
    system_prompt="You are a helpful assistant with access to time, math, and greeting tools."
)
