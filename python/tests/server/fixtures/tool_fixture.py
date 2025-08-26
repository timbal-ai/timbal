from timbal import Tool


def simple_handler(x: str) -> str:
    """A simple test handler."""
    return f"result: {x}"

tool_fixture = Tool(name="test_tool", handler=simple_handler)