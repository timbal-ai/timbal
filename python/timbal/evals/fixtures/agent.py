import asyncio
from datetime import datetime

from pytz import timezone as tz
from timbal import Agent


def get_datetime(timezone: str) -> str:
    """Get the current datetime in a specific timezone."""
    now = datetime.now(tz(timezone))
    return now.strftime("%Y-%m-%d %H:%M:%S")


def get_weather(city: str) -> str:
    """Get the current weather for a city (mock)."""
    return f"Sunny, 22°C in {city}"


def get_stock_price(symbol: str) -> float:
    """Get the current stock price (mock)."""
    prices = {"AAPL": 178.50, "GOOGL": 141.25, "MSFT": 378.90}
    return prices.get(symbol.upper(), 100.0)


def search_products(query: str, max_results: int = 5) -> list[dict]:
    """Search for products (mock)."""
    return [
        {"id": f"prod_{i}", "name": f"{query} item {i}", "price": 10.0 * i} for i in range(1, min(max_results, 5) + 1)
    ]


async def fetch_user_data(user_id: str) -> dict:
    """Fetch user data with simulated delay."""
    await asyncio.sleep(0.1)  # Simulate API call
    return {"id": user_id, "name": f"User {user_id}", "email": f"{user_id}@example.com"}


async def fetch_user_orders(user_id: str) -> list[dict]:
    """Fetch user orders with simulated delay."""
    await asyncio.sleep(0.1)  # Simulate API call
    return [{"order_id": f"ord_{i}", "total": 50.0 * i} for i in range(1, 4)]


# Basic agent with single tool
agent = Agent(
    name="agent",
    model="anthropic/claude-haiku-4-5",
    tools=[get_datetime],
    model_params={"max_tokens": 8192},
)

# Agent with multiple tools for testing parallel execution
multi_tool_agent = Agent(
    name="multi_tool_agent",
    model="anthropic/claude-haiku-4-5",
    tools=[get_datetime, get_weather, get_stock_price],
    model_params={"max_tokens": 8192},
)

# Agent with async tools that can run in parallel
parallel_agent = Agent(
    name="parallel_agent",
    model="anthropic/claude-haiku-4-5",
    tools=[fetch_user_data, fetch_user_orders],
    model_params={"max_tokens": 8192},
)

# Agent with search capability
search_agent = Agent(
    name="search_agent",
    model="anthropic/claude-haiku-4-5",
    tools=[search_products, get_stock_price],
    model_params={"max_tokens": 8192},
)


# --- Subagent examples ---


def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of text (mock)."""
    if any(word in text.lower() for word in ["great", "good", "excellent", "happy"]):
        return "positive"
    elif any(word in text.lower() for word in ["bad", "terrible", "sad", "angry"]):
        return "negative"
    return "neutral"


def summarize_text(text: str) -> str:
    """Summarize text (mock)."""
    words = text.split()
    if len(words) > 20:
        return " ".join(words[:20]) + "..."
    return text


# A simple research subagent that can analyze and summarize
research_subagent = Agent(
    name="research_subagent",
    model="anthropic/claude-haiku-4-5",
    tools=[analyze_sentiment, summarize_text],
    system_prompt="You are a research assistant. Use the tools to analyze and summarize text.",
    model_params={"max_tokens": 4096},
)

# Main agent that uses the research subagent as a tool
main_agent_with_subagent = Agent(
    name="main_agent_with_subagent",
    model="anthropic/claude-haiku-4-5",
    tools=[get_datetime, research_subagent],
    system_prompt="You are a helpful assistant. Use the research_subagent for text analysis tasks.",
    model_params={"max_tokens": 8192},
)


async def main():
    while True:
        prompt = input("User: ")
        if prompt.strip() == "q":
            break
        await agent(prompt=prompt).collect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
