from datetime import datetime

from pytz import timezone as tz
from timbal import Agent


def get_datetime(timezone: str) -> str:
    now = datetime.now(tz(timezone))
    return now.strftime("%Y-%m-%d %H:%M:%S")


agent = Agent(
    name="agent",
    model="anthropic/claude-haiku-4-5",
    tools=[get_datetime],
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
