import asyncio
from datetime import datetime

from dotenv import load_dotenv
from timbal import Agent

load_dotenv()


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = Agent(
    name="demo_agent",
    model="openai/gpt-4o-mini",
    tools=[get_datetime],
)


async def main():
    while True:
        prompt = input("User: ")
        if prompt == "q":
            break
        agent_output_event = await agent(prompt=prompt).collect()
        print(f"Agent: {agent_output_event.output}") # noqa: T201


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch any Ctrl+C that wasn't caught in main()
        print("\nGoodbye!") # noqa: T201
    finally:
        print("Goodbye!") # noqa: T201
