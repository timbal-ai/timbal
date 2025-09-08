import asyncio
from datetime import datetime

from timbal import Agent
from timbal.state import RunContext, set_run_context
from timbal.state.savers import TimbalPlatformSaver


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


agent = Agent(
    model="gpt-4.1-nano",
    tools=[get_datetime],
    state_saver=TimbalPlatformSaver(),
)


async def main():
    while True:
        prompt = input("User: ")
        if prompt == "q":
            break
        agent_output_event = await agent.complete(prompt=prompt)
        print(f"Agent: {agent_output_event.output}")
        run_context = RunContext(parent_id=agent_output_event.run_id)
        set_run_context(run_context)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch any Ctrl+C that wasn't caught in main()
        print("\nGoodbye!")
    finally:
        print("Goodbye!")
