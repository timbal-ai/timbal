import asyncio
from datetime import datetime

from timbal import Agent
from timbal.state import RunContext
from timbal.state.savers import TimbalPlatformSaver


def get_datetime() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


flow = Agent(
    model="gpt-4o-mini",
    tools=[get_datetime],
).compile(state_saver=TimbalPlatformSaver())


async def main():
    run_context = RunContext()
    while True:
        prompt = input("User: ")
        if prompt == "q":
            break
        flow_output_event = await flow.complete(context=run_context, prompt=prompt)
        print(f"Agent: {flow_output_event.output}")
        run_context = RunContext(parent_id=flow_output_event.run_id)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch any Ctrl+C that wasn't caught in main()
        print("\nGoodbye!")
    finally:
        print("Goodbye!")
