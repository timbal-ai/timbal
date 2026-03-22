import os

os.environ["TIMBAL_DELTA_EVENTS"] = "true"
os.environ["TIMBAL_LOG_EVENTS"] = "START,DELTA,OUTPUT"

from openai import AsyncOpenAI
from timbal import Agent
from timbal.tools import WebSearch


def get_datetime() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


client = AsyncOpenAI(base_url="https://api.dev.timbal.ai/ace/v1")

agent = Agent(
    name="agent",
    model="openai/gpt-4.1-mini",
    base_url="https://api.dev.timbal.ai/ace/v1",
    tools=[get_datetime, WebSearch()],
)


async def main():
    prompt = "what time is it?"
    # response = await client.responses.create(
    #     model="gpt-4.1-mini",
    #     input=prompt,
    # )
    # print(response.output_text)
    await agent(prompt=prompt).collect()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
