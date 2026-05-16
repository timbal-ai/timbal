import asyncio

from timbal import Workflow

workflow = Workflow(name="workflow")


async def main():
    result = await workflow().collect()
    print(result.output)  # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
