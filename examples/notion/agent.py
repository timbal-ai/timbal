# ruff: noqa: T201
"""A Notion assistant: finds, reads and writes pages and databases in your workspace.

The Notion tools and their instructions ship as a skill (skills/notion): the agent loads it
when a request involves Notion, then searches, reads the schema and acts.

Prerequisites: fill in .env from .env.example (see README.md for OAuth and API key).

Usage:
    python agent.py "Summarise the Q4 roadmap page"
    python agent.py            # interactive
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from timbal import Agent

load_dotenv()

agent = Agent(
    name="notion_assistant",
    model="anthropic/claude-sonnet-4-6",
    max_tokens=4096,
    max_iter=15,
    system_prompt=(
        "You help the user with their Notion workspace. Load the `notion` skill for anything that "
        "involves Notion, and only state what the tools returned."
    ),
    skills_path=Path(__file__).parent / "skills",
)


async def ask(prompt: str) -> None:
    result = await agent(prompt=prompt).collect()
    if result.status.code != "success":
        print(f"Error: {result.error}")
        return
    print(result.output.collect_text())


async def main() -> None:
    if len(sys.argv) > 1:
        await ask(" ".join(sys.argv[1:]))
        return
    print("Ask about your Notion workspace (empty line to quit).")
    while prompt := input("> ").strip():
        await ask(prompt)


if __name__ == "__main__":
    asyncio.run(main())
