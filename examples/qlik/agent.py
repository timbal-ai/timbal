# ruff: noqa: T201
"""A Qlik analyst: answers business questions with figures from your Qlik Cloud apps.

The Qlik tools and their instructions ship as a skill (skills/qlik): the agent loads it
when a question needs Qlik, then finds the app, reads its fields and runs the query.

Prerequisites: fill in .env from .env.example (see README.md for both auth modes).

Usage:
    python agent.py "What were total sales by region in 2026?"
    python agent.py            # interactive
"""

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
from timbal import Agent

load_dotenv()

agent = Agent(
    name="qlik_analyst",
    model="anthropic/claude-sonnet-4-6",
    max_tokens=4096,
    max_iter=15,
    system_prompt=(
        "You are a data analyst. Business figures live in Qlik Sense: load the `qlik` skill "
        "for any question about them and answer only with figures the tools returned."
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
    print("Ask about your Qlik data (empty line to quit).")
    while prompt := input("> ").strip():
        await ask(prompt)


if __name__ == "__main__":
    asyncio.run(main())
