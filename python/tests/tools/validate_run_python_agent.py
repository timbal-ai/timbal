"""
Manual validation: RunPython with a real Agent on realistic use cases.

Run:
  uv run python python/tests/tools/validate_run_python_agent.py
"""
# ruff: noqa: T201  # this is a manual runner; prints are the intended output

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from timbal import Agent
from timbal.tools.run_python import RunPython

# Load API keys from examples if present
load_dotenv(Path(__file__).resolve().parents[3] / "examples" / "mcp" / ".env")

SALES_CSV = """product,units,price
Widget A,120,9.99
Widget B,85,14.50
Gadget C,200,4.25
Widget A,30,9.99
Gadget C,50,4.25
Premium D,15,49.00
Widget B,40,14.50
"""

PRICE_CATALOG = {
    "WIDGET-A": 9.99,
    "WIDGET-B": 14.50,
    "GADGET-C": 4.25,
    "PREMIUM-D": 49.00,
}


def get_unit_price(sku: str) -> float:
    """Look up unit price for a SKU from the product catalog."""
    key = sku.upper().replace(" ", "-")
    if key not in PRICE_CATALOG:
        raise ValueError(f"Unknown SKU: {sku}")
    return PRICE_CATALOG[key]


async def scenario_pandas_analysis() -> None:
    """Use case: data analyst computes top products by revenue with pandas."""
    print("\n=== Scenario 1: Sales analysis with pandas (RunPython only) ===")
    agent = Agent(
        name="sales_analyst",
        model="openai/gpt-4o-mini",
        max_tokens=1024,
        system_prompt=(
            "You are a data analyst. Always use run_python to analyze data. "
            "Write Python that uses pandas. Put the CSV in a triple-quoted string variable. "
            "Return a concise summary of findings after running the code."
        ),
        tools=[RunPython(executor="auto", timeout=120)],
    )
    prompt = (
        "Analyze this sales CSV and tell me the top 3 products by total revenue "
        "(units * price summed per product). Include the revenue numbers.\n\n"
        f"CSV:\n{SALES_CSV}"
    )
    result = await agent(prompt=prompt).collect()
    text = result.output.collect_text() if result.output else ""
    print("Agent response:", text[:800])
    assert result.error is None, result.error
    assert result.output is not None
    # Ground truth: Widget A = 150*9.99=1498.5, Gadget C = 250*4.25=1062.5, Widget B = 125*14.5=1812.5
    # Top 3 by revenue: Widget B (~1812.5), Widget A (~1498.5), Gadget C (~1062.5)
    lower = text.lower()
    assert "widget b" in lower or "widget-b" in lower.replace(" ", "")
    print("Scenario 1: PASS")


async def scenario_code_mode_pricing() -> None:
    """Use case: code mode — script calls get_unit_price for each line item."""
    print("\n=== Scenario 2: Code mode batch pricing (RunPython + get_unit_price) ===")
    agent = Agent(
        name="pricing_agent",
        model="openai/gpt-4o-mini",
        max_tokens=1024,
        system_prompt=(
            "You compute order totals using run_python. "
            "Inside the script you MUST call get_unit_price(sku) for each SKU — "
            "do not hardcode prices. SKUs: WIDGET-A x2, GADGET-C x5, PREMIUM-D x1."
        ),
        tools=[RunPython(tools=[get_unit_price], executor="auto", timeout=60)],
    )
    prompt = (
        "Use run_python to calculate the total order value for: "
        "2x WIDGET-A, 5x GADGET-C, 1x PREMIUM-D. "
        "Call get_unit_price for each SKU inside the script. Return the numeric total."
    )
    result = await agent(prompt=prompt).collect()
    text = result.output.collect_text() if result.output else ""
    print("Agent response:", text[:500])
    assert result.error is None, result.error
    expected = 2 * 9.99 + 5 * 4.25 + 1 * 49.00  # 90.23
    assert "90.23" in text.replace(",", "") or "90.2" in text
    print(f"Scenario 2: PASS (expected total {expected:.2f})")


async def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY required for live agent validation")
    await scenario_pandas_analysis()
    await scenario_code_mode_pricing()
    print("\nAll live agent scenarios passed.")


if __name__ == "__main__":
    asyncio.run(main())
