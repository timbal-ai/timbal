import asyncio

from timbal import Workflow
from timbal.state import get_run_context


def fetch_data() -> list:
    return [
        {"id": 1, "name": "Alice", "age": 25, "city": "New York"},
        {"id": 2, "name": "Bob", "age": 30, "city": "San Francisco"},
        {"id": 3, "name": "Charlie", "age": 35, "city": "Chicago"},
        {"id": 4, "name": "Diana", "age": 28, "city": "Boston"}
    ]


def filter_data(csv_data: list, name: str) -> dict:
    for row in csv_data:
        if row["name"].lower() == name.lower():
            return row
    return {}


def prettify_result(filtered_row: dict) -> str:
    if not filtered_row:
        return "❌ No data found"
    
    return f"""
🎯 User Profile
━━━━━━━━━━━━━━
👤 Name: {filtered_row['name']}
🎂 Age: {filtered_row['age']} years old
🏙️ City: {filtered_row['city']}
🆔 ID: {filtered_row['id']}
    """.strip()


workflow = (Workflow(name="demo_workflow")
    .step(fetch_data)
    .step(filter_data, csv_data=lambda: get_run_context().step_trace("fetch_data").output)
    .step(prettify_result, filtered_row=lambda: get_run_context().step_trace("filter_data").output)
)


async def main():
    result = await workflow(name="Bob").collect()
    print(result.output) # noqa: T201


if __name__ == "__main__":
    asyncio.run(main())
