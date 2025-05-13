import asyncio

from timbal import Flow
from timbal.state.savers import TimbalPlatformSaver


def get_email() -> str:
    return """Subject: Meeting Reminder

Hi,

I just wanted to remind you about the meeting tomorrow at 10am.

Best,
John Doe"""


flow = (Flow()
    .add_step(get_email)
    .add_llm(model="gpt-4.1-nano")
    .set_data_value("llm.prompt", "Please summarize this email: {{get_email.return}}")
    .set_output("llm.return", "email_summary")
).compile(state_saver=TimbalPlatformSaver())


async def main():
    flow_output_event = await flow.complete()
    print("Email summary: ", flow_output_event.output["email_summary"].content[0].text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Catch any Ctrl+C that wasn't caught in main()
        print("\nGoodbye!")
    finally:
        print("Goodbye!")
