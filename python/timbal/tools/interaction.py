"""Built-in human-in-the-loop interaction tools.

These tools pause the run via :func:`timbal.state.suspend` and wait for an
externally-supplied value. The frontend renders the suspension ``payload``
(keyed by ``kind``) and resumes with ``resume={interaction_id: value}``.

Pass them straight to an agent::

    from timbal import Agent
    from timbal.tools.interaction import ask_user

    agent = Agent(name="assistant", model="...", tools=[ask_user])

When the model is blocked it calls ``ask_user``; the run ends with status
``input_required`` and emits an ``InteractionEvent``. Resume with the answer::

    result = await agent(
        prompt="...",
        parent_id=paused_run_id,
        resume={interaction_id: "use postgres"},
    ).collect()
"""

from ..state import suspend


def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the user a clarifying question and wait for their answer.

    Use ONLY when you are blocked and the answer cannot be inferred from the
    conversation or context. Ask a single, most-blocking question. Prefer
    providing ``options`` when the answer is a choice — it is more reliable and
    renders as a picker.

    Args:
        question: The question to ask the user.
        options: Optional list of choices. If omitted, free-text is expected.

    Returns:
        The user's answer.
    """
    return suspend({"question": question, "options": options}, kind="ask_user")


def confirm(action: str) -> bool:
    """Ask the user to confirm before proceeding with an action.

    Args:
        action: Human-readable description of what is about to happen.

    Returns:
        True if the user confirmed, False otherwise.
    """
    return bool(suspend({"action": action}, kind="confirm"))
