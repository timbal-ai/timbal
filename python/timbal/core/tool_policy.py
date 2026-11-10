"""Run tool policies (``RunContext._tool_policy``): how one is described in a trace, and restored from one.

A policy is ``"read_only"``, a collection of tool names, a callable ``(tool) -> bool``, or ``None`` (all
tools). Every agent of a run filters what its model sees with it (``Agent._tool_allowed``) and records it on
its span; the run's root span records it too (``metadata.tool_policy``). A callable is not stored: a run
continuing a paused one rebuilds it from the tool paths the agents noted they were offered
(``tool_policy_allowed``; agents of runs started inside a call note theirs on that call's span too,
``nested_tool_policy_allowed``).
"""

from dataclasses import dataclass
from typing import Any

READ_ONLY_POLICY = "read_only"


def describe_tool_policy(policy: Any) -> Any:
    """The policy as stored in a trace: ``None``, ``"read_only"``, ``"custom"`` (a callable) or sorted names."""
    if policy is None or policy == READ_ONLY_POLICY:
        return policy
    if callable(policy):
        return "custom"
    return sorted(policy)


@dataclass(frozen=True)
class ToolPaths:
    """A tool policy restored from a trace: the tools with these paths (agent-scoped, unlike names)."""

    paths: frozenset[str]

    def __call__(self, tool: Any) -> bool:
        return getattr(tool, "_path", None) in self.paths


def allowed_tool_paths(trace: Any) -> frozenset[str]:
    """What a callable tool policy let through in ``trace``, as last decided.

    Each agent notes on its span what its latest tool resolution allowed (``tool_policy_allowed``), and an
    agent of a run started inside one of this run's calls (a tool or step body, a background tool) notes it on
    that call's span as well (``nested_tool_policy_allowed``, by agent path), since its own spans live in
    another trace. An agent seen more than once counts by its latest record, so a later decision (a
    permission taken back) wins over an earlier one.
    """
    latest: dict[str, tuple[int, list[str]]] = {}
    for span in trace.values():
        metadata = span.metadata or {}
        t0 = span.t0 or 0
        records = list((metadata.get("nested_tool_policy_allowed") or {}).items())
        if "tool_policy_allowed" in metadata:
            records.append((span.path, metadata["tool_policy_allowed"]))
        for path, allowed in records:
            seen = latest.get(path)
            if seen is None or t0 >= seen[0]:
                latest[path] = (t0, allowed)
    return frozenset(path for _, allowed in latest.values() for path in allowed)


def restore_tool_policy(described: Any, trace: Any) -> Any:
    """The policy ``describe_tool_policy`` stored, back from ``trace``: a callable becomes the paths it allowed
    (an agent that never resolved its tools in the run gets none)."""
    if described == "custom":
        return ToolPaths(allowed_tool_paths(trace))
    if isinstance(described, list):
        return frozenset(described)
    return described
