"""Run tool policies (``RunContext._tool_policy``): how one is described in a trace, and restored from one.

A policy is ``"read_only"``, a collection of tool names, a callable ``(tool) -> bool``, or ``None`` (all
tools). Every agent of a run filters what its model sees with it (``Agent._tool_allowed``) and records it on
its span; the run's root span records it too (``metadata.tool_policy``). A callable is not stored: a run
continuing a paused one rebuilds it from the tool paths the agents noted they were offered
(``tool_policy_allowed``, timed by ``tool_policy_decided_at``; agents of runs started inside a call note
theirs on that call's span too, ``nested_tool_policy_allowed``).
"""

from dataclasses import dataclass
from typing import Any

READ_ONLY_POLICY = "read_only"


def describe_tool_policy(policy: Any) -> Any:
    """The policy as stored in a trace: ``None``, ``"read_only"``, ``"custom"`` (a callable), sorted names,
    or :class:`ReadOnlyWithin`'s own form."""
    if policy is None or policy == READ_ONLY_POLICY:
        return policy
    if isinstance(policy, ReadOnlyWithin):
        return policy.to_stored()
    if callable(policy):
        return "custom"
    return sorted(policy)


@dataclass(frozen=True)
class ReadOnlyWithin:
    """``"read_only"`` narrowed to what another policy allowed.

    The policy of a turn that follows a run which was itself restricted — a recovery turn after a crash
    (``ChatSession``): read-only, and never reaching further than the run it reports on could. ``within``
    holds that run's policy as names, or as tool paths (``by_path``) when it was a callable's. Not callable
    on purpose: the read-only half is applied by the agent (``Agent._tool_allowed``), so a caller that took
    this for a plain predicate would get a wider policy than it is.
    """

    within: frozenset[str]
    by_path: bool = False

    def allows(self, tool: Any) -> bool:
        """The ``within`` half alone (the agent applies read-only on top)."""
        return (getattr(tool, "_path", None) if self.by_path else tool.name) in self.within

    def to_stored(self) -> dict[str, Any]:
        return {"read_only_within": sorted(self.within), "by": "path" if self.by_path else "name"}

    @classmethod
    def from_stored(cls, stored: dict[str, Any]) -> "ReadOnlyWithin":
        return cls(frozenset(stored.get("read_only_within") or ()), stored.get("by") == "path")


def read_only_within(policy: Any) -> Any:
    """``"read_only"``, narrowed to what ``policy`` allowed: the policy for a turn that follows a run under
    ``policy`` and may only read. ``None`` / ``"read_only"`` narrow to nothing, so they stay ``"read_only"``;
    a callable that could not be restored (no trace) allowed nothing, and neither does this."""
    if policy is None or policy == READ_ONLY_POLICY:
        return READ_ONLY_POLICY
    if isinstance(policy, ReadOnlyWithin):
        return policy
    if isinstance(policy, ToolPaths):
        return ReadOnlyWithin(policy.paths, by_path=True)
    if callable(policy):
        return ReadOnlyWithin(frozenset(), by_path=True)
    return ReadOnlyWithin(frozenset(policy))


@dataclass(frozen=True)
class ToolPaths:
    """A tool policy restored from a trace: the tools with these paths (agent-scoped, unlike names)."""

    paths: frozenset[str]

    def __call__(self, tool: Any) -> bool:
        return getattr(tool, "_path", None) in self.paths


def allowed_tool_paths(trace: Any) -> frozenset[str]:
    """What a callable tool policy let through in ``trace``, as last decided.

    Each agent notes on its span what its latest tool resolution allowed (``tool_policy_allowed``, made at
    ``tool_policy_decided_at``, ns), and an agent of a run started inside one of this run's calls (a tool or step
    body, a background tool) notes it on that call's span as well (``nested_tool_policy_allowed``: agent path ->
    ``{"allowed", "at"}``), since its own spans live in another trace. An agent seen more than once counts by its
    latest decision, so a permission taken back stays taken back, also when concurrent calls decide out of the
    order they started in. Records from before decisions were timed fall back to their span's start.
    """
    latest: dict[str, tuple[int, list[str]]] = {}
    for span in trace.values():
        metadata = span.metadata or {}
        started = (span.t0 or 0) * 1_000_000  # ms -> ns
        records = [
            (path, record.get("at", started), record.get("allowed", []))
            for path, record in (metadata.get("nested_tool_policy_allowed") or {}).items()
        ]
        if "tool_policy_allowed" in metadata:
            records.append(
                (span.path, metadata.get("tool_policy_decided_at", started), metadata["tool_policy_allowed"])
            )
        for path, at, allowed in records:
            seen = latest.get(path)
            if seen is None or at >= seen[0]:
                latest[path] = (at, allowed)
    return frozenset(path for _, allowed in latest.values() for path in allowed)


def restore_tool_policy(described: Any, trace: Any) -> Any:
    """The policy ``describe_tool_policy`` stored, back from ``trace``: a callable becomes the paths it allowed
    (an agent that never resolved its tools in the run gets none)."""
    if described == "custom":
        return ToolPaths(allowed_tool_paths(trace))
    if isinstance(described, dict) and "read_only_within" in described:
        return ReadOnlyWithin.from_stored(described)
    if isinstance(described, list):
        return frozenset(described)
    return described
