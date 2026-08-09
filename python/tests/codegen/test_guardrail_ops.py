"""Codegen ops: add-guardrail / remove-guardrail."""

import textwrap
from pathlib import Path

import pytest
from timbal.codegen.transformers import apply_operation

AGENT_YAML = 'fqn: "agent.py::agent"\n'
WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def workspace(tmp_path):
    def _write(source: str, *, filename: str = "agent.py", yaml: str = AGENT_YAML) -> Path:
        (tmp_path / filename).write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(yaml)
        return tmp_path

    return _write


BARE_AGENT = """\
from timbal import Agent

agent = Agent(name="agent", model="openai/gpt-4o-mini", tools=[])
"""


class TestAddGuardrail:
    def test_adds_list_when_absent(self, workspace):
        ws = workspace(BARE_AGENT)
        out = apply_operation(ws, "add_guardrail", spec="pii:redact", step=None)
        assert 'guardrails=["pii:redact"]' in out

    def test_sets_default_preset(self, workspace):
        ws = workspace(BARE_AGENT)
        out = apply_operation(ws, "add_guardrail", spec="default", step=None)
        assert 'guardrails="default"' in out

    def test_appends_to_existing_list(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["pii:redact"])
        """)
        out = apply_operation(ws, "add_guardrail", spec="injection:block", step=None)
        assert '"pii:redact"' in out and '"injection:block"' in out

    def test_same_rail_name_replaces_entry(self, workspace):
        """Duplicate rail names are invalid at runtime — changing the action replaces."""
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["pii:redact"])
        """)
        out = apply_operation(ws, "add_guardrail", spec="pii:block", step=None)
        assert '"pii:block"' in out and '"pii:redact"' not in out

    def test_idempotent_re_add(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["pii:redact"])
        """)
        out = apply_operation(ws, "add_guardrail", spec="pii:redact", step=None)
        assert out.count("pii:redact") == 1

    def test_default_string_expands_before_append(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails="default")
        """)
        out = apply_operation(ws, "add_guardrail", spec="moderation:warn", step=None)
        assert '"pii:redact"' in out and '"secrets"' in out and '"injection:block"' in out
        assert '"moderation:warn"' in out

    def test_unknown_shorthand_rejected(self, workspace):
        ws = workspace(BARE_AGENT)
        with pytest.raises(ValueError, match="Unknown guardrail shorthand"):
            apply_operation(ws, "add_guardrail", spec="pie:redact", step=None)

    def test_non_literal_value_rejected(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        my_rails = ["pii:redact"]
        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=my_rails)
        """)
        with pytest.raises(ValueError, match="not a literal"):
            apply_operation(ws, "add_guardrail", spec="secrets", step=None)

    def test_workflow_step_target(self, workspace):
        ws = workspace(
            """\
            from timbal import Agent, Workflow

            agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
            workflow = Workflow(name="workflow").step(agent_a)
            """,
            filename="workflow.py",
            yaml=WORKFLOW_YAML,
        )
        out = apply_operation(ws, "add_guardrail", spec="pii:redact", step="agent_a")
        # the guardrail lands on the step's Agent constructor (formatter may wrap lines)
        agent_a_src = out.split("workflow =")[0]
        assert 'guardrails=["pii:redact"]' in agent_a_src

    def test_step_on_agent_entry_point_rejected(self, workspace):
        ws = workspace(BARE_AGENT)
        with pytest.raises(ValueError, match="--step requires a Workflow"):
            apply_operation(ws, "add_guardrail", spec="pii:redact", step="agent_a")


class TestRemoveGuardrail:
    def test_removes_by_rail_name(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["pii:redact", "secrets"])
        """)
        out = apply_operation(ws, "remove_guardrail", name="pii", step=None)
        assert '"pii:redact"' not in out and '"secrets"' in out

    def test_removing_last_rail_drops_kwarg(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["pii:redact"])
        """)
        out = apply_operation(ws, "remove_guardrail", name="pii", step=None)
        assert "guardrails" not in out

    def test_default_string_expands_on_removal(self, workspace):
        ws = workspace("""\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails="default")
        """)
        out = apply_operation(ws, "remove_guardrail", name="injection", step=None)
        assert '"pii:redact"' in out and '"secrets"' in out
        assert "injection" not in out

    def test_removing_absent_rail_is_idempotent(self, workspace):
        source = """\
        from timbal import Agent

        agent = Agent(name="agent", model="openai/gpt-4o-mini", guardrails=["secrets"])
        """
        ws = workspace(source)
        out = apply_operation(ws, "remove_guardrail", name="pii", step=None)
        assert '"secrets"' in out

    def test_no_kwarg_is_idempotent(self, workspace):
        ws = workspace(BARE_AGENT)
        out = apply_operation(ws, "remove_guardrail", name="pii", step=None)
        assert "guardrails" not in out
