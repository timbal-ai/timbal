import textwrap
from pathlib import Path

import pytest
from timbal.codegen.transformers import apply_operation
from timbal.codegen.transformers.set_param import accessor_to_key, key_to_accessor

WORKFLOW_YAML = 'fqn: "workflow.py::workflow"\n'


@pytest.fixture
def wf_workspace(tmp_path):
    def _write(source: str) -> Path:
        (tmp_path / "workflow.py").write_text(textwrap.dedent(source))
        (tmp_path / "timbal.yaml").write_text(WORKFLOW_YAML)
        return tmp_path

    return _write


def _run(ws: Path, **kwargs) -> str:
    return apply_operation(ws, "set_param", **kwargs)


class TestMapParam:
    def test_map_param(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(ws, target="agent_b", name="prompt", param_type="map", source="agent_a", key=None, value=None)
        assert 'get_run_context().step_span("agent_a").output' in output
        assert "from timbal.state import get_run_context" in output

    def test_map_param_with_key(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(ws, target="agent_b", name="prompt", param_type="map", source="agent_a", key="output.valor", value=None)
        assert 'get_run_context().step_span("agent_a").output["valor"]' in output

    def test_map_param_with_nested_key(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        """)
        output = _run(
            ws, target="agent_b", name="prompt", param_type="map",
            source="agent_a", key="output.0.items.name", value=None,
        )
        assert 'get_run_context().step_span("agent_a").output[0]["items"]["name"]' in output

    def test_map_param_from_different_source(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")
        agent_c = Agent(name="agent_c", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b)
        workflow.step(agent_c)
        """)
        # Set prompt from agent_a
        output = _run(ws, target="agent_c", name="prompt", param_type="map", source="agent_a", key=None, value=None)
        assert 'get_run_context().step_span("agent_a").output' in output

    def test_update_existing_map_param(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run(ws, target="agent_b", name="prompt", param_type="map", source="agent_a", key="output.text", value=None)
        assert 'get_run_context().step_span("agent_a").output["text"]' in output

    def test_preserves_other_kwargs(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, depends_on=["agent_a"])
        """)
        output = _run(ws, target="agent_b", name="prompt", param_type="map", source="agent_a", key=None, value=None)
        assert "depends_on" in output
        assert 'get_run_context().step_span("agent_a").output' in output


class TestMapParamDictAccess:
    """Mapping a sub-key of a dict step output must use subscript, not attribute access.

    Custom function steps return dicts; ``.output.key`` raises AttributeError at
    runtime, whereas ``.output["key"]`` works.
    """

    def test_map_dict_subkey_uses_subscript(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def transcribe() -> dict:
            return {"transcript_text": "hello", "lang": "en"}

        def summarize(text: str) -> str:
            return text

        transcribe_tool = Tool(name="transcribe", handler=transcribe)
        summarize_tool = Tool(name="summarize", handler=summarize)

        workflow = Workflow(name="wf")
        workflow.step(transcribe_tool)
        workflow.step(summarize_tool)
        """)
        output = _run(
            ws, target="summarize", name="text", param_type="map",
            source="transcribe", key="output.transcript_text", value=None,
        )
        assert 'get_run_context().step_span("transcribe").output["transcript_text"]' in output
        assert ".output.transcript_text" not in output

    async def test_mapped_dict_subkey_runs(self, wf_workspace):
        """End-to-end: the generated workflow must execute against a dict output."""
        import importlib.util
        import sys
        import uuid

        ws = wf_workspace("""\
        from timbal import Workflow
        from timbal.core import Tool

        def transcribe() -> dict:
            return {"transcript_text": "hello world", "lang": "en"}

        def summarize(text: str) -> str:
            return f"SUMMARY: {text}"

        transcribe_tool = Tool(name="transcribe", handler=transcribe)
        summarize_tool = Tool(name="summarize", handler=summarize)

        workflow = Workflow(name="wf")
        workflow.step(transcribe_tool)
        workflow.step(summarize_tool)
        """)
        output = _run(
            ws, target="summarize", name="text", param_type="map",
            source="transcribe", key="output.transcript_text", value=None,
        )
        # Write to disk and import as a real module so the auto-wiring's
        # source introspection (inspect.getsource on the lambda) works.
        wf_path = ws / "workflow.py"
        wf_path.write_text(output)
        mod_name = f"_gen_wf_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(mod_name, wf_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
            result = await module.workflow().collect()
        finally:
            sys.modules.pop(mod_name, None)

        assert result.status.code == "success", result.error
        assert result.output == "SUMMARY: hello world"


class TestValueParam:
    def test_set_static_value_string(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        """)
        output = _run(ws, target="agent_a", name="prompt", param_type="value", source=None, key=None, value='"Hello world"')
        assert 'prompt="Hello world"' in output

    def test_set_static_value_number(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        """)
        output = _run(ws, target="agent_a", name="temperature", param_type="value", source=None, key=None, value="0.5")
        assert "temperature=0.5" in output

    def test_remove_param_with_null(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow
        from timbal.state import get_run_context

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")
        agent_b = Agent(name="agent_b", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        workflow.step(agent_b, prompt=lambda: get_run_context().step_span("agent_a").output)
        """)
        output = _run(ws, target="agent_b", name="prompt", param_type="value", source=None, key=None, value="null")
        assert "prompt=" not in output

    def test_update_existing_value_param(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a, prompt="old value")
        """)
        output = _run(ws, target="agent_a", name="prompt", param_type="value", source=None, key=None, value='"new value"')
        assert 'prompt="new value"' in output
        assert "old value" not in output


class TestValidation:
    def test_rejects_non_workflow(self, tmp_path):
        (tmp_path / "agent.py").write_text(textwrap.dedent("""\
        from timbal import Agent

        agent = Agent(name="a", model="openai/gpt-4o-mini")
        """))
        (tmp_path / "timbal.yaml").write_text('fqn: "agent.py::agent"\n')
        with pytest.raises(ValueError, match="Workflow"):
            apply_operation(tmp_path, "set_param", target="a", name="prompt", param_type="map", source="b", key=None, value=None)

    def test_map_requires_source(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        """)
        with pytest.raises(ValueError, match="--source"):
            _run(ws, target="agent_a", name="prompt", param_type="map", source=None, key=None, value=None)

    def test_value_requires_value(self, wf_workspace):
        ws = wf_workspace("""\
        from timbal import Agent, Workflow

        agent_a = Agent(name="agent_a", model="openai/gpt-4o-mini")

        workflow = Workflow(name="wf")
        workflow.step(agent_a)
        """)
        with pytest.raises(ValueError, match="--value"):
            _run(ws, target="agent_a", name="prompt", param_type="value", source=None, key=None, value=None)


class TestKeyConversion:
    def test_key_to_accessor_simple(self):
        assert key_to_accessor("output") == ".output"

    def test_key_to_accessor_attribute(self):
        assert key_to_accessor("output.cleaned") == '.output["cleaned"]'

    def test_key_to_accessor_index(self):
        assert key_to_accessor("output.0") == ".output[0]"

    def test_key_to_accessor_nested(self):
        assert key_to_accessor("output.0.items.name") == '.output[0]["items"]["name"]'

    def test_key_to_accessor_deep(self):
        assert key_to_accessor("output.0.something.something_else.0") == '.output[0]["something"]["something_else"][0]'

    def test_key_to_accessor_string_subkey_is_subscript(self):
        # First segment (`.output`) is a Span attribute; sub-keys index a dict.
        assert key_to_accessor("output.transcript_text") == '.output["transcript_text"]'

    def test_accessor_to_key_simple(self):
        assert accessor_to_key(".output") == "output"

    def test_accessor_to_key_attribute(self):
        assert accessor_to_key(".output.cleaned") == "output.cleaned"

    def test_accessor_to_key_bracket_string(self):
        assert accessor_to_key('.output["cleaned"]') == "output.cleaned"

    def test_accessor_to_key_bracket_index(self):
        assert accessor_to_key(".output[0]") == "output.0"

    def test_accessor_to_key_mixed(self):
        assert accessor_to_key(".output[0].items.name") == "output.0.items.name"

    def test_roundtrip(self):
        key = "output.0.something.something_else.0"
        assert accessor_to_key(key_to_accessor(key)) == key
