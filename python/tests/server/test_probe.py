import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from timbal import __version__


class TestProbeScript:
    @pytest.fixture
    def probe_script_path(self):
        return Path(__file__).parent.parent.parent / "timbal" / "server" / "probe.py"
    
    @pytest.fixture
    def tool_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "tool_fixture.py"
    
    @pytest.fixture
    def agent_fixture_file(self):
        return Path(__file__).parent / "fixtures" / "agent_fixture.py"
    
    @pytest.fixture
    def not_runnable_file(self):
        return Path(__file__).parent / "fixtures" / "not_runnable_fixture.py"

    def test_version_flag(self, probe_script_path):
        """Test the --version flag outputs correct version."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--version"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        assert f"timbal.server.http {__version__}" in result.stderr

    def test_no_import_spec_error(self, probe_script_path):
        """Test error when no import spec is provided."""
        with patch.dict(os.environ, {}, clear=True):
            result = subprocess.run(
                [sys.executable, "-m", "timbal.server.probe"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 1
            assert "No import spec provided" in result.stderr

    def test_invalid_import_spec_format(self, probe_script_path):
        """Test error with invalid import spec format."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", "invalid_format"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr

    def test_probe_tool_success(self, probe_script_path, tool_fixture_file):
        """Test successful probing of a tool."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{tool_fixture_file}::tool_fixture"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        
        # Parse the JSON output
        output_data = json.loads(result.stdout.strip())
        assert output_data["version"] == __version__
        assert output_data["type"] == "Tool"
        assert "params_model_schema" in output_data
        assert "return_model_schema" in output_data
        
        # Verify tool-specific schema content
        params_schema = output_data["params_model_schema"]
        assert "properties" in params_schema
        assert "x" in params_schema["properties"]

    def test_probe_agent_success(self, probe_script_path, agent_fixture_file):
        """Test successful probing of an agent."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{agent_fixture_file}::agent_fixture"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        
        # Parse the JSON output
        output_data = json.loads(result.stdout.strip())
        assert output_data["version"] == __version__
        assert output_data["type"] == "Agent"
        assert "params_model_schema" in output_data
        assert "return_model_schema" in output_data

    def test_env_var_timbal_runnable(self, probe_script_path, tool_fixture_file):
        """Test using TIMBAL_RUNNABLE environment variable."""
        with patch.dict('os.environ', {'TIMBAL_RUNNABLE': f'{tool_fixture_file}::tool_fixture'}):
            result = subprocess.run(
                [sys.executable, "-m", "timbal.server.probe"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            output_data = json.loads(result.stdout.strip())
            assert output_data["type"] == "Tool"

    def test_legacy_env_var_timbal_flow(self, probe_script_path, tool_fixture_file):
        """Test using legacy TIMBAL_FLOW environment variable."""
        with patch.dict('os.environ', {'TIMBAL_FLOW': f'{tool_fixture_file}::tool_fixture'}):
            result = subprocess.run(
                [sys.executable, "-m", "timbal.server.probe"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            output_data = json.loads(result.stdout.strip())
            assert output_data["type"] == "Tool"

    def test_non_runnable_object_error(self, probe_script_path, not_runnable_file):
        """Test error when loaded object is not a Runnable."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{not_runnable_file}::not_runnable_fixture"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode != 0
        assert "not a valid Runnable instance" in result.stderr

    def test_import_spec_arg_overrides_env(self, probe_script_path, tool_fixture_file, agent_fixture_file):
        """Test that --import_spec argument overrides environment variables."""
        with patch.dict('os.environ', {'TIMBAL_RUNNABLE': f'{tool_fixture_file}::tool_fixture'}):
            result = subprocess.run(
                [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{agent_fixture_file}::agent_fixture"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent.parent
            )
            
            assert result.returncode == 0
            output_data = json.loads(result.stdout.strip())
            # Should be Agent (from arg), not Tool (from env)
            assert output_data["type"] == "Agent"

    def test_empty_import_spec_parts(self, probe_script_path):
        """Test error with malformed import spec (missing ::)."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", "just_a_path"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr

    def test_too_many_import_spec_parts(self, probe_script_path):
        """Test error with malformed import spec (too many :: parts)."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", "path::target::extra"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 1
        assert "Invalid import spec format" in result.stderr

    def test_output_json_format(self, probe_script_path, tool_fixture_file):
        """Test that output is valid JSON with required fields."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{tool_fixture_file}::tool_fixture"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode == 0
        
        # Parse and validate JSON output structure
        output_data = json.loads(result.stdout.strip())
        
        required_keys = ["version", "type", "params_model_schema", "return_model_schema"]
        for key in required_keys:
            assert key in output_data, f"Missing required key: {key}"
        
        assert output_data["version"] == __version__
        assert isinstance(output_data["params_model_schema"], dict)
        assert isinstance(output_data["return_model_schema"], dict)

    def test_file_not_found_error(self, probe_script_path):
        """Test error when import file doesn't exist."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", "/nonexistent/file.py::target"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode != 0

    def test_target_not_found_error(self, probe_script_path, tool_fixture_file):
        """Test error when target object doesn't exist in file."""
        result = subprocess.run(
            [sys.executable, "-m", "timbal.server.probe", "--import_spec", f"{tool_fixture_file}::nonexistent_target"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        
        assert result.returncode != 0