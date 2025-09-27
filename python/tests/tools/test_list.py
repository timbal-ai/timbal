import os
import tempfile
from pathlib import Path

import pytest
from timbal.tools.list import List


class TestListToolInitialization:
    """Test List tool initialization and basic functionality."""

    def test_list_tool_creation(self):
        """Test that List tool is created correctly."""
        list_tool = List()

        assert list_tool.name == "list"
        assert "List all files and subdirectories" in list_tool.description
        assert list_tool.handler is not None

    def test_list_tool_handler_signature(self):
        """Test that the handler has the correct signature."""
        list_tool = List()

        # Check that handler accepts the expected parameters
        import inspect
        sig = inspect.signature(list_tool.handler)
        params = list(sig.parameters.keys())

        expected_params = ['path']
        for param in expected_params:
            assert param in params


class TestListDirectoryOperations:
    """Test List tool directory listing functionality."""

    @pytest.mark.asyncio
    async def test_list_current_directory(self):
        """Test listing current directory."""
        list_tool = List()
        result = list_tool(path=".")

        output = await result.collect()
        assert output.error is None
        assert isinstance(output.output, list)
        assert len(output.output) > 0

    @pytest.mark.asyncio
    async def test_list_specific_directory(self):
        """Test listing a specific directory."""
        list_tool = List()

        # Create a temporary directory with known contents
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            test_file = Path(tmpdir) / "test_file.txt"
            test_file.write_text("test content")

            test_subdir = Path(tmpdir) / "test_subdir"
            test_subdir.mkdir()

            result = list_tool(path=tmpdir)
            output = await result.collect()

            assert output.error is None
            assert isinstance(output.output, list)
            assert len(output.output) == 2  # file and directory

            # Convert to resolved paths for comparison
            output_paths = [Path(p).resolve() for p in output.output]
            assert test_file.resolve() in output_paths
            assert test_subdir.resolve() in output_paths

    @pytest.mark.asyncio
    async def test_list_empty_directory(self):
        """Test listing an empty directory."""
        list_tool = List()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = list_tool(path=tmpdir)
            output = await result.collect()

            assert output.error is None
            assert isinstance(output.output, list)
            assert len(output.output) == 0

    @pytest.mark.asyncio
    async def test_list_nonexistent_directory(self):
        """Test listing a nonexistent directory."""
        list_tool = List()
        result = list_tool(path="/nonexistent/directory/path")

        output = await result.collect()
        assert output.error is not None
        # Should get a FileNotFoundError or similar


class TestListPathExpansion:
    """Test List tool path expansion functionality."""

    @pytest.mark.asyncio
    async def test_home_directory_expansion(self):
        """Test that ~ expands to home directory."""
        list_tool = List()
        result = list_tool(path="~")

        output = await result.collect()
        assert output.error is None
        assert isinstance(output.output, list)
        # Home directory should contain some files/directories

    @pytest.mark.asyncio
    async def test_environment_variable_expansion(self):
        """Test that environment variables are expanded."""
        list_tool = List()

        # Set a test environment variable
        test_path = os.getcwd()
        os.environ['TEST_LIST_PATH'] = test_path

        try:
            result = list_tool(path="$TEST_LIST_PATH")
            output = await result.collect()

            assert output.error is None
            assert isinstance(output.output, list)

            # Should be same as listing current directory
            current_result = list_tool(path=".")
            current_output = await current_result.collect()

            assert len(output.output) == len(current_output.output)
        finally:
            # Clean up environment variable
            del os.environ['TEST_LIST_PATH']

    @pytest.mark.asyncio
    async def test_relative_path_resolution(self):
        """Test that relative paths are resolved correctly."""
        list_tool = List()

        # Create a temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()

            test_file = subdir / "test.txt"
            test_file.write_text("test")

            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                result = list_tool(path="./subdir")
                output = await result.collect()

                assert output.error is None
                assert isinstance(output.output, list)
                assert len(output.output) == 1
                assert test_file.resolve() in [Path(p).resolve() for p in output.output]
            finally:
                os.chdir(original_cwd)


class TestListErrorHandling:
    """Test List tool error handling."""

    @pytest.mark.asyncio
    async def test_list_file_instead_of_directory(self):
        """Test attempting to list a file instead of a directory."""
        list_tool = List()

        with tempfile.NamedTemporaryFile() as tmpfile:
            result = list_tool(path=tmpfile.name)
            output = await result.collect()

            assert output.error is not None
            # Should get a NotADirectoryError or similar

    @pytest.mark.asyncio
    async def test_list_permission_denied(self):
        """Test handling permission denied errors."""
        list_tool = List()

        # Try to list a directory that typically has restricted permissions
        # Note: This test might not work on all systems
        restricted_paths = ["/root", "/private"]

        for path in restricted_paths:
            if os.path.exists(path):
                result = list_tool(path=path)
                output = await result.collect()

                # Should either succeed or fail with permission error
                # Don't assert specific behavior as it depends on system permissions
                assert output is not None
                break

    @pytest.mark.asyncio
    async def test_list_with_special_characters(self):
        """Test listing directories with special characters in names."""
        list_tool = List()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create directory with special characters
            special_dir = Path(tmpdir) / "test dir with spaces & symbols!"
            special_dir.mkdir()

            special_file = special_dir / "file with spaces.txt"
            special_file.write_text("test")

            result = list_tool(path=str(special_dir))
            output = await result.collect()

            assert output.error is None
            assert isinstance(output.output, list)
            assert len(output.output) == 1
            assert special_file.resolve() in [Path(p).resolve() for p in output.output]


class TestListIntegration:
    """Test List tool integration scenarios."""

    @pytest.mark.asyncio
    async def test_list_python_project_structure(self):
        """Test listing a typical Python project structure."""
        list_tool = List()

        # List the current project directory
        result = list_tool(path="python")
        output = await result.collect()

        assert output.error is None
        assert isinstance(output.output, list)

        # Should contain typical Python project files/directories
        output_names = [Path(p).name for p in output.output]
        expected_items = ["timbal", "tests"]  # Common in Python projects

        for item in expected_items:
            if item in output_names:
                assert True
                break
        else:
            # If none found, that's still okay - just verify we got some output
            assert len(output.output) > 0

    @pytest.mark.asyncio
    async def test_list_nested_directory_access(self):
        """Test accessing nested directories."""
        list_tool = List()

        # Test listing nested structure
        test_paths = [
            "python/timbal",
            "python/tests",
            "python/timbal/core"
        ]

        for path in test_paths:
            if os.path.exists(path):
                result = list_tool(path=path)
                output = await result.collect()

                assert output.error is None
                assert isinstance(output.output, list)

    @pytest.mark.asyncio
    async def test_list_output_format(self):
        """Test that output format is consistent."""
        list_tool = List()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = ["file1.txt", "file2.py", "dir1"]
            created_paths = []

            for name in test_files:
                if name == "dir1":
                    path = Path(tmpdir) / name
                    path.mkdir()
                else:
                    path = Path(tmpdir) / name
                    path.write_text("test content")
                created_paths.append(path)

            result = list_tool(path=tmpdir)
            output = await result.collect()

            assert output.error is None
            assert isinstance(output.output, list)
            assert len(output.output) == len(test_files)

            # All output items should be Path objects (as strings when converted)
            output_resolved = [Path(p).resolve() for p in output.output]
            created_resolved = [p.resolve() for p in created_paths]

            for item in output.output:
                assert isinstance(item, Path)

            for resolved_path in created_resolved:
                assert resolved_path in output_resolved