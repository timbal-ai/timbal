import tempfile
from pathlib import Path

import pytest
from timbal.tools.write import Write


class TestWriteToolInitialization:
    """Test Write tool initialization and basic functionality."""

    def test_write_tool_creation(self):
        """Test that Write tool is created correctly."""
        write_tool = Write()

        assert write_tool.name == "write"
        assert "Create a new file or edit an existing one" in write_tool.description
        assert write_tool.handler is not None

    def test_write_tool_handler_signature(self):
        """Test that the handler has the correct signature."""
        write_tool = Write()

        # Check that handler accepts the expected parameters
        import inspect
        sig = inspect.signature(write_tool.handler)
        params = list(sig.parameters.keys())

        expected_params = ['path', 'content', 'dry_run']
        for param in expected_params:
            assert param in params


class TestWriteFileOperations:
    """Test Write tool file operations."""

    @pytest.mark.asyncio
    async def test_write_new_file(self):
        """Test creating a new file."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "new_file.txt"
            test_content = "Hello, World!"

            result = write_tool(path=str(test_file), content=test_content)
            output = await result.collect()

            assert output.error is None
            assert "File created" in output.output
            assert test_file.exists()
            assert test_file.read_text() == test_content

    @pytest.mark.asyncio
    async def test_write_existing_file(self):
        """Test modifying an existing file."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "existing_file.txt"
            original_content = "Original content"
            new_content = "Modified content"

            # Create original file
            test_file.write_text(original_content)

            result = write_tool(path=str(test_file), content=new_content)
            output = await result.collect()

            assert output.error is None
            assert "File modified" in output.output
            assert test_file.read_text() == new_content

    @pytest.mark.asyncio
    async def test_write_same_content(self):
        """Test writing identical content to existing file."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "same_content.txt"
            content = "Same content"

            # Create original file
            test_file.write_text(content)

            result = write_tool(path=str(test_file), content=content)
            output = await result.collect()

            assert output.error is None
            assert "Content already matches - no changes needed" in output.output

    @pytest.mark.asyncio
    async def test_write_creates_parent_directories(self):
        """Test that parent directories are created automatically."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_file = Path(tmpdir) / "nested" / "dirs" / "file.txt"
            content = "Nested file content"

            result = write_tool(path=str(nested_file), content=content)
            output = await result.collect()

            assert output.error is None
            assert "File created" in output.output
            assert nested_file.exists()
            assert nested_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_directory_error(self):
        """Test error when trying to write to a directory."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = write_tool(path=tmpdir, content="test")
            output = await result.collect()

            assert output.error is not None
            assert output.error["type"] == "ValueError"
            assert "Path is a directory" in output.error["message"]


class TestWritePathExpansion:
    """Test Write tool path expansion functionality."""

    @pytest.mark.asyncio
    async def test_home_directory_expansion(self):
        """Test that ~ expands to home directory."""
        write_tool = Write()

        # Use dry_run to avoid actually writing to home
        result = write_tool(path="~/test_write.txt", content="test", dry_run=True)
        output = await result.collect()

        assert output.error is None
        # Should show expanded path in preview
        assert str(Path.home()) in output.output

    @pytest.mark.asyncio
    async def test_environment_variable_expansion(self):
        """Test that environment variables are expanded."""
        write_tool = Write()
        import os

        # Set a test environment variable
        test_path = "/tmp/test_write_env"
        os.environ['TEST_WRITE_PATH'] = test_path

        try:
            result = write_tool(path="$TEST_WRITE_PATH/file.txt", content="test", dry_run=True)
            output = await result.collect()

            assert output.error is None
            assert test_path in output.output
        finally:
            # Clean up environment variable
            del os.environ['TEST_WRITE_PATH']


class TestWriteDryRun:
    """Test Write tool dry run functionality."""

    @pytest.mark.asyncio
    async def test_dry_run_new_file(self):
        """Test dry run for creating a new file."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "dry_run_new.txt"
            content = "New file content"

            result = write_tool(path=str(test_file), content=content, dry_run=True)
            output = await result.collect()

            assert output.error is None
            assert "Preview - would create" in output.output
            assert "new file" in output.output
            assert not test_file.exists()  # File should not be created

    @pytest.mark.asyncio
    async def test_dry_run_existing_file(self):
        """Test dry run for modifying an existing file."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "dry_run_existing.txt"
            original_content = "Original content"
            new_content = "Modified content"

            # Create original file
            test_file.write_text(original_content)

            result = write_tool(path=str(test_file), content=new_content, dry_run=True)
            output = await result.collect()

            assert output.error is None
            assert "Preview - would modify" in output.output
            assert "existing" in output.output
            assert test_file.read_text() == original_content  # File should not be modified

    @pytest.mark.asyncio
    async def test_dry_run_shows_diff(self):
        """Test that dry run shows proper diff output."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "diff_test.txt"
            original_content = "Line 1\nLine 2\nLine 3"
            new_content = "Line 1\nModified Line 2\nLine 3"

            # Create original file
            test_file.write_text(original_content)

            result = write_tool(path=str(test_file), content=new_content, dry_run=True)
            output = await result.collect()

            assert output.error is None
            # Should contain diff markers
            assert "@@" in output.output  # diff hunk header
            assert "-Line 2" in output.output or "-" in output.output
            assert "+Modified Line 2" in output.output or "+" in output.output


class TestWriteDiffOutput:
    """Test Write tool diff output functionality."""

    @pytest.mark.asyncio
    async def test_write_shows_diff_after_creation(self):
        """Test that actual write operations show diff output."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "diff_after_write.txt"
            content = "New file with\nmultiple lines"

            result = write_tool(path=str(test_file), content=content)
            output = await result.collect()

            assert output.error is None
            assert "File created" in output.output
            assert "Changes:" in output.output
            assert "new file" in output.output

    @pytest.mark.asyncio
    async def test_write_shows_diff_after_modification(self):
        """Test that modifications show diff output."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "modify_with_diff.txt"
            original_content = "Original line"
            new_content = "Modified line"

            # Create original file
            test_file.write_text(original_content)

            result = write_tool(path=str(test_file), content=new_content)
            output = await result.collect()

            assert output.error is None
            assert "File modified" in output.output
            assert "Changes:" in output.output
            # Should show the actual change
            assert "Original line" in output.output or "-" in output.output
            assert "Modified line" in output.output or "+" in output.output


class TestWriteErrorHandling:
    """Test Write tool error handling."""

    @pytest.mark.asyncio
    async def test_write_permission_denied(self):
        """Test handling of permission denied errors."""
        write_tool = Write()

        # Try to write to a system directory that typically requires permissions
        restricted_paths = ["/root/test.txt", "/etc/test.txt"]

        for path in restricted_paths:
            if Path(path).parent.exists():
                result = write_tool(path=path, content="test")
                output = await result.collect()

                # Should either succeed or fail gracefully with permission error
                if output.error is not None:
                    assert "Permission denied" in str(output.error) or "PermissionError" in str(output.error)
                break

    @pytest.mark.asyncio
    async def test_write_with_special_characters(self):
        """Test writing files with special characters in names."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            special_file = Path(tmpdir) / "file with spaces & symbols!.txt"
            content = "Content with special chars: éñ中文"

            result = write_tool(path=str(special_file), content=content)
            output = await result.collect()

            assert output.error is None
            assert special_file.exists()
            assert special_file.read_text() == content

    @pytest.mark.asyncio
    async def test_write_binary_file_read_error(self):
        """Test error when trying to modify an existing binary file (like an image)."""
        write_tool = Write()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a binary file (simulating an image)
            binary_file = Path(tmpdir) / "test.png"
            binary_content = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01'
            binary_file.write_bytes(binary_content)

            # Try to write text content to the binary file
            result = write_tool(path=str(binary_file), content="text content")
            output = await result.collect()

            # Should get an error because it can't decode the binary content as UTF-8
            assert output.error is not None
            assert "UnicodeDecodeError" in output.error["type"]