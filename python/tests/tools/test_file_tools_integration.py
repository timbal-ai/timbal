"""
Integration tests for Read, Write, and Edit tools with RunContext session state tracking.

Tests the complete workflow of file operations with state tracking to ensure:
- Files must be read before editing
- Files cannot be edited if modified externally
- State tracking works correctly across read/write/edit operations
- Error handling for various edge cases

Every test creates its own fresh RunContext and tool instances.
No fixtures. No shared state. Complete isolation.

The fs_state is stored in RunContext._session_state["fs_state"] which is automatically
preserved when child RunContexts are created during nested runnable execution.
"""

import hashlib
from pathlib import Path

import pytest
from timbal.state import RunContext, set_parent_call_id, set_run_context
from timbal.tools.edit import Edit
from timbal.tools.read import Read
from timbal.tools.write import Write


def get_fs_state(ctx: RunContext) -> dict:
    """Helper to get fs_state from session state."""
    return ctx._session_state.get("fs_state", {})


class TestReadToolWithFsState:
    """Test Read tool with fs_state tracking."""

    @pytest.mark.asyncio
    async def test_read_updates_fs_state(self, tmp_path: Path):
        """Test that reading a file updates the fs_state."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_read_updates_fs_state")

        test_file = tmp_path / "test.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content, encoding="utf-8")

        # Create fresh tool
        read_tool = Read()
        result = await read_tool(path=str(test_file)).collect()

        assert result.error is None
        assert result.output == test_content
        fs_state = get_fs_state(ctx)
        assert str(test_file) in fs_state
        expected_hash = hashlib.sha256(test_content.encode("utf-8")).hexdigest()
        assert fs_state[str(test_file)] == expected_hash

    @pytest.mark.asyncio
    async def test_read_multiple_files_tracks_all(self, tmp_path: Path):
        """Test that reading multiple files tracks all of them."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_read_multiple_files_tracks_all")

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1", encoding="utf-8")
        file2.write_text("Content 2", encoding="utf-8")

        # Create fresh tool
        read_tool = Read()
        await read_tool(path=str(file1)).collect()
        await read_tool(path=str(file2)).collect()

        fs_state = get_fs_state(ctx)
        assert str(file1) in fs_state
        assert str(file2) in fs_state

    @pytest.mark.asyncio
    async def test_read_updates_hash_on_reread(self, tmp_path: Path):
        """Test that re-reading a modified file updates the hash."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_read_updates_hash_on_reread")

        test_file = tmp_path / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content, encoding="utf-8")

        # Create fresh tool
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()
        original_hash = get_fs_state(ctx)[str(test_file)]

        new_content = "Modified content"
        test_file.write_text(new_content, encoding="utf-8")

        await read_tool(path=str(test_file)).collect()
        new_hash = get_fs_state(ctx)[str(test_file)]

        expected_new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        assert new_hash != original_hash
        assert new_hash == expected_new_hash


class TestEditToolWithFsState:
    """Test Edit tool with fs_state tracking."""

    @pytest.mark.asyncio
    async def test_edit_requires_prior_read(self, tmp_path: Path):
        """Test that editing a file requires it to be read first."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_requires_prior_read")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content", encoding="utf-8")

        # Read a different file to populate fs_state (so the check is triggered)
        other_file = tmp_path / "other.txt"
        other_file.write_text("Other content", encoding="utf-8")
        read_tool = Read()
        await read_tool(path=str(other_file)).collect()

        # Create fresh tool
        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        assert result.error is not None
        assert result.error["type"] == "FileNotReadError"
        assert "has not been read" in result.error["message"]

    @pytest.mark.asyncio
    async def test_edit_after_read_succeeds(self, tmp_path: Path):
        """Test that editing after reading succeeds."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_after_read_succeeds")

        test_file = tmp_path / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content, encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        assert result.error is None
        assert "Modified content" == test_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_edit_detects_external_modification(self, tmp_path: Path):
        """Test that edit detects if file was modified externally after read."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_detects_external_modification")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        # Modify file externally
        test_file.write_text("Original content but externally modified", encoding="utf-8")

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        assert result.error is not None
        assert result.error["type"] == "FileModifiedError"
        assert "has been modified since you last read it" in result.error["message"]

    @pytest.mark.asyncio
    async def test_edit_updates_fs_state(self, tmp_path: Path):
        """Test that successful edit updates the fs_state."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_updates_fs_state")

        test_file = tmp_path / "test.txt"
        original_content = "Original content"
        test_file.write_text(original_content, encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()
        original_hash = get_fs_state(ctx)[str(test_file)]

        edit_tool = Edit()
        await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        new_hash = get_fs_state(ctx)[str(test_file)]
        new_content = "Modified content"
        expected_new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        assert new_hash != original_hash
        assert new_hash == expected_new_hash

    @pytest.mark.asyncio
    async def test_multiple_edits_in_sequence(self, tmp_path: Path):
        """Test multiple sequential edits work correctly."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_multiple_edits_in_sequence")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result1 = await edit_tool(path=str(test_file), old_string="Line 1", new_string="Modified Line 1").collect()
        assert result1.error is None

        result2 = await edit_tool(path=str(test_file), old_string="Line 2", new_string="Modified Line 2").collect()
        assert result2.error is None

        final_content = test_file.read_text(encoding="utf-8")
        assert "Modified Line 1" in final_content
        assert "Modified Line 2" in final_content

    @pytest.mark.asyncio
    async def test_edit_identical_strings_fails(self, tmp_path: Path):
        """Test that editing with identical old and new strings fails."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_identical_strings_fails")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Same content", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Same", new_string="Same").collect()

        assert result.error is not None
        assert result.error["type"] == "ValueError"
        assert "identical" in result.error["message"].lower()


class TestWriteToolWithFsState:
    """Test Write tool with fs_state tracking."""

    @pytest.mark.asyncio
    async def test_write_new_file_updates_fs_state(self, tmp_path: Path):
        """Test that writing a new file updates fs_state."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_new_file_updates_fs_state")

        test_file = tmp_path / "new_file.txt"
        content = "New file content"

        # Create fresh tool
        write_tool = Write()
        result = await write_tool(path=str(test_file), content=content).collect()

        assert result.error is None
        assert test_file.exists()
        fs_state = get_fs_state(ctx)
        assert str(test_file) in fs_state
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert fs_state[str(test_file)] == expected_hash

    @pytest.mark.asyncio
    async def test_write_overwrites_and_updates_fs_state(self, tmp_path: Path):
        """Test that overwriting a file updates fs_state."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_overwrites_and_updates_fs_state")

        test_file = tmp_path / "test.txt"
        original_content = "Original"
        test_file.write_text(original_content, encoding="utf-8")

        # Create fresh tool
        write_tool = Write()
        await write_tool(path=str(test_file), content=original_content).collect()
        original_hash = get_fs_state(ctx)[str(test_file)]

        new_content = "New content"
        await write_tool(path=str(test_file), content=new_content).collect()

        new_hash = get_fs_state(ctx)[str(test_file)]
        expected_new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
        assert new_hash != original_hash
        assert new_hash == expected_new_hash

    @pytest.mark.asyncio
    async def test_write_then_edit_workflow(self, tmp_path: Path):
        """Test write followed by edit works correctly."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_then_edit_workflow")

        test_file = tmp_path / "test.txt"
        initial_content = "Initial content"

        # Create fresh tools
        write_tool = Write()
        await write_tool(path=str(test_file), content=initial_content).collect()

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Initial", new_string="Modified").collect()

        assert result.error is None
        assert "Modified content" == test_file.read_text(encoding="utf-8")


class TestComplexWorkflows:
    """Test complex workflows combining read, write, and edit."""

    @pytest.mark.asyncio
    async def test_read_edit_read_workflow(self, tmp_path: Path):
        """Test read -> edit -> read workflow."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_read_edit_read_workflow")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        result1 = await read_tool(path=str(test_file)).collect()
        assert result1.output == "Original content"

        edit_tool = Edit()
        await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        result2 = await read_tool(path=str(test_file)).collect()
        assert result2.output == "Modified content"

    @pytest.mark.asyncio
    async def test_write_read_edit_workflow(self, tmp_path: Path):
        """Test write -> read -> edit workflow."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_read_edit_workflow")

        test_file = tmp_path / "test.txt"

        # Create fresh tools
        write_tool = Write()
        await write_tool(path=str(test_file), content="Initial content").collect()

        read_tool = Read()
        result = await read_tool(path=str(test_file)).collect()
        assert result.output == "Initial content"

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Initial", new_string="Updated").collect()
        assert result.error is None

    @pytest.mark.asyncio
    async def test_multiple_files_independent_tracking(self, tmp_path: Path):
        """Test that multiple files are tracked independently."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_multiple_files_independent_tracking")

        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("Content 1", encoding="utf-8")
        file2.write_text("Content 2", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(file1)).collect()

        edit_tool = Edit()
        result1 = await edit_tool(path=str(file1), old_string="Content 1", new_string="Modified 1").collect()
        assert result1.error is None

        result2 = await edit_tool(path=str(file2), old_string="Content 2", new_string="Modified 2").collect()
        assert result2.error is not None
        assert result2.error["type"] == "FileNotReadError"
        assert "has not been read" in result2.error["message"]

    @pytest.mark.asyncio
    async def test_edit_after_external_modification_then_reread(self, tmp_path: Path):
        """Test that re-reading after external modification allows edit."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_after_external_modification_then_reread")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        # External modification
        test_file.write_text("Original content but externally modified", encoding="utf-8")

        edit_tool = Edit()
        result1 = await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()
        assert result1.error is not None
        assert result1.error["type"] == "FileModifiedError"

        # Re-read to update fs_state
        await read_tool(path=str(test_file)).collect()

        result2 = await edit_tool(
            path=str(test_file),
            old_string="Original content but externally modified",
            new_string="Now properly modified",
        ).collect()
        assert result2.error is None

    @pytest.mark.asyncio
    async def test_write_overwrites_without_read_requirement(self, tmp_path: Path):
        """Test that write can overwrite without prior read."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_overwrites_without_read_requirement")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original", encoding="utf-8")

        # Create fresh tool
        write_tool = Write()
        result = await write_tool(path=str(test_file), content="Overwritten").collect()
        assert result.error is None
        assert test_file.read_text(encoding="utf-8") == "Overwritten"

    @pytest.mark.asyncio
    async def test_nested_directory_creation_and_tracking(self, tmp_path: Path):
        """Test that writing to nested directories works with fs_state."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_nested_directory_creation_and_tracking")

        nested_file = tmp_path / "dir1" / "dir2" / "file.txt"
        content = "Nested content"

        # Create fresh tools
        write_tool = Write()
        result = await write_tool(path=str(nested_file), content=content).collect()
        assert result.error is None
        assert nested_file.exists()
        fs_state = get_fs_state(ctx)
        assert str(nested_file) in fs_state

        edit_tool = Edit()
        result = await edit_tool(path=str(nested_file), old_string="Nested", new_string="Modified nested").collect()
        assert result.error is None


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_edit_nonexistent_file(self, tmp_path: Path):
        """Test editing a file that doesn't exist."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_nonexistent_file")

        test_file = tmp_path / "nonexistent.txt"

        # Create fresh tool
        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="old", new_string="new").collect()

        assert result.error is not None
        assert result.error["type"] == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_edit_string_not_found(self, tmp_path: Path):
        """Test editing with a string that doesn't exist in file."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_string_not_found")

        test_file = tmp_path / "test.txt"
        test_file.write_text("Some content", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="NonexistentString", new_string="new").collect()

        assert result.error is not None
        assert result.error["type"] == "ValueError"
        assert "not found" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_edit_directory_fails(self, tmp_path: Path):
        """Test that trying to edit a directory fails."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_edit_directory_fails")

        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        # Create fresh tool
        edit_tool = Edit()
        result = await edit_tool(path=str(test_dir), old_string="old", new_string="new").collect()

        assert result.error is not None
        assert result.error["type"] == "ValueError"
        assert "directory" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_write_to_directory_fails(self, tmp_path: Path):
        """Test that trying to write to a directory fails."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_write_to_directory_fails")

        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        # Create fresh tool
        write_tool = Write()
        result = await write_tool(path=str(test_dir), content="test").collect()

        assert result.error is not None
        assert result.error["type"] == "ValueError"
        assert "directory" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_empty_file_operations(self, tmp_path: Path):
        """Test operations on empty files."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_empty_file_operations")

        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        result = await read_tool(path=str(test_file)).collect()
        assert result.output == ""

        write_tool = Write()
        result = await write_tool(path=str(test_file), content="Now has content").collect()
        assert result.error is None

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Now has", new_string="Still has").collect()
        assert result.error is None

    @pytest.mark.asyncio
    async def test_unicode_content_handling(self, tmp_path: Path):
        """Test handling of unicode content in files."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_unicode_content_handling")

        test_file = tmp_path / "unicode.txt"
        unicode_content = "Hello 世界 🌍 Привет"
        test_file.write_text(unicode_content, encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        result = await read_tool(path=str(test_file)).collect()
        assert result.output == unicode_content

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="世界", new_string="World").collect()
        assert result.error is None
        assert "Hello World 🌍 Привет" == test_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_replace_all_flag(self, tmp_path: Path):
        """Test edit with replace_all flag."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_replace_all_flag")

        test_file = tmp_path / "test.txt"
        content = "foo bar foo baz foo"
        test_file.write_text(content, encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result = await edit_tool(
            path=str(test_file), old_string="foo", new_string="replaced", replace_all=True
        ).collect()

        assert result.error is None
        final_content = test_file.read_text(encoding="utf-8")
        assert final_content == "replaced bar replaced baz replaced"
        assert "foo" not in final_content

    @pytest.mark.asyncio
    async def test_replace_first_occurrence_only(self, tmp_path: Path):
        """Test edit replaces only first occurrence by default."""
        # Create fresh context
        ctx = RunContext()
        set_run_context(ctx)
        set_parent_call_id("test_replace_first_occurrence_only")

        test_file = tmp_path / "test.txt"
        content = "foo bar foo baz foo"
        test_file.write_text(content, encoding="utf-8")

        # Create fresh tools
        read_tool = Read()
        await read_tool(path=str(test_file)).collect()

        edit_tool = Edit()
        result = await edit_tool(
            path=str(test_file), old_string="foo", new_string="replaced", replace_all=False
        ).collect()

        assert result.error is None
        final_content = test_file.read_text(encoding="utf-8")
        assert final_content == "replaced bar foo baz foo"


class TestWithoutRunContext:
    """Test that tools work without RunContext (no fs_state tracking)."""

    @pytest.mark.asyncio
    async def test_edit_without_run_context_works(self, tmp_path: Path):
        """Test that edit works without RunContext (bypasses fs_state checks)."""
        set_run_context(None)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Original content", encoding="utf-8")

        # Create fresh tool
        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Original", new_string="Modified").collect()

        assert result.error is None
        assert "Modified content" == test_file.read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_all_tools_work_without_run_context(self, tmp_path: Path):
        """Test that all tools work without RunContext."""
        set_run_context(None)

        test_file = tmp_path / "test.txt"
        content = "Test content"

        # Create fresh tools
        write_tool = Write()
        result = await write_tool(path=str(test_file), content=content).collect()
        assert result.error is None

        read_tool = Read()
        result = await read_tool(path=str(test_file)).collect()
        assert result.error is None
        assert result.output == content

        edit_tool = Edit()
        result = await edit_tool(path=str(test_file), old_string="Test", new_string="Modified").collect()
        assert result.error is None
