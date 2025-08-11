from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pydantic import ValidationError
from timbal.server.fs.operations import operation_adapter
from timbal.server.fs.operations.base import BaseOperation


class TestFileSystemOperationValidation:
    """Test path validation for all filesystem operations."""
    
    def setup_method(self):
        """Set up test with temporary directory."""
        self.temp_dir = TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        BaseOperation._base_path = self.base_path
        
    def teardown_method(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        BaseOperation._base_path = None

    def test_read_operation_valid_path(self):
        """Test read operation with valid path."""
        data = {"type": "read", "path": "test.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "test.txt"
        
    def test_read_operation_invalid_path(self):
        """Test read operation with path traversal attempt."""
        data = {"type": "read", "path": "../etc/passwd"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_write_operation_valid_path(self):
        """Test write operation with valid path."""
        data = {"type": "write", "path": "test.txt", "content": "hello"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "test.txt"
        
    def test_write_operation_invalid_path(self):
        """Test write operation with path traversal attempt."""
        data = {"type": "write", "path": "../../tmp/evil.txt", "content": "evil"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_move_operation_valid_paths(self):
        """Test move operation with valid source and destination paths."""
        data = {"type": "move", "path": "old.txt", "new_path": "new.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "old.txt"
        assert operation.new_path == "new.txt"
        
    def test_move_operation_invalid_source_path(self):
        """Test move operation with invalid source path."""
        data = {"type": "move", "path": "../etc/passwd", "new_path": "new.txt"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)
        
    def test_move_operation_invalid_destination_path(self):
        """Test move operation with invalid destination path."""
        data = {"type": "move", "path": "old.txt", "new_path": "../../../tmp/evil.txt"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "New path outside allowed directory" in str(exc_info.value)

    def test_copy_operation_valid_paths(self):
        """Test copy operation with valid source and destination paths."""
        data = {"type": "copy", "path": "source.txt", "new_path": "dest.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "source.txt"
        assert operation.new_path == "dest.txt"
        
    def test_copy_operation_invalid_source_path(self):
        """Test copy operation with invalid source path."""
        data = {"type": "copy", "path": "../sensitive.txt", "new_path": "dest.txt"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)
        
    def test_copy_operation_invalid_destination_path(self):
        """Test copy operation with invalid destination path."""
        data = {"type": "copy", "path": "source.txt", "new_path": "../../../../tmp/leaked.txt"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "New path outside allowed directory" in str(exc_info.value)

    def test_delete_operation_valid_path(self):
        """Test delete operation with valid path."""
        data = {"type": "delete", "path": "to_delete.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "to_delete.txt"
        
    def test_delete_operation_invalid_path(self):
        """Test delete operation with path traversal attempt."""
        data = {"type": "delete", "path": "../important_file.txt"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_list_operation_valid_path(self):
        """Test list operation with valid path."""
        data = {"type": "list", "path": "subdir"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "subdir"
        
    def test_list_operation_invalid_path(self):
        """Test list operation with path traversal attempt."""
        data = {"type": "list", "path": "../../etc"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_mkdir_operation_valid_path(self):
        """Test mkdir operation with valid path."""
        data = {"type": "mkdir", "path": "new_directory"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "new_directory"
        
    def test_mkdir_operation_invalid_path(self):
        """Test mkdir operation with path traversal attempt."""
        data = {"type": "mkdir", "path": "../../../tmp/evil_dir"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_subdirectory_paths_allowed(self):
        """Test that paths within subdirectories are allowed."""
        data = {"type": "read", "path": "subdir/file.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "subdir/file.txt"
        
    def test_dot_paths_allowed(self):
        """Test that current directory references are allowed."""
        data = {"type": "read", "path": "./file.txt"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "./file.txt"

    def test_complex_traversal_blocked(self):
        """Test that complex path traversal attempts are blocked."""
        data = {"type": "read", "path": "subdir/../../../etc/passwd"}
        with pytest.raises(ValidationError) as exc_info:
            operation_adapter.validate_python(data)
        assert "Path outside allowed directory" in str(exc_info.value)

    def test_validation_without_base_path(self):
        """Test that validation is skipped when no base path is set."""
        BaseOperation._base_path = None
        
        # This should not raise an error
        data = {"type": "read", "path": "../etc/passwd"}
        operation = operation_adapter.validate_python(data)
        assert operation.path == "../etc/passwd"


class TestFileSystemOperationExecution:
    """Test actual execution of filesystem operations."""
    
    def setup_method(self):
        """Set up test with temporary directory and test files."""
        self.temp_dir = TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)
        BaseOperation._base_path = self.base_path
        
        # Create test files and directories
        (self.base_path / "test.txt").write_text("Hello World")
        (self.base_path / "subdir").mkdir()
        (self.base_path / "subdir" / "nested.txt").write_text("Nested content")
        
    def teardown_method(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        BaseOperation._base_path = None

    async def test_read_operation_execution(self):
        """Test reading an existing file."""
        data = {"type": "read", "path": "test.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        assert "Hello World" in result["content"]

    async def test_read_operation_nonexistent_file(self):
        """Test reading a non-existent file."""
        data = {"type": "read", "path": "nonexistent.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    async def test_write_operation_execution(self):
        """Test writing to a new file."""
        data = {"type": "write", "path": "new_file.txt", "content": "New content"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify file was created with correct content
        written_content = (self.base_path / "new_file.txt").read_text()
        assert written_content == "New content"

    async def test_write_operation_overwrite(self):
        """Test overwriting an existing file."""
        data = {"type": "write", "path": "test.txt", "content": "Overwritten"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify file was overwritten
        written_content = (self.base_path / "test.txt").read_text()
        assert written_content == "Overwritten"

    async def test_move_operation_execution(self):
        """Test moving a file."""
        data = {"type": "move", "path": "test.txt", "new_path": "moved.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify old file is gone and new file exists
        assert not (self.base_path / "test.txt").exists()
        assert (self.base_path / "moved.txt").exists()
        assert (self.base_path / "moved.txt").read_text() == "Hello World"

    async def test_move_operation_nonexistent_source(self):
        """Test moving a non-existent file."""
        data = {"type": "move", "path": "nonexistent.txt", "new_path": "moved.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    async def test_move_directory_execution(self):
        """Test moving a directory."""
        data = {"type": "move", "path": "subdir", "new_path": "moved_dir"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify old directory is gone and new directory exists with contents
        assert not (self.base_path / "subdir").exists()
        assert (self.base_path / "moved_dir").exists()
        assert (self.base_path / "moved_dir").is_dir()
        assert (self.base_path / "moved_dir" / "nested.txt").exists()
        assert (self.base_path / "moved_dir" / "nested.txt").read_text() == "Nested content"

    async def test_move_operation_destination_exists(self):
        """Test moving when destination already exists."""
        # Create destination file
        (self.base_path / "existing.txt").write_text("Already exists")
        
        data = {"type": "move", "path": "test.txt", "new_path": "existing.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "already exists" in result["error"].lower()

    async def test_copy_operation_execution(self):
        """Test copying a file."""
        data = {"type": "copy", "path": "test.txt", "new_path": "copied.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify both files exist with same content
        assert (self.base_path / "test.txt").exists()
        assert (self.base_path / "copied.txt").exists()
        assert (self.base_path / "test.txt").read_text() == (self.base_path / "copied.txt").read_text()

    async def test_copy_directory_execution(self):
        """Test copying a directory."""
        data = {"type": "copy", "path": "subdir", "new_path": "copied_dir"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify directory and its contents were copied
        assert (self.base_path / "subdir").exists()
        assert (self.base_path / "copied_dir").exists()
        assert (self.base_path / "copied_dir" / "nested.txt").exists()
        assert (self.base_path / "copied_dir" / "nested.txt").read_text() == "Nested content"

    async def test_copy_operation_nonexistent_source(self):
        """Test copying a non-existent file."""
        data = {"type": "copy", "path": "nonexistent.txt", "new_path": "copied.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    async def test_copy_operation_destination_exists(self):
        """Test copying when destination already exists."""
        # Create destination file
        (self.base_path / "existing.txt").write_text("Already exists")
        
        data = {"type": "copy", "path": "test.txt", "new_path": "existing.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "already exists" in result["error"].lower()

    async def test_delete_operation_execution(self):
        """Test deleting a file."""
        data = {"type": "delete", "path": "test.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify file is deleted
        assert not (self.base_path / "test.txt").exists()

    async def test_delete_directory_execution(self):
        """Test deleting a directory with contents (recursive delete)."""
        data = {"type": "delete", "path": "subdir"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify directory and its contents are deleted
        assert not (self.base_path / "subdir").exists()

    async def test_delete_operation_nonexistent_path(self):
        """Test deleting a non-existent file."""
        data = {"type": "delete", "path": "nonexistent.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "not an existing file or directory" in result["error"].lower()

    async def test_list_operation_execution(self):
        """Test listing directory contents."""
        data = {"type": "list", "path": "."}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        assert "contents" in result
        
        # Check that our test files are listed
        file_names = [item["name"] for item in result["contents"]]
        assert "test.txt" in file_names
        assert "subdir" in file_names

    async def test_list_nonexistent_directory(self):
        """Test listing a non-existent directory."""
        data = {"type": "list", "path": "nonexistent_dir"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "does not exist" in result["error"].lower()

    async def test_mkdir_operation_execution(self):
        """Test creating a directory."""
        data = {"type": "mkdir", "path": "new_directory"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify directory was created
        assert (self.base_path / "new_directory").exists()
        assert (self.base_path / "new_directory").is_dir()

    async def test_mkdir_nested_directory(self):
        """Test creating nested directories."""
        data = {"type": "mkdir", "path": "deep/nested/directory"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify nested directory was created
        assert (self.base_path / "deep" / "nested" / "directory").exists()
        assert (self.base_path / "deep" / "nested" / "directory").is_dir()

    async def test_mkdir_existing_directory(self):
        """Test creating a directory that already exists."""
        data = {"type": "mkdir", "path": "subdir"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "already exists" in result["error"].lower()

    async def test_list_operation_on_file(self):
        """Test listing a file instead of a directory."""
        data = {"type": "list", "path": "test.txt"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert "error" in result
        assert "not a directory" in result["error"].lower()

    async def test_write_operation_in_nested_path(self):
        """Test writing to a file in a nested directory that doesn't exist."""
        data = {"type": "write", "path": "deep/nested/file.txt", "content": "Deep content"}
        operation = operation_adapter.validate_python(data)
        
        result = await operation(self.base_path)
        assert result["success"] is True
        
        # Verify file was created with parent directories
        assert (self.base_path / "deep" / "nested" / "file.txt").exists()
        assert (self.base_path / "deep" / "nested" / "file.txt").read_text() == "Deep content"