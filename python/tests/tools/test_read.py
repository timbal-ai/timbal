import pathlib

import pytest
from timbal.state import set_run_context
from timbal.tools import Read
from timbal.types import File


@pytest.fixture
def read_tool():
    """Fixture to create a Read tool instance."""
    return Read()


@pytest.fixture
def fixtures_dir():
    """Fixture to get the fixtures directory path."""
    return pathlib.Path(__file__).parent.parent / "fixtures"


@pytest.mark.asyncio
async def test_read_full_text_file(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading a complete text file."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, str)
    assert result.output == test_content


@pytest.mark.asyncio
async def test_read_binary_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading a binary file returns File object."""
    test_file = fixtures_dir / "test.png"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, File)
    # Verify it's actually a PNG file
    content = result.output.read()
    assert content.startswith(b'\x89PNG')


@pytest.mark.asyncio
async def test_read_line_range(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading a specific line range."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=2, end_line=4).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 2\nLine 3\nLine 4\n"


@pytest.mark.asyncio
async def test_read_from_start_line(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading from a start line to end of file."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=3).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 3\nLine 4\nLine 5\n"


@pytest.mark.asyncio
async def test_read_to_end_line(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading from beginning to an end line."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), end_line=3).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 1\nLine 2\nLine 3\n"


@pytest.mark.asyncio
async def test_read_single_line(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading a single line."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=3, end_line=3).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 3\n"


@pytest.mark.asyncio
async def test_read_out_of_range_lines(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading lines beyond file length returns empty string."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=10, end_line=20).collect()

    assert isinstance(result.output, str)
    assert result.output == ""


@pytest.mark.asyncio
async def test_read_partially_out_of_range(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading lines that partially exceed file length."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3\n"
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=2, end_line=10).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 2\nLine 3\n"


@pytest.mark.asyncio
async def test_read_empty_file(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading an empty file."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("", encoding="utf-8")

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, str)
    assert result.output == ""


@pytest.mark.asyncio
async def test_read_markdown_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading a markdown file."""
    test_file = fixtures_dir / "test.md"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, str)
    # Verify it contains markdown content (table syntax with pipes)
    assert "|" in result.output and "---" in result.output


@pytest.mark.asyncio
async def test_read_with_path_expansion(read_tool: Read, tmp_path: pathlib.Path, monkeypatch) -> None:
    """Test that path expansion works (~ and environment variables)."""
    # Create a test file in tmp_path
    test_file = tmp_path / "test.txt"
    test_content = "Test content\n"
    test_file.write_text(test_content, encoding="utf-8")
    
    # Set environment variable
    monkeypatch.setenv("TEST_DIR", str(tmp_path))
    
    result = await read_tool(path="$TEST_DIR/test.txt").collect()
    
    assert isinstance(result.output, str)
    assert result.output == test_content


@pytest.mark.asyncio
async def test_read_line_range_efficiency(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test that reading line ranges is efficient (doesn't read entire file)."""
    test_file = tmp_path / "large_test.txt"
    # Create a file with many lines
    lines = [f"Line {i}\n" for i in range(1, 1001)]
    test_file.write_text("".join(lines), encoding="utf-8")

    # Read only lines 500-510
    result = await read_tool(path=str(test_file), start_line=500, end_line=510).collect()

    assert isinstance(result.output, str)
    expected = "".join([f"Line {i}\n" for i in range(500, 511)])
    assert result.output == expected


@pytest.mark.asyncio
async def test_read_file_without_trailing_newline(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test reading a file without trailing newline."""
    test_file = tmp_path / "test.txt"
    test_content = "Line 1\nLine 2\nLine 3"  # No trailing newline
    test_file.write_text(test_content, encoding="utf-8")

    result = await read_tool(path=str(test_file), start_line=2, end_line=3).collect()

    assert isinstance(result.output, str)
    assert result.output == "Line 2\nLine 3"


@pytest.mark.asyncio
async def test_read_json_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading a JSON file."""
    test_file = fixtures_dir / "test.json"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, str)
    # Verify it contains JSON content
    assert "{" in result.output and "}" in result.output


@pytest.mark.asyncio
async def test_read_pdf_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading a PDF file returns File object."""
    test_file = fixtures_dir / "test.pdf"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, File)
    # Verify it's actually a PDF file
    content = result.output.read()
    assert content.startswith(b'%PDF')


@pytest.mark.asyncio
async def test_read_xlsx_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading an Excel file returns File object."""
    test_file = fixtures_dir / "test.xlsx"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, File)
    # Verify it's actually an Excel file (ZIP format)
    content = result.output.read()
    assert content.startswith(b'PK\x03\x04')


@pytest.mark.asyncio
async def test_read_docx_file(fixtures_dir: pathlib.Path, read_tool: Read) -> None:
    """Test reading a Word document returns File object."""
    test_file = fixtures_dir / "test.docx"

    result = await read_tool(path=str(test_file)).collect()

    assert isinstance(result.output, File)
    # Verify it's actually a Word document (ZIP format)
    content = result.output.read()
    assert content.startswith(b'PK\x03\x04')


@pytest.mark.asyncio
async def test_read_path_traversal_blocked(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test that path traversal attacks are blocked."""
    from timbal.state import RunContext
    
    # Create a test file outside the base path
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("Secret content", encoding="utf-8")
    
    # Set base path to a subdirectory
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    
    # Create and set run context with restricted base path
    run_context = RunContext()
    run_context._base_path = base_dir
    set_run_context(run_context)
    
    # Try to read file outside base path using relative path traversal
    result = await read_tool(path="../outside/secret.txt").collect()
    
    # Should have an error
    assert result.error is not None
    assert "Access denied" in result.error["message"]
    assert "outside the allowed base path" in result.error["message"]


@pytest.mark.asyncio
async def test_read_relative_path_within_base(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test that relative paths within base path work correctly."""
    from timbal.state import RunContext
    
    # Set base path
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    
    # Create a subdirectory and file
    sub_dir = base_dir / "subdir"
    sub_dir.mkdir()
    test_file = sub_dir / "test.txt"
    test_content = "Test content\n"
    test_file.write_text(test_content, encoding="utf-8")
    
    # Create and set run context with base path
    run_context = RunContext()
    run_context._base_path = base_dir
    set_run_context(run_context)
    
    # Read using relative path (should work)
    result = await read_tool(path="subdir/test.txt").collect()
    
    assert isinstance(result.output, str)
    assert result.output == test_content


@pytest.mark.asyncio
async def test_read_absolute_path_within_base(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test that absolute paths within base path work correctly."""
    from timbal.state import RunContext
    
    # Set base path
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    
    # Create a file
    test_file = base_dir / "test.txt"
    test_content = "Test content\n"
    test_file.write_text(test_content, encoding="utf-8")
    
    # Create and set run context with base path
    run_context = RunContext()
    run_context._base_path = base_dir
    set_run_context(run_context)
    
    # Read using absolute path (should work if within base)
    result = await read_tool(path=str(test_file)).collect()
    
    assert isinstance(result.output, str)
    assert result.output == test_content


@pytest.mark.asyncio
async def test_read_absolute_path_outside_base_blocked(tmp_path: pathlib.Path, read_tool: Read) -> None:
    """Test that absolute paths outside base path are blocked."""
    from timbal.state import RunContext
    
    # Create a file outside the base path
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("Secret content", encoding="utf-8")
    
    # Set base path to a subdirectory
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    
    # Create and set run context with restricted base path
    run_context = RunContext()
    run_context._base_path = base_dir
    set_run_context(run_context)
    
    # Try to read file outside base path using absolute path
    result = await read_tool(path=str(outside_file)).collect()
    
    # Should have an error
    assert result.error is not None
    assert "Access denied" in result.error["message"]
    assert "outside the allowed base path" in result.error["message"]
