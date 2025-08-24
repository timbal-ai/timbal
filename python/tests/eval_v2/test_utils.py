from pathlib import Path

from timbal.eval_v2.utils import discover_files


class TestDiscoverFiles:
    """Test the discover_files utility function."""

    def test_discover_files_single_file(self, simple_test_file):
        """Test discovering a single eval file."""
        files = discover_files(simple_test_file)
        
        assert len(files) == 1
        assert files[0] == simple_test_file

    def test_discover_files_directory_with_eval_files(self, fixtures_dir):
        """Test discovering eval files in a directory."""
        files = discover_files(fixtures_dir)
        
        # Should find all eval_*.yaml files in fixtures directory
        for f in files:
            file_name = f.name
            assert f.exists()
            assert file_name.startswith("eval_") and file_name.endswith(".yaml")
        
    def test_discover_files_nonexistent_path(self):
        """Test behavior with nonexistent path."""
        nonexistent_path = Path("/nonexistent/path")
        
        files = discover_files(nonexistent_path)
        assert files == []