"""CLI startup laziness: libcst must only load for transformer operations.

The __main__ op table lets --help and lightweight ops (get-flow, get-tools,
get-models, test, evals) dispatch without importing the transformer modules
(each imports libcst at module level, ~160ms of CLI startup).
"""

import subprocess
import sys

from timbal.codegen.__main__ import _TRANSFORMER_OPS, _requested_operation


class TestOpTableSync:
    def test_table_matches_modules_on_disk(self):
        """The static table must list exactly the transformer modules."""
        from timbal.codegen.transformers import load_modules

        on_disk = set(load_modules().keys())
        in_table = {module for module, _ in _TRANSFORMER_OPS.values()}
        assert in_table == on_disk

    def test_cli_names_map_to_module_names(self):
        for cli_name, (module_name, help_line) in _TRANSFORMER_OPS.items():
            assert cli_name.replace("-", "_") == module_name
            assert help_line


class TestRequestedOperation:
    def test_simple(self):
        assert _requested_operation(["add-mcp", "--name", "x"]) == "add-mcp"

    def test_global_flag_values_are_not_operations(self):
        assert _requested_operation(["--path", "some-dir", "add-mcp"]) == "add-mcp"
        assert _requested_operation(["--path=some-dir", "add-tool"]) == "add-tool"

    def test_no_operation(self):
        assert _requested_operation(["--help"]) is None
        assert _requested_operation([]) is None


class TestHelpDoesNotImportLibcst:
    def test_top_level_help_lists_ops_without_libcst(self):
        code = (
            "import sys\n"
            "sys.argv = ['timbal-codegen', '--help']\n"
            "from timbal.codegen.__main__ import main\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('LIBCST_IMPORTED', 'libcst' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        # Help must still list every transformer op (stub parsers).
        for op in _TRANSFORMER_OPS:
            assert op in result.stdout
        assert "LIBCST_IMPORTED False" in result.stdout

    def test_transformer_subcommand_help_registers_full_parser(self):
        code = (
            "import sys\n"
            "sys.argv = ['timbal-codegen', 'add-mcp', '--help']\n"
            "from timbal.codegen.__main__ import main\n"
            "try:\n"
            "    main()\n"
            "except SystemExit:\n"
            "    pass\n"
            "print('LIBCST_IMPORTED', 'libcst' in sys.modules)\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        # Full parser for the requested op: its specific flags must show.
        assert "--transport" in result.stdout or "--name" in result.stdout
        assert "LIBCST_IMPORTED True" in result.stdout
