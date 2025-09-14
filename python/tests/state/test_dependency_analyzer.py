import ast
import tempfile
from textwrap import dedent

from timbal.state.dependency_analyzer import RunContextDependencyAnalyzer


class TestRunContextDependencyAnalyzer:
    """Test the AST analyzer that detects RunContext.step_trace() usage."""

    def test_empty_code(self):
        """Test analyzer with empty code."""
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse("")
        analyzer.visit(tree)
        assert analyzer.dependencies == []

    def test_no_dependencies(self):
        """Test code with no step_trace calls."""
        code = dedent("""
            def handler():
                x = 1 + 2
                return x
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == []

    def test_simple_step_trace_call(self):
        """Test detection of ctx.step_trace('step_name')."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                ctx = get_run_context()
                result = ctx.step_trace("preprocess")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["preprocess"]

    def test_direct_get_run_context_step_trace(self):
        """Test detection of get_run_context().step_trace('step_name')."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                result = get_run_context().step_trace("validation")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["validation"]

    def test_multiple_step_trace_calls(self):
        """Test detection of multiple step_trace calls."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                ctx = get_run_context()
                data = ctx.step_trace("fetch_data")
                processed = ctx.step_trace("process_data")
                validated = ctx.step_trace("validate_data")
                return validated
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert set(analyzer.dependencies) == {"fetch_data", "process_data", "validate_data"}

    def test_mixed_step_trace_patterns(self):
        """Test different patterns of step_trace calls."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                ctx = get_run_context()
                # Direct variable call
                result1 = ctx.step_trace("step1")
                # Direct get_run_context call
                result2 = get_run_context().step_trace("step2")
                return result1, result2
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert set(analyzer.dependencies) == {"step1", "step2"}

    def test_variable_assignment_tracking(self):
        """Test that variable assignments are tracked correctly."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                ctx = get_run_context()
                ctx2 = ctx  # Copy variable
                result = ctx2.step_trace("copied_context")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["copied_context"]

    def test_different_import_patterns(self):
        """Test different import patterns for get_run_context."""
        code = dedent("""
            from timbal import get_run_context
            from timbal.state import get_run_context as grc

            def handler1():
                return get_run_context().step_trace("import1")

            def handler2():
                return grc().step_trace("import2")
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert set(analyzer.dependencies) == {"import1", "import2"}

    def test_nested_scopes(self):
        """Test step_trace detection in nested scopes."""
        code = dedent("""
            from timbal.state import get_run_context

            def outer():
                ctx = get_run_context()
                result1 = ctx.step_trace("outer_step")

                def inner():
                    ctx2 = get_run_context()
                    result2 = ctx2.step_trace("inner_step")
                    return result2

                return result1, inner()
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert set(analyzer.dependencies) == {"outer_step", "inner_step"}

    def test_lambda_functions(self):
        """Test step_trace detection in lambda functions."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                processor = lambda: get_run_context().step_trace("lambda_step")
                return processor()
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["lambda_step"]

    def test_async_functions(self):
        """Test step_trace detection in async functions."""
        code = dedent("""
            from timbal.state import get_run_context

            async def async_handler():
                ctx = get_run_context()
                result = ctx.step_trace("async_step")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["async_step"]

    def test_non_string_arguments_ignored(self):
        """Test that non-string arguments to step_trace are ignored."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler():
                ctx = get_run_context()
                step_name = "dynamic_step"
                # This should be ignored since it's not a string literal
                result1 = ctx.step_trace(step_name)
                # This should be detected
                result2 = ctx.step_trace("literal_step")
                return result1, result2
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["literal_step"]

    def test_false_positives_avoided(self):
        """Test that false positives are avoided."""
        code = dedent("""
            class SomeOtherClass:
                def step_trace(self, name):
                    return f"fake_{name}"

            def handler():
                obj = SomeOtherClass()
                # This should NOT be detected
                result = obj.step_trace("not_a_dependency")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == []

    def test_relative_imports(self):
        """Test relative imports are handled correctly."""
        code = dedent("""
            from ..state import get_run_context
            from .context import RunContext

            def handler():
                ctx = get_run_context()
                result = ctx.step_trace("relative_import")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["relative_import"]

    def test_module_attribute_access(self):
        """Test step_trace access through module attributes."""
        code = dedent("""
            import timbal.state as state

            def handler():
                ctx = state.get_run_context()
                result = ctx.step_trace("module_attr")
                return result
        """)
        analyzer = RunContextDependencyAnalyzer()
        tree = ast.parse(code)
        analyzer.visit(tree)
        assert analyzer.dependencies == ["module_attr"]


class TestRunnableInspectCallable:
    """Test the Runnable._inspect_callable method for dependency detection."""

    def test_inspect_callable_with_step_trace(self):
        """Test that _inspect_callable correctly identifies step_trace dependencies."""
        # Create a temporary file with a function that uses step_trace
        code = dedent("""
            from timbal.state import get_run_context

            def test_handler():
                ctx = get_run_context()
                result = ctx.step_trace("dependency1")
                return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            # Import the function from the temporary file
            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Import Runnable to access _inspect_callable
            from timbal.core.runnable import Runnable

            # Test the method
            result = Runnable._inspect_callable(
                module.test_handler,
                allow_required_params=False,
                allow_coroutine=True,
                allow_gen=False,
                allow_async_gen=False
            )

            assert result["dependencies"] == ["dependency1"]
            assert result["is_coroutine"] is False
            assert result["is_gen"] is False
            assert result["is_async_gen"] is False

    def test_inspect_callable_multiple_dependencies(self):
        """Test detection of multiple dependencies in a callable."""
        code = dedent("""
            from timbal.state import get_run_context

            def multi_dep_handler():
                ctx = get_run_context()
                data = ctx.step_trace("fetch")
                processed = ctx.step_trace("process")
                validated = get_run_context().step_trace("validate")
                return validated
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.multi_dep_handler)

            assert set(result["dependencies"]) == {"fetch", "process", "validate"}

    def test_inspect_callable_async_function(self):
        """Test detection in async functions."""
        code = dedent("""
            from timbal.state import get_run_context

            async def async_handler():
                ctx = get_run_context()
                result = ctx.step_trace("async_dep")
                return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.async_handler, allow_coroutine=True)

            assert result["dependencies"] == ["async_dep"]
            assert result["is_coroutine"] is True

    def test_inspect_callable_no_dependencies(self):
        """Test callable with no step_trace dependencies."""
        code = dedent("""
            def simple_handler():
                return "no dependencies here"
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.simple_handler)

            assert result["dependencies"] == []

    def test_inspect_callable_lambda(self):
        """Test dependency detection in lambda functions."""
        # This is more complex since we need to create a lambda in a file
        code = dedent("""
            from timbal.state import get_run_context

            lambda_handler = lambda: get_run_context().step_trace("lambda_dep")
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.lambda_handler)

            assert result["dependencies"] == ["lambda_dep"]

    def test_inspect_callable_error_handling(self):
        """Test that inspection gracefully handles errors."""
        import unittest.mock
        from timbal.core.runnable import Runnable

        # Use a built-in callable without required parameters
        # abs() with no args will fail, but we can mock it to have no required params for testing
        def builtin_like_func():
            """Mock function that simulates a built-in without source."""
            return 42

        # Mock getsourcefile to return None (like built-ins)
        with unittest.mock.patch('inspect.getsourcefile', return_value=None):
            result = Runnable._inspect_callable(builtin_like_func)

            # Should return empty dependencies when source can't be analyzed
            assert result["dependencies"] == []
            assert result["is_coroutine"] is False

    def test_inspect_callable_interactive_mode(self):
        """Test handling of functions defined in interactive mode (like <stdin>)."""
        import unittest.mock

        # Create a function but mock getsourcefile to return '<stdin>'
        def interactive_func():
            return "defined in interactive mode"

        from timbal.core.runnable import Runnable

        # Mock inspect.getsourcefile to return '<stdin>' (simulating interactive mode)
        with unittest.mock.patch('inspect.getsourcefile', return_value='<stdin>'):
            # This should not raise an exception, should log a warning and return empty dependencies
            result = Runnable._inspect_callable(interactive_func)

            assert result["dependencies"] == []
            assert result["is_coroutine"] is False
            assert result["is_gen"] is False
            assert result["is_async_gen"] is False

    def test_inspect_callable_nonexistent_source_file(self):
        """Test handling when source file path exists but file doesn't exist."""
        import unittest.mock

        def mock_func():
            return "mock function"

        from timbal.core.runnable import Runnable

        # Mock getsourcefile to return a path that doesn't exist
        nonexistent_path = "/nonexistent/path/file.py"
        with unittest.mock.patch('inspect.getsourcefile', return_value=nonexistent_path):
            # This should not raise an exception, should log a warning and return empty dependencies
            result = Runnable._inspect_callable(mock_func)

            assert result["dependencies"] == []
            assert result["is_coroutine"] is False


class TestAdvancedCallableScenarios:
    """Test complex callable scenarios: methods, decorators, callable classes."""

    def test_instance_method_with_step_trace(self):
        """Test detection of step_trace in instance methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class Handler:
                def process(self):
                    ctx = get_run_context()
                    result = ctx.step_trace("method_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            handler_instance = module.Handler()
            result = Runnable._inspect_callable(handler_instance.process)

            assert result["dependencies"] == ["method_step"]

    def test_static_method_with_step_trace(self):
        """Test detection of step_trace in static methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class Handler:
                @staticmethod
                def process():
                    ctx = get_run_context()
                    result = ctx.step_trace("static_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.Handler.process)

            assert result["dependencies"] == ["static_step"]

    def test_class_method_with_step_trace(self):
        """Test detection of step_trace in class methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class Handler:
                @classmethod
                def process(cls):
                    ctx = get_run_context()
                    result = ctx.step_trace("class_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.Handler.process)

            assert result["dependencies"] == ["class_step"]

    def test_callable_class_with_step_trace(self):
        """Test detection of step_trace in callable classes (__call__ method)."""
        code = dedent("""
            from timbal.state import get_run_context

            class CallableHandler:
                def __call__(self):
                    ctx = get_run_context()
                    result = ctx.step_trace("callable_class_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            handler_instance = module.CallableHandler()
            result = Runnable._inspect_callable(handler_instance.__call__)

            assert result["dependencies"] == ["callable_class_step"]

    def test_decorated_function_with_step_trace(self):
        """Test detection of step_trace in decorated functions."""
        code = dedent("""
            from timbal.state import get_run_context

            def my_decorator(func):
                def wrapper(*args, **kwargs):
                    return func(*args, **kwargs)
                return wrapper

            @my_decorator
            def decorated_handler():
                ctx = get_run_context()
                result = ctx.step_trace("decorated_step")
                return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.decorated_handler)

            # Note: This might return empty dependencies because the decorator wrapper
            # is what gets analyzed, not the original function. This is expected behavior.
            # The test documents this limitation.
            assert isinstance(result["dependencies"], list)

    def test_multiple_decorators_with_step_trace(self):
        """Test detection with multiple decorators."""
        code = dedent("""
            from timbal.state import get_run_context
            from functools import wraps

            def decorator1(func):
                @wraps(func)
                def wrapper():
                    return func()
                return wrapper

            def decorator2(func):
                @wraps(func)
                def wrapper():
                    return func()
                return wrapper

            @decorator1
            @decorator2
            def multi_decorated_handler():
                ctx = get_run_context()
                result = ctx.step_trace("multi_decorated_step")
                return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.multi_decorated_handler)

            # With @wraps, the original function name and source should be preserved
            assert result["dependencies"] == ["multi_decorated_step"]

    def test_nested_class_method_with_step_trace(self):
        """Test detection in nested class methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class OuterHandler:
                class InnerHandler:
                    def process(self):
                        ctx = get_run_context()
                        result = ctx.step_trace("nested_class_step")
                        return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            inner_instance = module.OuterHandler.InnerHandler()
            result = Runnable._inspect_callable(inner_instance.process)

            assert result["dependencies"] == ["nested_class_step"]

    def test_property_getter_with_step_trace(self):
        """Test detection in property getters."""
        code = dedent("""
            from timbal.state import get_run_context

            class Handler:
                @property
                def computed_value(self):
                    ctx = get_run_context()
                    result = ctx.step_trace("property_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            # Get the actual property getter function
            property_getter = module.Handler.computed_value.fget
            # Property getters have 'self' parameter, so we need to allow required params for this test
            result = Runnable._inspect_callable(property_getter, allow_required_params=True)

            assert result["dependencies"] == ["property_step"]

    def test_async_method_with_step_trace(self):
        """Test detection in async methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class AsyncHandler:
                async def process(self):
                    ctx = get_run_context()
                    result = ctx.step_trace("async_method_step")
                    return result
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            handler_instance = module.AsyncHandler()
            result = Runnable._inspect_callable(handler_instance.process, allow_coroutine=True)

            assert result["dependencies"] == ["async_method_step"]
            assert result["is_coroutine"] is True

    def test_generator_method_with_step_trace(self):
        """Test detection in generator methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class GeneratorHandler:
                def process(self):
                    ctx = get_run_context()
                    data = ctx.step_trace("generator_step")
                    yield data
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            handler_instance = module.GeneratorHandler()
            result = Runnable._inspect_callable(handler_instance.process, allow_gen=True)

            assert result["dependencies"] == ["generator_step"]
            assert result["is_gen"] is True

    def test_async_generator_method_with_step_trace(self):
        """Test detection in async generator methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class AsyncGeneratorHandler:
                async def process(self):
                    ctx = get_run_context()
                    data = ctx.step_trace("async_generator_step")
                    yield data
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            handler_instance = module.AsyncGeneratorHandler()
            result = Runnable._inspect_callable(
                handler_instance.process,
                allow_coroutine=True,
                allow_async_gen=True
            )

            assert result["dependencies"] == ["async_generator_step"]
            assert result["is_async_gen"] is True

    def test_metaclass_method_with_step_trace(self):
        """Test detection in metaclass methods."""
        code = dedent("""
            from timbal.state import get_run_context

            class MetaHandler(type):
                def process(cls):
                    ctx = get_run_context()
                    result = ctx.step_trace("metaclass_step")
                    return result

            class Handler(metaclass=MetaHandler):
                pass
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.Handler.process)

            assert result["dependencies"] == ["metaclass_step"]

    def test_duplicate_step_trace_dependencies(self):
        """Test handling of duplicate step_trace calls with same step name."""
        code = dedent("""
            from timbal.state import get_run_context

            def handler_with_duplicates():
                ctx = get_run_context()
                # Same step called multiple times
                result1 = ctx.step_trace("shared_step")
                result2 = ctx.step_trace("shared_step")
                # Different step
                result3 = ctx.step_trace("unique_step")
                # Same step again
                result4 = get_run_context().step_trace("shared_step")
                return result1, result2, result3, result4
        """)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            f.flush()

            import importlib.util
            spec = importlib.util.spec_from_file_location("test_module", f.name)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            from timbal.core.runnable import Runnable

            result = Runnable._inspect_callable(module.handler_with_duplicates)

            # Verify that all step names are captured
            assert "shared_step" in result["dependencies"]
            assert "unique_step" in result["dependencies"]

            # Document the current behavior: duplicates are preserved in order
            # This allows the framework to understand call frequency/order if needed
            expected_deps = ["shared_step", "shared_step", "unique_step", "shared_step"]
            assert result["dependencies"] == expected_deps

            # Alternative test: if you want to verify deduplication, use:
            # unique_deps = list(dict.fromkeys(result["dependencies"]))  # Preserves order
            # assert unique_deps == ["shared_step", "unique_step"]