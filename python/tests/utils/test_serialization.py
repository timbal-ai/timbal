"""Tests for utils/serialization.py — dump() and coerce_to_dict()."""

import math
from pathlib import Path

import pytest
from pydantic import BaseModel

from timbal.utils.serialization import coerce_to_dict, dump


class TestDumpSyncFastPath:
    async def test_primitives_passthrough(self):
        assert await dump(42) == 42
        assert await dump("hello") == "hello"
        assert await dump(True) is True
        assert await dump(None) is None

    async def test_nan_float_becomes_none(self):
        assert await dump(float("nan")) is None

    async def test_float_rounded(self):
        result = await dump(1.123456789012345)
        assert isinstance(result, float)
        # should be rounded to 10 decimal places
        assert result == round(1.123456789012345, 10)

    async def test_whole_float_unchanged(self):
        assert await dump(3.0) == 3.0

    async def test_dict_recursed(self):
        assert await dump({"a": 1, "b": "x"}) == {"a": 1, "b": "x"}

    async def test_list_recursed(self):
        assert await dump([1, "two", None]) == [1, "two", None]

    async def test_tuple_becomes_tuple(self):
        result = await dump((1, 2, 3))
        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    async def test_path_as_posix(self):
        assert await dump(Path("/tmp/file.txt")) == "/tmp/file.txt"

    async def test_exception_serialized(self):
        result = await dump(ValueError("bad input"))
        assert result == {"error_type": "ValueError", "message": "bad input"}

    async def test_pydantic_model(self):
        class Inner(BaseModel):
            x: int

        result = await dump(Inner(x=5))
        assert result == {"x": 5}

    async def test_nested_dict_with_list(self):
        value = {"items": [1, 2, 3], "meta": {"count": 3}}
        assert await dump(value) == value


class TestDumpAsyncPath:
    """Tests that force the async path via File objects in the value tree."""

    async def test_file_in_list_triggers_async(self):
        from timbal.types.file import File

        file_obj = File.validate(b"hello world")

        result = await dump([file_obj])
        assert isinstance(result, list)
        assert len(result) == 1

    async def test_file_in_dict_triggers_async(self):
        from timbal.types.file import File

        file_obj = File.validate(b"hello world")

        result = await dump({"file": file_obj, "name": "test"})
        assert isinstance(result, dict)
        assert result["name"] == "test"

    async def test_async_nan_becomes_none(self, monkeypatch):
        """Exercise the async path's NaN handling by patching _NEEDS_ASYNC."""
        # We test the async path for float NaN directly
        from timbal.utils import serialization

        result = await serialization._dump_async(float("nan"))
        assert result is None

    async def test_async_passthrough_primitives(self):
        from timbal.utils import serialization

        assert await serialization._dump_async(42) == 42
        assert await serialization._dump_async("hi") == "hi"
        assert await serialization._dump_async(None) is None

    async def test_async_float_rounded(self):
        from timbal.utils import serialization

        result = await serialization._dump_async(1.123456789012345)
        assert result == round(1.123456789012345, 10)

    async def test_async_dict(self):
        from timbal.utils import serialization

        result = await serialization._dump_async({"a": 1, "b": "x"})
        assert result == {"a": 1, "b": "x"}

    async def test_async_empty_dict(self):
        from timbal.utils import serialization

        result = await serialization._dump_async({})
        assert result == {}

    async def test_async_list(self):
        from timbal.utils import serialization

        result = await serialization._dump_async([1, 2, 3])
        assert result == [1, 2, 3]

    async def test_async_tuple(self):
        from timbal.utils import serialization

        result = await serialization._dump_async((1, 2))
        assert isinstance(result, tuple)
        assert result == (1, 2)

    async def test_async_path(self):
        from timbal.utils import serialization

        result = await serialization._dump_async(Path("/tmp/x"))
        assert result == "/tmp/x"

    async def test_async_exception(self):
        from timbal.utils import serialization

        result = await serialization._dump_async(RuntimeError("oops"))
        assert result == {"error_type": "RuntimeError", "message": "oops"}

    async def test_async_pydantic_model(self):
        from timbal.utils import serialization

        class M(BaseModel):
            val: int

        result = await serialization._dump_async(M(val=7))
        assert result == {"val": 7}

    async def test_async_unknown_type_becomes_str(self):
        from timbal.utils import serialization

        class Custom:
            def __str__(self):
                return "custom_str"

        result = await serialization._dump_async(Custom())
        assert result == "custom_str"


class TestCoerceToDict:
    def test_dict_passthrough(self):
        d = {"key": "value"}
        assert coerce_to_dict(d) is d

    def test_json_string(self):
        assert coerce_to_dict('{"a": 1}') == {"a": 1}

    def test_empty_string_returns_empty_dict(self):
        assert coerce_to_dict("") == {}
        assert coerce_to_dict("   ") == {}

    def test_literal_eval_fallback(self):
        # Python dict literal that isn't valid JSON (single quotes)
        assert coerce_to_dict("{'a': 1}") == {"a": 1}

    def test_unparseable_string_raises(self):
        with pytest.raises(ValueError, match="Cannot coerce value to dict"):
            coerce_to_dict("not a dict at all!")

    def test_non_string_non_dict_raises(self):
        with pytest.raises(ValueError, match="Cannot coerce value to dict"):
            coerce_to_dict(42)

    def test_nested_json(self):
        assert coerce_to_dict('{"outer": {"inner": [1, 2, 3]}}') == {
            "outer": {"inner": [1, 2, 3]}
        }
