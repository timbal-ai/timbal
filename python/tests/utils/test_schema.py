"""Tests for utils/schema.py — transform_schema OpenAI strict-mode compatibility."""

import pytest
from pydantic import BaseModel

from timbal.utils.schema import transform_schema


class TestTransformSchemaRef:
    def test_ref_passthrough(self):
        """$ref schemas are returned as-is without further processing."""
        schema = {"$ref": "#/$defs/MyModel"}
        result = transform_schema(schema)
        assert result == {"$ref": "#/$defs/MyModel"}

    def test_ref_strips_other_keys(self):
        """Only $ref is kept when it appears in the schema."""
        schema = {"$ref": "#/$defs/Foo", "description": "ignored"}
        result = transform_schema(schema)
        assert result == {"$ref": "#/$defs/Foo"}


class TestTransformSchemaDefs:
    def test_defs_are_recursively_transformed(self):
        schema = {
            "type": "object",
            "properties": {},
            "$defs": {
                "Inner": {"type": "string"},
            },
        }
        result = transform_schema(schema)
        assert "$defs" in result
        assert result["$defs"]["Inner"] == {"type": "string"}

    def test_defs_object_gets_additionalProperties(self):
        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"y": {"type": "string"}},
                }
            },
        }
        result = transform_schema(schema)
        assert result["$defs"]["Inner"]["additionalProperties"] is False


class TestTransformSchemaComposition:
    def test_any_of_is_preserved(self):
        schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
        result = transform_schema(schema)
        assert "anyOf" in result
        assert len(result["anyOf"]) == 2

    def test_one_of_converted_to_any_of(self):
        schema = {"oneOf": [{"type": "string"}, {"type": "integer"}]}
        result = transform_schema(schema)
        assert "anyOf" in result
        assert "oneOf" not in result
        assert len(result["anyOf"]) == 2

    def test_all_of_raises(self):
        schema = {"allOf": [{"type": "string"}]}
        with pytest.raises(ValueError, match="allOf is not supported"):
            transform_schema(schema)

    def test_missing_type_raises(self):
        schema = {"description": "no type here"}
        with pytest.raises(ValueError, match="must have a 'type'"):
            transform_schema(schema)


class TestTransformSchemaUnsupportedKeywords:
    @pytest.mark.parametrize("keyword", ["not", "dependentRequired", "dependentSchemas", "if", "then", "else"])
    def test_unsupported_keywords_raise(self, keyword):
        schema = {"type": "object", "properties": {}, keyword: {}}
        with pytest.raises(ValueError, match=f"Unsupported JSON Schema keyword '{keyword}'"):
            transform_schema(schema)


class TestTransformSchemaObject:
    def test_object_additionalProperties_false(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = transform_schema(schema)
        assert result["additionalProperties"] is False

    def test_object_all_properties_required(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = transform_schema(schema)
        assert set(result["required"]) == {"name", "age"}

    def test_object_no_properties_no_required(self):
        schema = {"type": "object", "properties": {}}
        result = transform_schema(schema)
        assert "required" not in result

    def test_description_preserved(self):
        schema = {"type": "string", "description": "a name"}
        result = transform_schema(schema)
        assert result["description"] == "a name"

    def test_title_preserved(self):
        schema = {"type": "string", "title": "Name"}
        result = transform_schema(schema)
        assert result["title"] == "Name"


class TestTransformSchemaString:
    def test_supported_format_preserved(self):
        schema = {"type": "string", "format": "date-time"}
        result = transform_schema(schema)
        assert result["format"] == "date-time"

    def test_all_supported_formats_preserved(self):
        for fmt in ("date", "date-time", "email", "hostname", "ipv4", "ipv6", "uri", "uuid"):
            result = transform_schema({"type": "string", "format": fmt})
            assert result.get("format") == fmt, f"expected {fmt!r} to be preserved"

    def test_unsupported_format_moved_to_description(self):
        schema = {"type": "string", "format": "custom-format"}
        result = transform_schema(schema)
        assert "format" not in result
        assert "custom-format" in result.get("description", "")

    def test_string_no_format(self):
        schema = {"type": "string"}
        result = transform_schema(schema)
        assert result == {"type": "string"}


class TestTransformSchemaArray:
    def test_array_items_recursed(self):
        schema = {"type": "array", "items": {"type": "string"}}
        result = transform_schema(schema)
        assert result["items"] == {"type": "string"}

    def test_array_no_items(self):
        schema = {"type": "array"}
        result = transform_schema(schema)
        assert "items" not in result

    def test_array_min_items_zero_preserved(self):
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 0}
        result = transform_schema(schema)
        assert result["minItems"] == 0

    def test_array_min_items_one_preserved(self):
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 1}
        result = transform_schema(schema)
        assert result["minItems"] == 1

    def test_array_min_items_gt_one_moved_to_description(self):
        schema = {"type": "array", "items": {"type": "string"}, "minItems": 5}
        result = transform_schema(schema)
        assert "minItems" not in result
        assert "5" in result.get("description", "")


class TestTransformSchemaPydanticModel:
    def test_pydantic_model_class(self):
        class MyModel(BaseModel):
            name: str
            value: int

        result = transform_schema(MyModel)
        assert result["type"] == "object"
        assert "name" in result["properties"]
        assert "value" in result["properties"]
        assert result["additionalProperties"] is False

    def test_pydantic_optional_field_still_required(self):
        from typing import Optional

        class MyModel(BaseModel):
            name: str
            nickname: Optional[str] = None

        result = transform_schema(MyModel)
        assert "nickname" in result.get("required", [])


class TestTransformSchemaExtraProps:
    def test_extra_leftover_props_appended_to_description(self):
        schema = {"type": "integer", "exclusiveMinimum": 0}
        result = transform_schema(schema)
        assert "exclusiveMinimum" in result.get("description", "")

    def test_extra_props_appended_after_existing_description(self):
        schema = {"type": "integer", "description": "a count", "exclusiveMinimum": 0}
        result = transform_schema(schema)
        assert result["description"].startswith("a count")
        assert "exclusiveMinimum" in result["description"]
