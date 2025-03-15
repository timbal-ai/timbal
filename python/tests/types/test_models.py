# ruff: noqa: B008
import inspect
from typing import Annotated, Any, Literal, Optional, TypeVar

import pytest
from pydantic import ValidationError
from timbal import Flow
from timbal.types import Field, File
from timbal.types.models import create_model_from_annotation, create_model_from_argspec


def test_no_annotations_no_defaults():
    def handler(a, b, c): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    handler_args = model_cls.model_validate({"a": 1, "b": 2, "c": 3})
    assert dict(handler_args) == {"a": 1, "b": 2, "c": 3}

    with pytest.raises(ValueError):
        model_cls.model_validate({"a": 1, "b": 2})


def test_no_defaults():
    def handler(a, b: str, c: float): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    handler_args = model_cls.model_validate({"a": 1, "b": "2", "c": 3.0})
    assert dict(handler_args) == {"a": 1, "b": "2", "c": 3.0}

    with pytest.raises(ValidationError):
        model_cls.model_validate({"a": 1, "b": "2"})

    with pytest.raises(ValidationError):
        model_cls.model_validate({"a": 1, "b": "2", "c": "this type is wrong"})


def test_choices():
    def handler(
        a: Literal["a", "b", "c"],
        b: str = Field(choices=["a", "b", "c"])
    ): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    handler_args = model_cls.model_validate({"a": "a", "b": "b"})
    assert dict(handler_args) == {"a": "a", "b": "b"}

    with pytest.raises(ValidationError):
        model_cls.model_validate({"a": "a", "b": "d"})

    with pytest.raises(ValidationError):
        model_cls.model_validate({"a": "f", "b": "b"})
    

def test_regex():
    def handler(a: str = Field(regex=r"^[a-z]+$")): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    model_args = {"a": "abc"}
    model = model_cls(**model_args)
    assert model.model_dump() == {"a": "abc"}

    model_args = {"a": "123"}
    with pytest.raises(ValidationError):
        model_cls(**model_args)


def test_optionals():
    def handler(
        a: Optional[str], # noqa: UP007
        b: float | None, 
        c: int | None = None,
        d: list[int] | None = None,
    ): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    model_args = {"a": "a", "b": 1.2, "c": 3, "d": [1, 2, 3]}
    model = model_cls(**model_args)
    assert model.model_dump() == {"a": "a", "b": 1.2, "c": 3, "d": [1, 2, 3]}

    # With missing fields
    model_cls.model_validate({"a": "a", "b": 1.2})

    # Wrong list item
    with pytest.raises(ValidationError):
        model_cls.model_validate({"a": "a", "b": 1.2, "d": [1, 2, "abc"]})

    # Passing None
    model_cls.model_validate({"a": None, "b": None, "c": None, "d": None})


def test_generics():
    T1 = TypeVar("T1")
    T2 = TypeVar("T2")
    def handler(
        a: list[int], 
        b: list[T1], 
        c: list[T2] = Field(), 
        d: list[list[T2]] = Field()
    ): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    model_args = {"a": [1, 2, 3], "b": [6.7, 8.99], "c": [], "d": [["d"]]}
    model = model_cls(**model_args)
    assert model.model_dump() == {"a": [1, 2, 3], "b": [6.7, 8.99], "c": [], "d": [["d"]]}

    # list[int] must be a list of ints
    model_args = {"a": ["a"], "b": [], "c": [], "d": [["d"]]}
    with pytest.raises(ValueError):
        model_cls(**model_args)

    # list[T] must be a list of the same type
    model_args = {"a": [1], "b": ["a", "b", 1], "c": [], "d": [["d"]]}
    with pytest.raises(ValueError):
        model_cls(**model_args)

    # list[T] must be a list of the same type between fields
    model_args = {"a": [1], "b": [1.2, 3.4], "c": [1], "d": [["d"]]}
    with pytest.raises(ValueError):
        model_cls(**model_args)


def test_file():
    def handler(a: File): pass

    argspec = inspect.getfullargspec(handler)
    model_cls = create_model_from_argspec("TestModel", argspec)

    file_url = "https://content.eticko.com/assets/timbal.png"

    model_args = {"a": file_url}
    model = model_cls(**model_args)

    file = File.validate(file_url)
    file_serialized = File.serialize(file)
    assert model.model_dump() == {"a": file_serialized}


# def test_message():
#     def handler(a: Message): pass

#     argspec = inspect.getfullargspec(handler)
#     model_cls = create_model_from_argspec("TestModel", argspec)

#     model_cls.model_validate({"a": {"role": "user", "content": "hello"}})
#     model_cls.model_validate({"a": Message.validate({"role": "user", "content": "hello"})})

    # from timbal import Flow

    # flow = (
    #     Flow(id="test_flow_message")
    #     .add_llm("llm")
    #     .set_data_map("llm.prompt", "prompt")
    # )

    # prompt = "hello"
    # prompt = File.validate("https://content.eticko.com/assets/timbal.png")
    # prompt = {"type": "text", "text": "hello"}
    # prompt = [File.validate("https://content.eticko.com/assets/timbal.png"), "describe the shape of the logo"]
    # prompt = [
    #     {"type": "file", "file": File.validate("https://content.eticko.com/assets/timbal.png")}, 
    #     "describe the shape of the logo",
    #     "what do you think the company behind the logo is about?",
    #     "invent the background story of the company"
    # ]
    # prompt = {
    #     "role": "user",
    #     "content": "how are you in this fine day?"
    # }
    # prompt = Message.validate({"role": "user", "content": "how are you in this fine day?"})
    # async for event_code, event_data in flow.run(prompt=prompt):
    #     print(event_code, event_data)


def test_return_model():
    def handler(a) -> int:
        return a

    return_annotation = inspect.getfullargspec(handler).annotations.get("return", ...)
    return_model = create_model_from_annotation("TempModel", return_annotation)

    return_model.model_validate({"return": handler(1)})

    with pytest.raises(ValidationError):
        return_model.model_validate({"return": handler("a")})


def test_return_model_any():
    def handler(a) -> Any:
        return a

    return_annotation = inspect.getfullargspec(handler).annotations.get("return", ...)
    return_model = create_model_from_annotation("TempModel", return_annotation)

    return_model.model_validate({"return": handler(23)})


def test_return_model_no_annotation():
    def handler(a):
        return a

    return_annotation = inspect.getfullargspec(handler).annotations.get("return", ...)
    return_model = create_model_from_annotation("TempModel", return_annotation)

    return_model.model_validate({"return": handler({})})


def test_return_model_none():
    def handler(a) -> None:
        pass

    return_annotation = inspect.getfullargspec(handler).annotations.get("return", ...)
    return_model = create_model_from_annotation("TempModel", return_annotation)

    return_model.model_validate({"return": handler(None)})


def test_return_model_annotated():
    def handler(a) -> Annotated[int, "this is a test"]:
        return a

    return_annotation = inspect.getfullargspec(handler).annotations.get("return", ...)
    return_model = create_model_from_annotation("TempModel", return_annotation)

    return_model.model_validate({"return": handler(1)})


def test_flow_return_model():
    def handler_1(a, b) -> int: pass
    def handler_2(a) -> str: pass
    def handler_3(a) -> list[str]: pass
    def handler_4(a) -> dict[str, int]: pass
    def handler_5(a) -> Any: pass

    flow = (
        Flow(id="test_flow_return_model")
        .add_step("1", handler_1)
        .add_step("2", handler_2)
        .add_step("3", handler_3)
        .add_step("4", handler_4)
        .add_step("5", handler_5)
        .set_data_value("something", ["hello", "world"])
        .set_output("1.return", "out_1")
        .set_output("2.return", "out_2")
        .set_output("3.return", "out_3")
        .set_output("4.return.abc", "out_4")
        .set_output("5.return", "out_5")
        .set_output("something.1", "out_6")
    )

    flow_return_model = flow.return_model()
    flow.return_model_schema() # Simply ensure this doesn't raise an error

    flow_return_model.model_validate({
        "out_1": 1,
        "out_2": "world",
        "out_3": ["hello", "world"],
        "out_4": {"abc": 123},
        "out_5": 34.12,
        "out_6": "world",
    })
    