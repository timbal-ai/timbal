import pytest
from pydantic import TypeAdapter
from timbal.errors import DataKeyError
from timbal.state.data import Data, DataError, DataMap, DataValue, get_data_key


def test_data_value():
    data = TypeAdapter(Data).validate_python({
        "type": "value", 
        "value": "test",
    })

    assert isinstance(data, DataValue)
    assert data.resolve() == "test"


def test_data_error():
    data = TypeAdapter(Data).validate_python({
        "type": "error", 
        "error": "test",
    })

    assert isinstance(data, DataError)
    assert data.resolve() == data


def test_data_map():
    data = TypeAdapter(Data).validate_python({
        "type": "map", 
        "key": "1.return"
    })

    assert isinstance(data, DataMap)

    context_data = {"1.return": DataValue(value="test")}
    assert data.resolve(context_data=context_data) == "test"

    with pytest.raises(DataKeyError):
        data = DataMap(key="abc.return")
        print(data.resolve(context_data=context_data)) # noqa: T201


def test_get_data_key():
    data = {"1.return": DataValue(value=[{"test_key": ["test_value"]}])}

    assert get_data_key(data, "1.return.0.test_key.0") == "test_value"

    # Accessing a list index that doesn't exist
    with pytest.raises(KeyError):
        get_data_key(data, "1.return.0.test_key.1")

    # Accessing a list index from a dict
    with pytest.raises(KeyError):
        get_data_key(data, "1.return.0.0")

    # Treating a list as an object with properties
    with pytest.raises(KeyError):
        get_data_key(data, "1.return.0.test_key.something")

    # Searching for a key that doesn't exist
    with pytest.raises(DataKeyError):
        get_data_key(data, "1.metadata")

    # Regex validation
    with pytest.raises(ValueError):
        get_data_key(data, "invalid data key")
