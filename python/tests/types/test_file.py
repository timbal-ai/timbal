import pathlib
from io import BytesIO

import pytest
from timbal.types import File


def test_file_validate_none() -> None:
    with pytest.raises(ValueError):
        File.validate(None)


def test_file_validate_any() -> None:
    with pytest.raises(ValueError):
        File.validate(object())


def test_file_serialize_none() -> None:
    assert File.serialize(None) is None


def test_file_serialize_any() -> None:
    with pytest.raises(ValueError):
        File.serialize(object())


def test_file_validate_bytes() -> None:
    test_content = b"Hello, World!"
    file = File.validate(test_content)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == test_content


def test_file_validate_bytearray() -> None:
    test_content = bytearray(b"Hello, World!")
    file = File.validate(test_content)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == bytes(test_content)


def test_file_validate_io_base() -> None:
    test_content = b"Hello, World!"
    test_io = BytesIO(test_content)
    file = File.validate(test_io)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "bytes"
    assert file.__source_extension__ is None
    assert file.read() == test_content


def test_file_serialize_byteslike() -> None:
    test_content = b"Hello, World!"
    test_io = BytesIO(test_content)

    file = File.validate(test_io)

    assert File.serialize(file) == "data:application/octet-stream;base64,SGVsbG8sIFdvcmxkIQ=="


def test_file_validate_local_path_from_str(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(str(test_file))

    assert isinstance(file, File)
    assert file.__source_scheme__ == "local_path"
    assert file.__source_extension__ == test_file.suffix
    assert file.read() == test_content


def test_file_validate_local_path_from_path(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(test_file)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "local_path"
    assert file.__source_extension__ == test_file.suffix
    assert file.read() == test_content


def test_file_serialize_local_path(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    file = File.validate(str(test_file))

    assert File.serialize(file) == "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="


def test_file_validate_data_url() -> None:
    test_content = b"Hello, World!"
    data_url = "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="

    file = File.validate(data_url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "data_url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content


def test_file_validate_data_url_with_custom_mime_type() -> None:
    test_content = b'{"key": "value"}'
    data_url = "data:timbal/jsonl;base64,eyJrZXkiOiAidmFsdWUifQ=="

    file = File.validate(data_url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "data_url"
    assert file.__source_extension__ == ".jsonl"
    assert file.read() == test_content


def test_file_serialize_data_url() -> None:
    data_url = "data:timbal/jsonl;base64,eyJrZXkiOiAidmFsdWUifQ=="

    file = File.validate(data_url)

    assert File.serialize(file) == data_url


def test_file_validate_url() -> None:
    test_content = b"Hello, World!\n"
    url = "https://content.timbal.ai/assets/hello_world.txt"

    file = File.validate(url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content


def test_file_serialize_url() -> None:
    url = "https://content.timbal.ai/assets/hello_world.txt"

    file = File.validate(url)

    assert File.serialize(file) == "data:text/plain;base64,SGVsbG8sIFdvcmxkIQo="
