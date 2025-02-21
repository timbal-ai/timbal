import pathlib

from timbal.types import File


def test_local_file_validation(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "test.txt"
    test_content = b"Hello, World!"
    test_file.write_bytes(test_content)

    # TODO Accept pathlib.Path objects in the File.validate method.
    test_file_str = str(test_file)

    file = File.validate(test_file_str)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "local_path"
    assert file.__source_extension__ == test_file.suffix
    assert file.read() == test_content


def test_data_url_validation() -> None:
    test_content = b"Hello, World!"
    data_url = "data:text/plain;base64,SGVsbG8sIFdvcmxkIQ=="

    file = File.validate(data_url)

    assert isinstance(file, File)
    assert file.__source_scheme__ == "data_url"
    assert file.__source_extension__ == ".txt"
    assert file.read() == test_content
