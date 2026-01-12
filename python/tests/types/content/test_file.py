import base64
import pathlib

import pytest
from timbal.types import File
from timbal.types.content import FileContent, content_factory


def test_basic_file_content_validation(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content)
    content = content_factory({"type": "file", "file": File.validate(str(test_file))})
    assert isinstance(content, FileContent)
    assert isinstance(content.file, File)
    assert content.type == "file"

    # file must be a File
    with pytest.raises(ValueError):
        content_factory({"type": "file", "file": "not a file"})
