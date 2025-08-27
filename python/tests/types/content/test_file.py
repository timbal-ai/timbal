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

def test_file_to_openai_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content) 
    file = File.validate(str(test_file))
    file_content = FileContent(file=file)

    with open(test_file, "rb") as f:
        png_content = f.read()
        base64_content = base64.b64encode(png_content).decode("utf-8")
        data_url = f"data:image/png;base64,{base64_content}"

    assert file_content.to_openai_input() == {"type": "image_url", "image_url": {"url": data_url}}


def test_file_to_anthropic_input(tmp_path: pathlib.Path) -> None:
    test_file = tmp_path / "image.png"
    png_content = bytes.fromhex(
        '89504e470d0a1a0a'  # PNG signature
    )
    test_file.write_bytes(png_content) 
    file = File.validate(str(test_file))
    file_content = FileContent(file=file)
    assert file_content.to_anthropic_input() == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png_content).decode("utf-8")}}
