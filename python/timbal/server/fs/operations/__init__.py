from typing import Annotated

from pydantic import Field, TypeAdapter

from .copy import CopyOperation
from .delete import DeleteOperation
from .list import ListOperation
from .mkdir import MkdirOperation
from .move import MoveOperation
from .read import ReadOperation
from .write import WriteOperation

FileSystemOperation = Annotated[
    CopyOperation | DeleteOperation | ListOperation | MkdirOperation |
    MoveOperation | ReadOperation | WriteOperation,
    Field(discriminator="type"),
]
operation_adapter = TypeAdapter(FileSystemOperation)
