from collections.abc import Callable
from typing import Any

from .base import BaseStep


RunnableLike = BaseStep | Callable[..., Any]
