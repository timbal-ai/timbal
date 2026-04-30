from typing import Any

try:
    from typing import override
except ImportError:
    from typing_extensions import override

from ...types.message import Message
from .. import register_collector
from ..base import BaseCollector


@register_collector
class MessageCollector(BaseCollector):
    """Collector for Message objects yielded by TestModel. No delta streaming — just stores the message."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._message: Message | None = None

    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, Message)

    @override
    def process(self, event: Message) -> None:
        """Store the message; emit no delta events."""
        self._message = event
        return None

    @override
    def result(self) -> Message | None:
        return self._message
