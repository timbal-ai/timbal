from typing import Any, override

from ...types.events import Event, OutputEvent
from .base import BaseCollector


class AgentCollector(BaseCollector):
    """"""

    last_output_event: OutputEvent | None = None

    @override
    def handle_chunk(self, chunk: Any) -> Any | None:
        """"""
        # NOTE: An agent handler should only yield events yielded by timbal ToolLike instances.
        assert isinstance(chunk, Event), \
            f"AgentCollector expected an Event, got {type(chunk)}"
        
        if isinstance(chunk, OutputEvent):
            self.last_output_event = chunk
        # We'll want to stream this upwards. 
        return chunk


    @override
    def collect(self) -> Any:
        """"""
        # NOTE: The last yielded event of an agent should be an OutputEvent, always.
        assert isinstance(self.last_output_event, OutputEvent), \
            f"AgentCollector expected an OutputEvent, got {type(self.last_output_event)}"

        return self.last_output_event.output
