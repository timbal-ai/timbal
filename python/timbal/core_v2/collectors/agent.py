from typing import Any, override

from ..events.base import BaseEvent
from .base import BaseCollector


class AgentCollector(BaseCollector):
    """"""

    last_event: BaseEvent | None = None

    @override
    def handle_chunk(self, chunk: Any) -> Any | None:
        """"""
        # NOTE: An agent handler should only yield events yielded by timbal ToolLike instances.
        assert isinstance(chunk, BaseEvent), \
            f"AgentCollector expected a BaseEvent, got {type(chunk)}"
        
        # Hence, all these chunks will be 'nested' steps inside of the agent. 
        # Therefore, we prepend the agent's path to each chunk.
        self.last_event = chunk
        # We'll want to stream this upwards. 
        return chunk


    @override
    def collect(self) -> Any:
        """"""
        
        # NOTE: The last yielded event of an agent should be an OutputEvent, always.
        assert isinstance(self.last_event, BaseEvent), \
            f"AgentCollector expected a BaseEvent, got {type(self.last_event)}"

        return self.last_event
