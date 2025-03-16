from collections.abc import Callable
from typing import Any

from ..errors import DataKeyError, FlowExecutionError
from ..state.data import BaseData, DataError, get_data_key
from ..types.message import Message
from .base import BaseStep
from .flow import Flow
from .utils import get_sources


class Agent(Flow):
    """Subclass of Flow that implements an LLM agent with tool-use capabilities.
    
    An Agent is a specialized Flow that creates a chain of LLM steps and tools, allowing
    the LLM to use tools multiple times in a conversation. Each LLM step can call any
    available tool, and the results are fed back to the next LLM step.

    Attributes:
        id (str): Unique identifier for the agent, defaults to "agent"
        tools (list): List of callable functions, BaseSteps, or dicts with tool configs
        max_iter (int): Maximum number of tool use iterations allowed, defaults to 1
        **kwargs: Additional keyword arguments for the LLMs

    Example:
        ```python
        agent = Agent(
            tools=[search_tool, calculator],
            max_iter=3,
            system_prompt="You are a helpful assistant"
        )
        ```
    """

    def __init__(
        self,
        id: str | None = None,
        tools: list[Callable | BaseStep | dict[str, Any]] = [],
        max_iter: int = 1,
        **kwargs: Any,
    ) -> None:
        """Initialize an Agent instance.

        Args:
            id: Unique identifier for the agent, defaults to "agent"
            tools: List of tools the agent can use
            max_iter: Maximum number of tool use iterations
            **kwargs: Additional LLM configuration options
        """
        if id is None:
            id = "agent"

        # We don't allow to pass a memory_id here. Agents will always use the same memory.
        # And they must have one, so they can use tool results with the corresponding tool calls.
        kwargs.pop("memory_id", None)
        memory_id = id

        kwargs_system_prompt = kwargs.pop("system_prompt", None)

        super().__init__(id=id)

        # Create initial LLM step that receives the prompt
        entrypoint_llm_id = f"{id}_llm_0"
        self.add_llm(
            id=entrypoint_llm_id,
            memory_id=memory_id,
            system_prompt=kwargs_system_prompt,
            **kwargs,
        )

        # TODO Think this through. Should we expose all the params here? Should we pass them from outside in __init__?
        self.set_data_map(f"{entrypoint_llm_id}.prompt", "prompt")

        if not len(tools):
            return

        # Default system prompt when processing tool results. Give the ability to the user to override this from the outside.
        tool_result_llm_system_prompt = """Please process the tool result given the call. 
        Do not remove references or citations from text. Leave them in markdown format."""
        if kwargs_system_prompt is not None:
            tool_result_llm_system_prompt = kwargs_system_prompt

        prev_llm_id = entrypoint_llm_id
        # Create additional LLM steps for each iteration, allowing the agent to use tools multiple times
        for i in range(max_iter):
            iter_llm_id = f"{id}_llm_{i + 1}"
            self.add_llm(
                id=iter_llm_id,
                memory_id=memory_id,
                system_prompt=tool_result_llm_system_prompt,
                **kwargs,
            )

            # For each tool, create a step and add links for tool use and tool result
            for tool_item in tools:
                if isinstance(tool_item, dict):
                    tool = tool_item["tool"]
                    description = tool_item.get("description")
                else:
                    tool = tool_item
                    description = None

                if callable(tool):
                    tool_id = tool.__name__
                elif isinstance(tool, BaseStep):
                    tool_id = tool.id

                iter_tool_id = f"{id}_{tool_id}_{i}"

                if iter_tool_id not in self.steps:
                    self.add_step(iter_tool_id, tool)

                self.add_link(
                    step_id=prev_llm_id,
                    description=description,
                    next_step_id=iter_tool_id,
                    is_tool=True,
                )

                self.add_link(
                    step_id=iter_tool_id,
                    next_step_id=iter_llm_id,
                    is_tool_result=True,
                )
    
            prev_llm_id = iter_llm_id


    def _collect_outputs(self, data: dict[str, BaseData]) -> Any:
        """Override the method to collect the last available LLM output from the flow always."""
        if len(self.outputs):
            return super()._collect_outputs(data)

        _, rev_dag = self.get_dags()
        rev_sources = list(get_sources(rev_dag))
        assert len(rev_sources) == 1, "Tool to LLM agent mode should have a single LLM as last step"

        last_llm_output = None
        last_llm_id = rev_sources[0]
        while last_llm_output is None:
            try: 
                last_llm_output_key = f"{last_llm_id}.return"
                last_llm_output = get_data_key(data, last_llm_output_key)
                if isinstance(last_llm_output, DataError):
                    flow_execution_error_key = f"{self.path}.{last_llm_id}"
                    raise FlowExecutionError(f"Error collecting outputs {{'{flow_execution_error_key}'}}.")
            except DataKeyError:
                last_llm_output = None

                last_tools_ids = list(rev_dag[last_llm_id])
                assert len(last_tools_ids) >= 1, \
                    "The last LLM of a tool to LLM agent must have at least one tool result."
                last_tools_errors = set()
                for last_tool_id in last_tools_ids:
                    last_tool_output_key = f"{last_tool_id}.return"
                    try:
                        last_tool_output = get_data_key(data, last_tool_output_key)
                        if isinstance(last_tool_output, DataError):
                            flow_execution_error_key = f"{self.path}.{last_tool_id}"
                            last_tools_errors.add(flow_execution_error_key)
                    except DataKeyError:
                        pass

                last_llms_ids = list(rev_dag[last_tools_ids[0]])
                assert len(last_llms_ids) == 1, \
                    "Tool to LLM agent mode should have a single LLM as last step"
                last_llm_id = last_llms_ids[0]
        return last_llm_output

    
    def return_model(self) -> Any:
        """Override the return model to return always a Message."""
        if len(self.outputs):
            return super().return_model()

        return Message
