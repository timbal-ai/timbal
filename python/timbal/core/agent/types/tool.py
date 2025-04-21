from typing import Literal

from pydantic import BaseModel, ConfigDict

from ...shared import RunnableLike


class Tool(BaseModel):
    """Wrapper to standardize defining a tool for an agent."""
    model_config = ConfigDict(extra="ignore")

    runnable: RunnableLike
    """The runnable to execute when the tool is used."""
    description: str | None = None
    """A description of the tool."""
    force_exit: bool = False
    """Whether to force the exit of the agent after the tool is used."""
    params_mode: Literal["all", "required"] = "all"
    """Whether to pass all parameters or only required parameters as the tool description to the LLM."""
    include_params: list[str] | None = None
    """If set, only include these parameters in the tool description."""
    exclude_params: list[str] | None = None
    """If set, exclude these parameters from the tool description."""
    