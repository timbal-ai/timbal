from pydantic import BaseModel, ConfigDict

from ....types.field import Field
from ....types.message import Message


class BaseAgentParamsModel(BaseModel):
    """Base fixed parameter model for Agents. Used for internal validation."""
    model_config = ConfigDict(extra="ignore")

    system_prompt: str | None = Field(
        default=None,
        description="The system prompt to use for the agent."
    )
    model: str | None = Field(
        default="gpt-4.1-nano",
        description="The model to use for the agent."
    )
    max_tokens: int | None = Field(
        default=None,
        description="The maximum number of tokens to generate."
    )
    stream: bool | None = Field(
        default=False,
        description="Whether to stream the output."
    )


class AgentParamsModel(BaseAgentParamsModel):
    """Fixed parameter model for Agents."""

    prompt: Message = Field(description="The prompt to use for the agent.")


agent_params_model_schema = AgentParamsModel.model_json_schema()
