from pydantic import BaseModel, ConfigDict

from ....types.field import Field
from ....types.message import Message


# TODO One will be for printing. The other one for validation.
class AgentParamsModel(BaseModel):
    """Fixed parameter model for Agents."""
    model_config = ConfigDict(extra="ignore")

    system_prompt: str = Field(
        default=None,
        description="The system prompt to use for the agent."
    )
    model: str = Field(
        default="gpt-4.1-nano",
        description="The model to use for the agent."
    )
    max_tokens: int = Field(
        default=None,
        description="The maximum number of tokens to generate."
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the output."
    )


class AgentParamsModel2(AgentParamsModel):
    """Fixed parameter model for Agents."""

    prompt: Message = Field(description="The prompt to use for the agent.")



if __name__ == "__main__":
    print(AgentParamsModel.model_json_schema())
    print(AgentParamsModel2.model_json_schema())