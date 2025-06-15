import copy
from typing import Any, override

from openai.types.chat import ChatCompletionChunk
from pydantic import Field

from ...types.message import Message
from ..context import get_run_context
from .base import BaseCollector


class DefaultCollector(BaseCollector):
    """"""

    chunks: list[Any] = Field(
        default_factory=list,
        description="The chunks collected by the collector.",
    )
    source: str | None = Field(
        None,
        description="The source of the chunks collected by the collector.",
    )


    @override
    def handle_chunk(self, chunk: Any) -> Any | None:
        """"""
        run_context = get_run_context()
        if isinstance(chunk, ChatCompletionChunk):
            self.source = "openai"
            openai_event = chunk
            # TODO Review. Gemini sends usage stats in every chunk. It doesn't send a separate chunk with the total usage.
            # ? We guess they send the revised number in the last chunk.
            if openai_event.usage:
                openai_model = openai_event.model
                openai_usage = openai_event.usage

                input_tokens = int(openai_usage.prompt_tokens)
                input_tokens_details = openai_usage.prompt_tokens_details
                if hasattr(input_tokens_details, "cached_tokens"):
                    input_cached_tokens = int(input_tokens_details.cached_tokens)
                    if input_cached_tokens:
                        input_tokens -= input_cached_tokens
                        run_context.update_usage(f"{openai_model}:input_cached_tokens", input_cached_tokens)
                if hasattr(input_tokens_details, "audio_tokens"):
                    input_audio_tokens = int(input_tokens_details.audio_tokens)
                    if input_audio_tokens:
                        input_tokens -= input_audio_tokens
                        run_context.update_usage(f"{openai_model}:input_audio_tokens", input_audio_tokens)

                run_context.update_usage(f"{openai_model}:input_text_tokens", input_tokens)

                output_tokens = int(openai_usage.completion_tokens)
                output_tokens_details = openai_usage.completion_tokens_details
                if hasattr(output_tokens_details, "audio_tokens"):
                    output_audio_tokens = int(output_tokens_details.audio_tokens)
                    if output_audio_tokens:
                        output_tokens -= output_audio_tokens
                        run_context.update_usage(f"{openai_model}:output_audio_tokens", output_audio_tokens)

                run_context.update_usage(f"{openai_model}:output_text_tokens", output_tokens)

            if not len(openai_event.choices):
                return None

            if openai_event.choices[0].delta.tool_calls:
                tool_call = copy.deepcopy(openai_event.choices[0].delta.tool_calls[0])
                if tool_call.id:
                    chunk_result = {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.function.name,
                        "input": ""
                    }
                    self.chunks.append(chunk_result)
                else:
                    # ? Gemini doesn't add an id to the tool call.
                    if not self.chunks:
                        tool_use_id = f"call_{uuid7(as_type='str').replace('-', '')}"
                        self.chunks.append({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": tool_call.function.name,
                            "input": ""
                        })
                    self.chunks[-1]["input"] += tool_call.function.arguments
                return None

            if openai_event.choices[0].delta.content:
                text_chunk = openai_event.choices[0].delta.content
                chunk_result = {
                    "type": "text",
                    "text": text_chunk,
                }

                # Extra is allowed in the async gen pydantic model.
                if hasattr(openai_event, "citations"):
                    self.citations = openai_event.citations

                if self.chunks:
                    self.chunks[-1]["text"] += text_chunk
                else:
                    self.chunks.append(chunk_result)
                return text_chunk
        else:
            return None

    @override
    def collect(self) -> Any:
        """"""
        if self.source == "openai":
            return Message.validate({
                "role": "assistant",
                "content": self.chunks,
            })
        return self.chunks
