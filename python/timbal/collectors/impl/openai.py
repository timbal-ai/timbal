from typing import Any, override

from openai.types.chat import ChatCompletionChunk
from uuid_extensions import uuid7

from ...state.context import RunContext
from ...types.message import Message
from .. import register_collector
from ..base import EventCollector

# Create a type alias for OpenAI events (in the future we may have more than one)
OpenAIEvent = ChatCompletionChunk


@register_collector
class OpenAICollector(EventCollector):
    """Collector for OpenAI streaming events."""
    
    def __init__(self, run_context: RunContext):
        super().__init__(run_context)
        self._content: str = ""
        self._tool_calls: list[dict[str, Any]] = []
        self._current_tool_call: dict[str, Any] | None = None
    
    @classmethod
    @override
    def can_handle(cls, event: Any) -> bool:
        return isinstance(event, OpenAIEvent)
    
    @override
    def process(self, event: OpenAIEvent) -> Any:
        """Processes OpenAI streaming events."""
        # Handle usage statistics
        if event.usage:
            self._handle_usage(event)

        if not len(event.choices):
            return None

        # Handle tool calls
        if event.choices[0].delta.tool_calls:
            return self._handle_tool_calls(event)

        # Handle text content
        if event.choices[0].delta.content:
            return self._handle_text_content(event)
    
    def _handle_usage(self, event: OpenAIEvent) -> None:
        """Handle usage statistics from OpenAI events."""
        openai_model = event.model
        openai_usage = event.usage

        input_tokens = int(openai_usage.prompt_tokens)
        input_tokens_details = openai_usage.prompt_tokens_details
        
        if hasattr(input_tokens_details, "cached_tokens"):
            input_cached_tokens = int(input_tokens_details.cached_tokens)
            if input_cached_tokens:
                input_tokens -= input_cached_tokens
                self._run_context.update_usage(f"{openai_model}:input_cached_tokens", input_cached_tokens)
        
        if hasattr(input_tokens_details, "audio_tokens"):
            input_audio_tokens = int(input_tokens_details.audio_tokens)
            if input_audio_tokens:
                input_tokens -= input_audio_tokens
                self._run_context.update_usage(f"{openai_model}:input_audio_tokens", input_audio_tokens)

        self._run_context.update_usage(f"{openai_model}:input_text_tokens", input_tokens)

        output_tokens = int(openai_usage.completion_tokens)
        output_tokens_details = openai_usage.completion_tokens_details
        
        if hasattr(output_tokens_details, "audio_tokens"):
            output_audio_tokens = int(output_tokens_details.audio_tokens)
            if output_audio_tokens:
                output_tokens -= output_audio_tokens
                self._run_context.update_usage(f"{openai_model}:output_audio_tokens", output_audio_tokens)

        self._run_context.update_usage(f"{openai_model}:output_text_tokens", output_tokens)
    
    def _handle_tool_calls(self, event: OpenAIEvent) -> None:
        """Handle tool call events from OpenAI."""
        tool_call = event.choices[0].delta.tool_calls[0]
        
        if tool_call.id:
            # Start new tool call
            self._current_tool_call = {
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.function.name,
                "input": ""
            }
            self._tool_calls.append(self._current_tool_call)
        else:
            # Continue current tool call
            if not self._current_tool_call:
                # TODO Review this
                # ? Gemini (via openai sdk) doesn't add an id to the tool call
                tool_call_id = f"tc_{uuid7(as_type='str').replace('-', '')}"
                self._current_tool_call = {
                    "type": "tool_use",
                    "id": tool_call_id,
                    "name": tool_call.function.name,
                    "input": ""
                }
                self._tool_calls.append(self._current_tool_call)
            
            self._current_tool_call["input"] += tool_call.function.arguments
        
        return None
    
    def _handle_text_content(self, event: OpenAIEvent) -> str:
        """Handle text content from OpenAI events."""
        text_chunk = event.choices[0].delta.content
        
        # Handle citations if present
        if hasattr(event, "citations"):
            self.citations = event.citations

        self._content += text_chunk
        return text_chunk
    
    @override
    def collect(self) -> Message:
        """Returns structured OpenAI response."""
        
        if self._content:
            return Message.validate({
                "role": "assistant",
                "content": self._content
            })
            
        if self._tool_calls:
            # Check if this is an output_model_tool call and intercept it
            for tool_call in self._tool_calls:
                if tool_call.get("name") == "output_model_tool":
                    # Convert to text message instead of tool call
                    return Message.validate({
                        "role": "assistant",
                        "content": [{
                            "type": "text",
                            "text": tool_call.get("input", "{}")
                        }]
                    })
            
            # Openai allows the use of custom IDs for tool calls. 
            # We choose to generate our own random IDs for consistency and to make sure they don't collide
            # (they are not transparent with the algs being used)
            tool_calls = [
                {**tc, "id": uuid7(as_type="str").replace("-", "")}
                for tc in self._tool_calls
            ]
            return Message.validate({
                "role": "assistant",
                "content": tool_calls
            })
            
        return None