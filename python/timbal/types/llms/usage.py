from anthropic.types.message import Message as AnthropicMessage
from openai.types.chat import ChatCompletion, ChatCompletionChunk


def acc_usage(
    acc: dict[str, int],
    model: str,
    llm_output: ChatCompletion | ChatCompletionChunk | AnthropicMessage,
) -> dict[str, int]:
    """Accumulate usage stats for a given model.

    Processes usage statistics from different LLM providers and accumulates them into
    a standardized format. Currently supports OpenAI and Anthropic usage formats.

    Args:
        acc: Current accumulated usage stats
        model: Name/ID of the LLM model
        llm_output: Response of the LLM sdk when creating a chat completion

    Returns:
        Updated Usage object with accumulated stats

    Supported formats:
        - openai.types.chat.ChatCompletion
        - anthropic.types.message.Message
    """
    if isinstance(llm_output, (ChatCompletion, ChatCompletionChunk)):
        openai_usage = llm_output.usage
        if openai_usage is None:
            return acc
        input_tokens = int(openai_usage.prompt_tokens)
        input_tokens_details = openai_usage.prompt_tokens_details

        if hasattr(input_tokens_details, "cached_tokens"):
            input_cached_tokens = int(input_tokens_details.cached_tokens)
            if input_cached_tokens:
                input_tokens -= input_cached_tokens
                input_cached_tokens_key = f"{model}:input_cached_tokens"
                existing_input_cached_tokens = acc.get(input_cached_tokens_key, 0)
                acc[input_cached_tokens_key] = existing_input_cached_tokens + input_cached_tokens

        if hasattr(input_tokens_details, "audio_tokens"):
            input_audio_tokens = int(input_tokens_details.audio_tokens)
            if input_audio_tokens:
                input_tokens -= input_audio_tokens
                input_audio_tokens_key = f"{model}:input_audio_tokens"
                existing_input_audio_tokens = acc.get(input_audio_tokens_key, 0)
                acc[input_audio_tokens_key] = existing_input_audio_tokens + input_audio_tokens

        # ? We've seen this in some responses. Usually they return the image tokens as regular text tokens.
        # if hasattr(input_tokens_details, "image_tokens"):
        #     input_image_tokens = int(input_tokens_details.image_tokens)
        #     if input_image_tokens:
        #         input_tokens -= input_image_tokens
        #         async_gen_state.usage[f"{openai_model}:input_image_tokens"] = input_image_tokens
        # Text tokens are used as the default.
        # if hasattr(input_tokens_details, "text_tokens"):
        input_text_tokens_key = f"{model}:input_text_tokens"
        existing_input_text_tokens = acc.get(input_text_tokens_key, 0)
        acc[input_text_tokens_key] = existing_input_text_tokens + input_tokens

        output_tokens = int(openai_usage.completion_tokens)
        output_tokens_details = openai_usage.completion_tokens_details

        if hasattr(output_tokens_details, "audio_tokens"):
            output_audio_tokens = int(output_tokens_details.audio_tokens)
            if output_audio_tokens:
                output_tokens -= output_audio_tokens
                output_audio_tokens_key = f"{model}:output_audio_tokens"
                existing_output_audio_tokens = acc.get(output_audio_tokens_key, 0)
                acc[output_audio_tokens_key] = existing_output_audio_tokens + output_audio_tokens

        if hasattr(output_tokens_details, "reasoning_tokens"):
            output_reasoning_tokens = int(output_tokens_details.reasoning_tokens)
            if output_reasoning_tokens:
                output_tokens -= output_reasoning_tokens
                output_reasoning_tokens_key = f"{model}:output_reasoning_tokens"
                existing_output_reasoning_tokens = acc.get(output_reasoning_tokens_key, 0)
                acc[output_reasoning_tokens_key] = existing_output_reasoning_tokens + output_reasoning_tokens

        output_text_tokens_key = f"{model}:output_text_tokens"
        existing_output_text_tokens = acc.get(output_text_tokens_key, 0)
        acc[output_text_tokens_key] = existing_output_text_tokens + output_tokens

    elif isinstance(llm_output, AnthropicMessage):
        anthropic_usage = llm_output.usage
        input_tokens_key = f"{model}:input_tokens"
        input_tokens = int(anthropic_usage.input_tokens)
        existing_input_tokens = acc.get(input_tokens_key, 0)
        acc[input_tokens_key] = existing_input_tokens + input_tokens

        output_tokens_key = f"{model}:output_tokens"
        output_tokens = int(anthropic_usage.output_tokens)
        existing_output_tokens = acc.get(output_tokens_key, 0)
        acc[output_tokens_key] = existing_output_tokens + output_tokens

        # cache_creation_input_tokens
        # cache_read_input_tokens

    else:
        raise NotImplementedError(f"Unsupported LLM output type: {type(llm_output)}")
    
    return acc
