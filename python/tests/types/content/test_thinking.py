"""ThinkingContent: Anthropic signature vs OpenAI Responses reasoning item replay."""

from timbal.types.content import ThinkingContent, ToolUseContent, content_factory
from timbal.types.message import Message


class TestOpenAIResponsesReplay:
    def test_assistant_reasoning_item_with_summary(self):
        c = ThinkingContent(thinking="plan the tool call", id="rs_1", encrypted_content="enc-1")
        assert c.is_openai_reasoning_item
        assert c.to_openai_responses_input(role="assistant") == {
            "type": "reasoning",
            "id": "rs_1",
            "encrypted_content": "enc-1",
            "summary": [{"type": "summary_text", "text": "plan the tool call"}],
        }

    def test_assistant_reasoning_item_without_summary_has_empty_summary_list(self):
        """`summary` is required by the API; no visible text means `[]`, never a fake entry."""
        c = ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1")
        item = c.to_openai_responses_input(role="assistant")
        assert item["type"] == "reasoning"
        assert item["summary"] == []
        assert item["encrypted_content"] == "enc-1"

    def test_encrypted_content_without_id_is_not_a_reasoning_item(self):
        """A reasoning item needs its `rs_…` id; without it fall back to the text behaviour."""
        c = ThinkingContent(thinking="visible", encrypted_content="enc-1")
        assert not c.is_openai_reasoning_item
        assert c.to_openai_responses_input(role="assistant") == {"type": "output_text", "text": "visible"}

    def test_id_without_encrypted_content_is_not_a_reasoning_item(self):
        c = ThinkingContent(thinking="visible", id="rs_1")
        assert not c.is_openai_reasoning_item
        assert c.to_openai_responses_input(role="assistant") == {"type": "output_text", "text": "visible"}

    def test_legacy_assistant_text_preserved(self):
        c = ThinkingContent(thinking="raw reasoning text")
        assert c.to_openai_responses_input(role="assistant") == {"type": "output_text", "text": "raw reasoning text"}

    def test_legacy_user_text_preserved(self):
        c = ThinkingContent(thinking="raw reasoning text")
        assert c.to_openai_responses_input(role="user") == {"type": "input_text", "text": "raw reasoning text"}

    def test_empty_legacy_thinking_is_dropped(self):
        """An empty text part would be rejected upstream; return None so the message skips it."""
        assert ThinkingContent(thinking="").to_openai_responses_input(role="assistant") is None
        assert ThinkingContent(thinking="").to_openai_responses_input(role="user") is None

    def test_user_role_never_emits_a_reasoning_item(self):
        """Reasoning items are model output; a user-role block with the fields stays a text part."""
        c = ThinkingContent(thinking="t", id="rs_1", encrypted_content="enc-1")
        assert c.to_openai_responses_input(role="user") == {"type": "input_text", "text": "t"}


class TestOtherProvidersUnaffected:
    def test_anthropic_input_ignores_openai_fields(self):
        c = ThinkingContent(thinking="t", signature="sig", id="rs_1", encrypted_content="enc-1")
        assert c.to_anthropic_input() == {"type": "thinking", "thinking": "t", "signature": "sig"}

    def test_anthropic_input_drops_thinking_without_a_signature(self):
        """A cross-provider fallback (OpenAI → Anthropic) must not send an unverifiable block."""
        assert ThinkingContent(thinking="t", id="rs_1", encrypted_content="enc-1").to_anthropic_input() is None
        assert ThinkingContent(thinking="raw xai reasoning").to_anthropic_input() is None

    def test_anthropic_message_skips_openai_reasoning_items(self):
        from timbal.types.content import TextContent, ToolUseContent

        msg = Message(
            role="assistant",
            content=[
                ThinkingContent(thinking="", id="rs_1", encrypted_content="enc-1"),
                ToolUseContent(id="call_1", name="search", input={"q": "x"}),
                TextContent(text="ok"),
            ],
        )
        blocks = msg.to_anthropic_input()["content"]
        assert [b["type"] for b in blocks] == ["tool_use", "text"]

    def test_anthropic_message_keeps_signed_thinking(self):
        msg = Message(role="assistant", content=[ThinkingContent(thinking="t", signature="sig")])
        assert msg.to_anthropic_input()["content"] == [{"type": "thinking", "thinking": "t", "signature": "sig"}]

    def test_chat_completions_input_is_none(self):
        c = ThinkingContent(thinking="t", id="rs_1", encrypted_content="enc-1")
        assert c.to_openai_chat_completions_input() is None


class TestPersistence:
    def test_content_factory_round_trip_keeps_reasoning_fields(self):
        original = ThinkingContent(thinking="t", signature="sig", id="rs_1", encrypted_content="enc-1")
        rebuilt = content_factory(original.model_dump())
        assert rebuilt == original

    def test_content_factory_defaults(self):
        rebuilt = content_factory({"type": "thinking", "thinking": "t"})
        assert rebuilt == ThinkingContent(thinking="t")
        assert rebuilt.id is None and rebuilt.encrypted_content is None

    def test_content_factory_missing_thinking_becomes_empty_string(self):
        rebuilt = content_factory({"type": "thinking", "id": "rs_1", "encrypted_content": "enc-1"})
        assert rebuilt.thinking == ""
        assert rebuilt.is_openai_reasoning_item

    def test_message_json_round_trip(self):
        """Memory is persisted as JSON (traces, sessions); a reload must keep the payload."""
        from pydantic import TypeAdapter

        adapter = TypeAdapter(Message)
        msg = Message(
            role="assistant",
            content=[
                ThinkingContent(thinking="t", id="rs_1", encrypted_content="enc-1"),
                ToolUseContent(id="call_1", name="search", input={"q": "x"}),
            ],
        )
        reloaded = adapter.validate_json(adapter.dump_json(msg))
        assert reloaded.content[0] == msg.content[0]
        assert reloaded.content[0].encrypted_content == "enc-1"
        items = reloaded.to_openai_responses_input()
        assert [i["type"] for i in items] == ["reasoning", "function_call"]
        assert items[0]["encrypted_content"] == "enc-1"
