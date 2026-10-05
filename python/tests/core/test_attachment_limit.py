"""Tests for request-time bounding of oversized file attachments."""

import asyncio
import json
import os
import re
from datetime import timedelta

import pytest
from timbal.core.agent import Agent
from timbal.core.attachment_limit import (
    ATTACHMENT_OFFLOAD_MARKER,
    ATTACHMENT_TOO_LARGE_MARKER,
    AttachmentSpills,
    apply_attachment_limit,
    attachment_text,
    bound_unfittable_attachments,
)
from timbal.core.memory_compaction import summarize
from timbal.core.models import get_context_window
from timbal.core.test_model import TestModel
from timbal.core.tool_result_offload import LocalOffloadStore, Spill, ToolResultLimit, Truncate
from timbal.types.content import FileContent, TextContent, ToolResultContent, ToolUseContent
from timbal.types.file import File
from timbal.types.message import Message

_HANDLE_RE = re.compile(r'read_offloaded\(handle="([^"]+)"\)')
_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _har(entries: int) -> str:
    return json.dumps(
        {"log": {"entries": [{"request": {"url": f"https://api.example/v1/{i}"}} for i in range(entries)]}}
    )


def _file(tmp_path, name: str, data: bytes | str) -> FileContent:
    path = tmp_path / name
    path.write_bytes(data.encode() if isinstance(data, str) else data)
    return FileContent(file=File(str(path)), name=name)


def _user(*content) -> Message:
    return Message(role="user", content=list(content))


def _texts(message: Message) -> list[str]:
    return [c.text for c in message.content if isinstance(c, TextContent)]


class TestAttachmentText:
    @pytest.mark.asyncio
    async def test_only_files_sent_as_pasted_text_are_measured(self, tmp_path) -> None:
        text = _file(tmp_path, "notes.txt", "hello world")
        image = _file(tmp_path, "shot.png", _PNG)
        pdf = _file(tmp_path, "brief.pdf", b"%PDF-1.4\n%%EOF")
        for item in (text, image, pdf):
            await item.file.load()
        assert attachment_text(text) == "hello world"
        assert attachment_text(image) is None
        assert attachment_text(pdf) is None

    @pytest.mark.asyncio
    async def test_below_the_threshold_is_none(self, tmp_path) -> None:
        item = _file(tmp_path, "small.json", '{"ok": true}')
        await item.file.load()
        assert attachment_text(item, threshold=1_000) is None
        assert attachment_text(item, threshold=5) == '{"ok": true}'


class TestApplyAttachmentLimit:
    @pytest.mark.asyncio
    async def test_a_large_text_attachment_is_spilled_and_readable(self, tmp_path) -> None:
        har = _har(2_000)
        messages = [_user(TextContent(text="check this capture"), _file(tmp_path, "session 2.har", har))]
        limit = ToolResultLimit(threshold=10_000, action=Spill(preview_chars=500))
        store = LocalOffloadStore(root=tmp_path / "offload")

        out = await apply_attachment_limit(messages, limit=limit, store=store, spills=AttachmentSpills())

        assert out is not messages and out[0] is not messages[0]
        assert isinstance(messages[0].content[1], FileContent)  # the input is never modified
        stand_in = out[0].content[1].text
        assert stand_in.startswith(ATTACHMENT_OFFLOAD_MARKER) and '"session 2.har"' in stand_in
        assert "Preview (first 500 of" in stand_in and len(stand_in) < 1_500
        handle = _HANDLE_RE.search(stand_in).group(1)
        assert (await store.read(handle)).decode() == har

    @pytest.mark.asyncio
    async def test_small_files_images_and_pdfs_pass_through_untouched(self, tmp_path) -> None:
        messages = [
            _user(
                _file(tmp_path, "small.json", '{"ok": true}'),
                _file(tmp_path, "shot.png", _PNG),
                _file(tmp_path, "brief.pdf", b"%PDF-1.4\n" + b"0" * 50_000),
            ),
            Message(role="assistant", content=[TextContent(text="seen")]),
        ]
        limit = ToolResultLimit(threshold=1_000)
        out = await apply_attachment_limit(messages, limit=limit, store=None, spills=AttachmentSpills())
        assert out == messages and all(a is b for a, b in zip(out, messages, strict=True))

    @pytest.mark.asyncio
    async def test_the_stand_in_is_identical_on_every_call(self, tmp_path) -> None:
        """Prompt caching: the same attachment must produce the same bytes each request,
        and be persisted once."""
        store = LocalOffloadStore(root=tmp_path / "offload")
        limit = ToolResultLimit(threshold=1_000)
        spills = AttachmentSpills()
        first = await apply_attachment_limit(
            [_user(_file(tmp_path, "a.har", _har(500)))], limit=limit, store=store, spills=spills
        )
        again = await apply_attachment_limit(
            [_user(_file(tmp_path, "a.har", _har(500)))], limit=limit, store=store, spills=AttachmentSpills()
        )
        assert _texts(first[0]) == _texts(again[0])
        # A fresh AttachmentSpills (a new process) finds the persisted copy instead of writing another.
        assert len([p for p in (tmp_path / "offload").rglob("*") if p.is_file()]) == 1

    @pytest.mark.asyncio
    async def test_pruned_attachment_is_restored_with_the_same_stand_in(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path / "offload")
        spills = AttachmentSpills()
        limit = ToolResultLimit(threshold=1_000)
        payload = _har(500)
        messages = [_user(_file(tmp_path, "capture.har", payload))]
        first = await apply_attachment_limit(messages, limit=limit, store=store, spills=spills)
        handle = _HANDLE_RE.search(first[0].content[0].text).group(1)
        os.utime(store.root / handle, (0, 0))
        store.cleanup_after = timedelta(hours=1)
        store._prune(store.root)
        with pytest.raises(FileNotFoundError):
            await store.read(handle)

        restored = await apply_attachment_limit(messages, limit=limit, store=store, spills=spills)

        assert _texts(restored[0]) == _texts(first[0])
        assert (await store.read(handle)).decode() == payload
        assert isinstance(messages[0].content[0], FileContent)

    @pytest.mark.asyncio
    async def test_opaque_handles_are_reused_and_restored_after_eviction(self) -> None:
        class OpaqueStore:
            def __init__(self):
                self.payloads = {}
                self.writes = 0

            async def write(self, _key, data):
                self.writes += 1
                handle = f"opaque-{self.writes}"
                self.payloads[handle] = data
                return handle

            async def read(self, handle):
                if handle not in self.payloads:
                    raise FileNotFoundError(handle)
                return self.payloads[handle]

        store = OpaqueStore()
        spills = AttachmentSpills()
        handles = await asyncio.gather(*(spills.handle(store, "payload") for _ in range(5)))
        assert handles == ["opaque-1"] * 5
        assert store.writes == 1
        store.payloads.clear()

        restored = await spills.handle(store, "payload")

        assert restored == "opaque-2"
        assert await store.read(restored) == b"payload"
        assert await spills.handle(store, "payload") == restored
        assert store.writes == 2

    @pytest.mark.asyncio
    async def test_truncate_and_missing_store_never_paste_the_whole_file(self, tmp_path) -> None:
        big = "line\n" * 20_000
        for limit, store in (
            (ToolResultLimit(threshold=1_000, action=Truncate(max_chars=300)), None),
            (ToolResultLimit(threshold=1_000, action=Spill()), None),  # Spill with no store → fallback
            (ToolResultLimit(threshold=1_000, action=Spill(fallback=None)), None),
        ):
            out = await apply_attachment_limit(
                [_user(_file(tmp_path, "build.log", big))], limit=limit, store=store, spills=AttachmentSpills()
            )
            text = out[0].content[0].text
            assert text.startswith('[Attached file "build.log"') and "truncated" in text
            assert len(text) < 3_000


class TestBoundUnfittableAttachments:
    @pytest.mark.asyncio
    async def test_a_file_that_cannot_fit_the_window_is_cut_to_a_preview(self, tmp_path) -> None:
        model = "openai/gpt-5.4-nano"
        window = get_context_window(model)
        assert window
        huge = _file(tmp_path, "dump.json", "x" * (window * 4 + 10))
        fits = _file(tmp_path, "ok.json", "y" * 1_000)
        for item in (huge, fits):
            await item.file.load()
        messages = [_user(TextContent(text="read these"), huge, fits)]

        out = bound_unfittable_attachments(messages, model)

        assert out[0].content[0].text == "read these"
        assert out[0].content[1].text.startswith(ATTACHMENT_TOO_LARGE_MARKER)
        assert len(out[0].content[1].text) < 21_000
        assert out[0].content[2] is fits
        assert isinstance(messages[0].content[1], FileContent)

    @pytest.mark.asyncio
    async def test_unknown_windows_and_fitting_files_are_left_alone(self, tmp_path) -> None:
        item = _file(tmp_path, "dump.json", "x" * 50_000)
        await item.file.load()
        messages = [_user(item)]
        assert bound_unfittable_attachments(messages, "openai/not-a-real-model") is messages
        assert bound_unfittable_attachments(messages, "openai/gpt-5.4-nano") is messages


class TestAgentAttachmentLimit:
    @pytest.mark.asyncio
    async def test_the_model_sees_a_stand_in_reads_it_back_and_memory_keeps_the_file(self, tmp_path) -> None:
        har = _har(3_000)
        seen: list[list[Message]] = []

        def model_handler(messages):
            seen.append(messages)
            if len(seen) == 1:
                handle = _HANDLE_RE.search(messages[-1].content[1].text).group(1)
                return Message(
                    role="assistant",
                    content=[ToolUseContent(id="r1", name="read_offloaded", input={"handle": handle, "limit": 1})],
                    stop_reason="tool_use",
                )
            return "done"

        agent = Agent(
            name="attach_agent",
            model=TestModel(handler=model_handler),
            tools=[],
            attachment_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path / "offload")),
        )
        prompt = Message(role="user", content=[TextContent(text="what's in it?"), _file(tmp_path, "s.har", har)])
        result = await agent(prompt=prompt).collect()
        assert result.status.code == "success", result.error

        first_request = seen[0][-1]
        assert first_request.content[1].text.startswith(ATTACHMENT_OFFLOAD_MARKER)
        read_back = [c for m in seen[1] if m.role == "tool" for c in m.content if isinstance(c, ToolResultContent)]
        assert read_back and "api.example" in read_back[0].content[0].text

    @pytest.mark.asyncio
    async def test_an_attachment_already_in_memory_is_bounded_on_the_next_turn(self, tmp_path) -> None:
        """A conversation that took a large file before the limit existed is fixed on its next
        call: inherited memory is bounded per request, without rewriting stored history."""
        har = _har(3_000)
        before = Agent(name="chat", model=TestModel(responses=["got it"]), tools=[])
        turn1 = await before(
            prompt=Message(role="user", content=[TextContent(text="here"), _file(tmp_path, "s.har", har)])
        ).collect()
        assert turn1.status.code == "success"

        seen: list[list[Message]] = []
        after = Agent(
            name="chat",
            model=TestModel(handler=lambda messages: seen.append(messages) or "ok"),
            tools=[],
            attachment_limit=ToolResultLimit(threshold=10_000, store=LocalOffloadStore(root=tmp_path / "offload")),
        )
        turn2 = await after(prompt="and now?", parent_id=turn1.run_id).collect()
        assert turn2.status.code == "success", turn2.error
        inherited = seen[0][0]
        assert inherited.role == "user" and inherited.content[1].text.startswith(ATTACHMENT_OFFLOAD_MARKER)

    def test_a_compactor_store_stays_the_one_read_offloaded_reads(self, tmp_path) -> None:
        store = LocalOffloadStore(root=tmp_path / "compactor")
        agent = Agent(
            name="compacting",
            model=TestModel(responses=["ok"]),
            tools=[],
            memory_compaction=summarize(threshold=50, store=store),
            attachment_limit=10_000,
        )
        assert agent._offload_store is store

    @pytest.mark.asyncio
    async def test_default_agents_send_attachments_as_before(self, tmp_path) -> None:
        seen: list[list[Message]] = []
        agent = Agent(name="plain", model=TestModel(handler=lambda m: seen.append(m) or "ok"), tools=[])
        item = _file(tmp_path, "data.json", "z" * 50_000)
        await agent(prompt=Message(role="user", content=[TextContent(text="x"), item])).collect()
        assert isinstance(seen[0][-1].content[1], FileContent)
        assert agent._read_offloaded is None
