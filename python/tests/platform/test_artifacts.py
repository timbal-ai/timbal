"""Run artifacts: the platform-backed offload store, durable file persistence, and requests
that survive attached files whose source is gone."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from timbal.core.attachment_limit import ATTACHMENT_UNAVAILABLE_MARKER, AttachmentSpills, load_attachments
from timbal.core.tool_result_offload import LocalOffloadStore, PlatformOffloadStore
from timbal.errors import PlatformError
from timbal.state import set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.context import RunContext
from timbal.types.content import FileContent, TextContent
from timbal.types.file import File
from timbal.types.message import Message

ARTIFACT_URL = "https://timbalusercontent.com/orgs/1/projects/5/users/9/artifacts/files/abc/report.pdf"


def _platform(app_id: str | None = "7") -> None:
    config = PlatformConfig(
        host="api.timbal.ai",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="t"),
        subject=PlatformSubject(org_id="1", app_id=app_id),
    )
    set_run_context(RunContext(platform_config=config, tracing_provider=None))


def _response(status: int = 200, *, content: bytes = b"", json: dict | None = None) -> MagicMock:
    res = MagicMock()
    res.status_code = status
    res.content = content
    res.json.return_value = json or {}
    return res


class FakePlatform:
    """In-memory stand-in for the artifacts endpoints, patched over ``_request``."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []
        self.down = False

    async def request(self, method, path, content=None, **_):
        self.calls.append((method, path))
        if self.down:
            raise PlatformError("not found", status_code=404)
        prefix = "orgs/1/apps/7/artifacts/"
        if path == "files" and method == "POST":
            return _response(
                json={
                    "url": "https://timbalusercontent.com/tmp/x/legacy.bin",
                    "name": "legacy.bin",
                    "content_type": "a/b",
                    "content_length": 1,
                    "created_at": "2026-01-01T00:00:00Z",
                    "expires_at": None,
                }
            )
        assert path.startswith(prefix), path
        key = path[len(prefix) :]
        if method == "PUT":
            self.objects[key] = content
            return _response(
                json={"handle": key, "url": f"https://timbalusercontent.com/orgs/1/projects/5/users/9/artifacts/{key}"}
            )
        if key not in self.objects:
            raise PlatformError("not found", status_code=404)
        return _response(content=self.objects[key])


@pytest.fixture
def platform():
    fake = FakePlatform()
    with patch("timbal.platform.utils._request", new=fake.request):
        yield fake


class TestPlatformOffloadStore:
    @pytest.mark.asyncio
    async def test_without_an_app_subject_it_is_the_local_store(self, tmp_path, platform) -> None:
        _platform(app_id=None)
        store = PlatformOffloadStore(fallback=LocalOffloadStore(root=tmp_path))
        handle = await store.write("run/c1", b"local")
        assert await store.read(handle) == b"local"
        assert (tmp_path / handle).is_file()
        assert platform.calls == []

    @pytest.mark.asyncio
    async def test_platform_runs_write_and_read_artifacts(self, tmp_path, platform) -> None:
        _platform()
        store = PlatformOffloadStore(fallback=LocalOffloadStore(root=tmp_path))
        handle = await store.write("run 1/call:1", b"payload")
        assert handle == "run_1/call_1"
        assert platform.objects == {"run_1/call_1": b"payload"}
        assert await store.read(handle) == b"payload"
        assert await store.exists(handle) is True
        assert await store.exists("run_1/missing") is False
        assert not any(tmp_path.iterdir())

    @pytest.mark.asyncio
    @pytest.mark.usefixtures("platform")
    async def test_a_missing_artifact_is_file_not_found(self, tmp_path) -> None:
        _platform()
        store = PlatformOffloadStore(fallback=LocalOffloadStore(root=tmp_path))
        with pytest.raises(FileNotFoundError):
            await store.read("run/gone")

    @pytest.mark.asyncio
    async def test_a_platform_without_artifacts_still_offloads_locally(self, tmp_path, platform) -> None:
        _platform()
        platform.down = True
        store = PlatformOffloadStore(fallback=LocalOffloadStore(root=tmp_path))
        handle = await store.write("run/c1", b"kept")
        assert await store.read(handle) == b"kept"
        assert await store.exists(handle) is True

    def test_agents_default_to_it(self) -> None:
        from timbal.core.agent import Agent
        from timbal.core.test_model import TestModel

        agent = Agent(name="a", model=TestModel(responses=["ok"]), tools=[], tool_result_limit=1_000)
        assert isinstance(agent._offload_store, PlatformOffloadStore)


class TestAttachmentSpillsProbe:
    @pytest.mark.asyncio
    async def test_an_existing_spill_is_probed_not_downloaded(self, tmp_path, platform) -> None:
        _platform()
        store = PlatformOffloadStore(fallback=LocalOffloadStore(root=tmp_path))
        text = "x" * 50
        first = await AttachmentSpills().handle(store, text)
        platform.calls.clear()
        again = await AttachmentSpills().handle(store, text)  # a fresh process
        assert again == first
        assert [m for m, _ in platform.calls] == ["HEAD"]


def _file_transport(status_by_url: dict[str, int]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        status = status_by_url.get(str(request.url), 200)
        return httpx.Response(status, content=b"%PDF-1.4" if status == 200 else b"", request=request)

    return httpx.MockTransport(handler)


def _file_client(status_by_url: dict[str, int]) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=_file_transport(status_by_url))


class TestLoadAttachments:
    @pytest.mark.asyncio
    async def test_a_file_that_is_gone_becomes_a_note(self) -> None:
        gone, fine = "https://cdn.example/gone.pdf", "https://cdn.example/fine.pdf"
        messages = [
            Message(role="user", content=[TextContent(text="old turn"), FileContent(file=File(gone), name="gone.pdf")]),
            Message(role="user", content=[FileContent(file=File(fine), name="fine.pdf")]),
        ]
        with patch("timbal.core.llm.clients._get_file_client", return_value=_file_client({gone: 404})):
            out = await load_attachments(messages)

        assert out[1] is messages[1]
        assert out[0].content[0].text == "old turn"
        note = out[0].content[1].text
        assert note.startswith(ATTACHMENT_UNAVAILABLE_MARKER) and '"gone.pdf"' in note and "404" in note
        assert isinstance(messages[0].content[1], FileContent)  # memory keeps the file

    @pytest.mark.asyncio
    async def test_transient_failures_still_raise(self) -> None:
        url = "https://cdn.example/flaky.pdf"
        messages = [Message(role="user", content=[FileContent(file=File(url))])]
        with patch("timbal.core.llm.clients._get_file_client", return_value=_file_client({url: 503})):
            with pytest.raises(httpx.HTTPStatusError):
                await load_attachments(messages)


class TestFilePersistArtifacts:
    @pytest.mark.asyncio
    async def test_platform_runs_persist_files_as_private_artifacts(self, platform) -> None:
        _platform()
        file = File.validate(b"%PDF-1.4 report")
        url = await file.persist()
        ((method, path),) = platform.calls
        assert method == "PUT" and path.startswith("orgs/1/apps/7/artifacts/files/")
        assert url.startswith("https://timbalusercontent.com/orgs/1/projects/5/users/9/artifacts/files/")
        assert file.__persisted__ == url

    @pytest.mark.asyncio
    async def test_a_signed_artifact_url_is_kept_in_its_stable_form(self, platform) -> None:
        _platform()
        file = File.validate(f"{ARTIFACT_URL}?Expires=1&Signature=x&Key-Pair-Id=k")
        assert await file.persist() == ARTIFACT_URL
        assert platform.calls == []

    @pytest.mark.asyncio
    async def test_a_public_upload_is_moved_into_artifacts(self, platform) -> None:
        _platform()
        upload = "https://timbalusercontent.com/tmp/019f/Invoice%2042.pdf"
        file = File.validate(upload)
        real_client, transport = httpx.AsyncClient, _file_transport({})
        with patch("httpx.AsyncClient", side_effect=lambda **kw: real_client(transport=transport, **kw)):
            url = await file.persist()
        assert url != upload and "/artifacts/files/" in url
        (key,) = platform.objects
        assert key.endswith("/Invoice_42.pdf") and platform.objects[key] == b"%PDF-1.4"

    @pytest.mark.asyncio
    async def test_a_platform_without_artifacts_keeps_temporary_uploads(self, platform) -> None:
        _platform()

        async def put_fails(method, path, **kw):
            if method == "PUT":
                raise PlatformError("not found", status_code=404)
            return await platform.request(method, path, **kw)

        with patch("timbal.platform.utils._request", new=AsyncMock(side_effect=put_fails)):
            url = await File.validate(b"data").persist()
        assert url == "https://timbalusercontent.com/tmp/x/legacy.bin"

    @pytest.mark.asyncio
    async def test_runs_without_an_app_subject_keep_temporary_uploads(self, platform) -> None:
        _platform(app_id=None)
        url = await File.validate(b"data").persist()
        assert url == "https://timbalusercontent.com/tmp/x/legacy.bin"
        assert [m for m, _ in platform.calls] == ["POST"]
