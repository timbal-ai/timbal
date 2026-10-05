"""Run artifacts end to end against a live Timbal platform.

Each conversation turn runs in its own process with its own ``$HOME``, the way the platform
runs every request in a fresh sandbox, so nothing local survives between turns. Exercises:
the artifacts endpoints, offload handles read back in a later turn (memory loaded from the
platform trace), a persisted file served signed in the trace and refused unsigned, and
another user being unable to read either.

Requires ``.env.test_run_artifacts`` in this directory:

    TIMBAL_API_HOST=api.dev.timbal.ai
    TIMBAL_API_KEY=<key of a user with projects.runs.write on the app's project>
    TIMBAL_ORG_ID=<org id>
    TIMBAL_APP_ID=<app id; test runs are recorded under it>
    TIMBAL_OTHER_API_KEY=<optional: another member of the org, for the isolation checks>

Skipped when the file is absent. Run with:
    uv run pytest python/tests/platform/test_artifacts_live.py -v
"""

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import httpx
import pytest

ENV_FILE = Path(__file__).parent / ".env.test_run_artifacts"
pytestmark = pytest.mark.skipif(not ENV_FILE.exists(), reason=f"needs {ENV_FILE.name}; see module docstring")

PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde"
    b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _env() -> dict[str, str]:
    values = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip()
    return values


def _set_platform(api_key: str) -> None:
    from timbal.state import set_run_context
    from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
    from timbal.state.context import RunContext

    env = _env()
    config = PlatformConfig(
        host=env["TIMBAL_API_HOST"],
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token=api_key),
        subject=PlatformSubject(org_id=env["TIMBAL_ORG_ID"], app_id=env["TIMBAL_APP_ID"]),
    )
    set_run_context(RunContext(platform_config=config, tracing_provider=None))


def _turn(script: str, tmp_path: Path, **args: str) -> dict:
    """Run one conversation turn in a fresh process with a throwaway ``$HOME``."""
    home = tmp_path / f"home-{uuid.uuid4().hex[:8]}"
    home.mkdir()
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("TIMBAL_")},
        **{k: v for k, v in _env().items() if k != "TIMBAL_OTHER_API_KEY"},
        "HOME": str(home),
        "TURN_ARGS": json.dumps(args),
    }
    proc = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


TURN_1 = r"""
import asyncio, json, os, re
from timbal.core.agent import Agent
from timbal.core.test_model import TestModel
from timbal.core.tool import Tool
from timbal.types.content import FileContent, TextContent, ToolUseContent
from timbal.types.file import File
from timbal.types.message import Message

PNG = bytes.fromhex(json.loads(os.environ["TURN_ARGS"])["png"])
seen = {}

def model(messages):
    if "handle" not in seen and messages[-1].role == "user" and not any(m.role == "tool" for m in messages):
        return Message(role="assistant", content=[ToolUseContent(id="t1", name="fetch", input={})], stop_reason="tool_use")
    text = "".join(c.content[0].text for m in messages if m.role == "tool" for c in m.content)
    seen["handle"] = re.search(r'read_offloaded\(handle="([^"]+)"\)', text).group(1)
    return "done"

agent = Agent(
    name="artifacts_e2e",
    model=TestModel(handler=model),
    tools=[Tool(name="fetch", handler=lambda: "".join(f"row-{i}\n" for i in range(5000)))],
    tool_result_limit=1_000,
)
prompt = Message(role="user", content=[TextContent(text="fetch it"), FileContent(file=File.validate(PNG), name="pixel.png")])

async def main():
    result = await agent(prompt=prompt).collect()
    assert result.status.code == "success", result.error
    print(json.dumps({"run_id": result.run_id, "handle": seen["handle"]}))

asyncio.run(main())
"""

TURN_2 = r"""
import asyncio, json, os
from timbal.core.agent import Agent
from timbal.core.test_model import TestModel
from timbal.types.content import ToolUseContent
from timbal.types.message import Message

args = json.loads(os.environ["TURN_ARGS"])
read = {}

def model(messages):
    tool_results = [c for m in messages if m.role == "tool" for c in m.content if getattr(c, "id", None) == "r1"]
    if not tool_results:
        call = ToolUseContent(id="r1", name="read_offloaded", input={"handle": args["handle"], "offset": 4999, "limit": 1})
        return Message(role="assistant", content=[call], stop_reason="tool_use")
    read["text"] = tool_results[0].content[0].text
    return "ok"

async def main():
    result = await Agent(name="artifacts_e2e", model=TestModel(handler=model), tools=[], tool_result_limit=1_000)(
        prompt="read it back", parent_id=args["run_id"]
    ).collect()
    assert result.status.code == "success", result.error
    print(json.dumps({"read": read["text"]}))

asyncio.run(main())
"""


@pytest.mark.asyncio
async def test_the_store_round_trips_through_the_platform() -> None:
    from timbal.core.tool_result_offload import PlatformOffloadStore
    from timbal.platform.artifacts import artifacts_available

    _set_platform(_env()["TIMBAL_API_KEY"])
    assert artifacts_available()
    store = PlatformOffloadStore()
    key = f"e2e/{uuid.uuid4().hex}"
    handle = await store.write(key, b"live payload")
    assert handle == key
    assert await store.read(handle) == b"live payload"
    assert await store.exists(handle) is True
    assert await store.exists(f"{key}-missing") is False

    other = _env().get("TIMBAL_OTHER_API_KEY")
    if other:
        _set_platform(other)
        with pytest.raises(FileNotFoundError):
            await PlatformOffloadStore().read(handle)


@pytest.mark.asyncio
async def test_a_later_turn_reads_an_offloaded_result_and_the_file_comes_back_signed(tmp_path) -> None:
    first = _turn(TURN_1, tmp_path, png=PNG.hex())
    second = _turn(TURN_2, tmp_path, run_id=first["run_id"], handle=first["handle"])
    assert "row-4999" in second["read"], second["read"]

    from timbal.platform.utils import _request

    env = _env()
    _set_platform(env["TIMBAL_API_KEY"])
    run = (
        await _request("GET", f"orgs/{env['TIMBAL_ORG_ID']}/apps/{env['TIMBAL_APP_ID']}/runs/{first['run_id']}")
    ).json()
    urls = [u for u in _strings(run.get("trace")) if "/artifacts/files/" in u]
    assert urls, "the persisted file should be an artifact URL in the trace"
    signed = urls[0]
    assert "Signature=" in signed
    async with httpx.AsyncClient(headers={"User-Agent": "timbal-run-artifacts-e2e"}) as client:
        assert (await client.get(signed)).content == PNG
        assert (await client.head(signed.split("?", 1)[0])).status_code == 403

    other = env.get("TIMBAL_OTHER_API_KEY")
    if other:
        from timbal.errors import PlatformError

        _set_platform(other)
        with pytest.raises(PlatformError) as denied:
            await _request("POST", f"orgs/{env['TIMBAL_ORG_ID']}/content/sign", json={"url": signed})
        assert denied.value.status_code == 403


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []
