"""The lease owner is per process, also for workers forked from a process that already imported the store."""

import json
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent(
    """
    import asyncio, json, os, sys
    from timbal.state import chat_store
    from timbal.state.chat_store import SessionLeased, SessionSnapshot, SqliteChatSessionStore, process_owner

    path = sys.argv[1]
    parent = process_owner()  # resolved before the fork, as a server that imports Timbal and then forks does

    async def seed():
        store = SqliteChatSessionStore(path)
        await store.save(SessionSnapshot(session_id="s1", agent="agent"), expect_version=None)

    asyncio.run(seed())

    async def worker():
        store = SqliteChatSessionStore(path)
        try:
            await store.acquire_lease("s1", process_owner())
            return "held"
        except SessionLeased:
            return "refused"

    results = []
    for _ in range(2):
        read, write = os.pipe()
        pid = os.fork()
        if pid == 0:
            os.close(read)
            outcome = asyncio.run(worker())
            os.write(write, json.dumps({"owner": process_owner(), "attr": chat_store.PROCESS_OWNER,
                                        "lease": outcome}).encode())
            os.close(write)
            if outcome == "held":
                import time; time.sleep(1.0)  # a live worker, still holding its lease while the next one asks
            os._exit(0)
        os.close(write)
        results.append((pid, json.loads(os.read(read, 4096).decode())))
    for pid, _ in results:
        os.waitpid(pid, 0)
    print(json.dumps({"parent": parent, "children": [result for _, result in results]}))
    """
)


@pytest.mark.skipif(sys.platform == "win32", reason="os.fork is POSIX only")
def test_forked_workers_each_get_their_own_owner_and_exclude_each_other(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT, str(tmp_path / "s.db")], capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr[-2000:]
    report = json.loads(out.stdout.strip().splitlines()[-1])
    first, second = report["children"]
    owners = {report["parent"], first["owner"], second["owner"]}
    assert len(owners) == 3, report  # not the parent's identity handed down to both
    assert first["attr"] == first["owner"] and second["attr"] == second["owner"]  # the module attribute too
    # The first forked worker holds the session; the second, a different live process, is refused.
    assert (first["lease"], second["lease"]) == ("held", "refused"), report
