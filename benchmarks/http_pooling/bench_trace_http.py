"""Actual CLI -> Agent -> PlatformTracingProvider -> HTTPS benchmark.

Uses a local TLS server with a temporary trusted certificate, a dummy token,
and TestModel. No production traces or model calls. Fresh mode disables only
the new runner HTTP scope. Both modes persist both snapshots before returning.
An optional per-connection delay is SYNTHETIC and is reported separately.
"""

# ruff: noqa: T201

import argparse
import asyncio
import contextlib
import io
import json
import os
import platform
import ssl
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from timbal import Agent
from timbal.codegen.test import run_test
from timbal.core.test_model import TestModel
from timbal.platform._http_session import platform_http_session
from timbal.state import RunContext, get_run_context, set_run_context
from timbal.state.config import PlatformAuth, PlatformAuthType, PlatformConfig, PlatformSubject
from timbal.state.tracing.providers.platform import PlatformTracingProvider


async def benchmark(samples, connection_delay, cert_dir, close_after_response=False):
    connections = 0
    active = 0
    idle = asyncio.Event()
    idle.set()
    writes = []
    errors = []

    async def handle(reader, writer):
        nonlocal connections, active
        connections += 1
        active += 1
        idle.clear()
        try:
            if connection_delay:
                await asyncio.sleep(connection_delay)
            while True:
                try:
                    raw = await reader.readuntil(b"\r\n\r\n")
                except asyncio.IncompleteReadError:
                    break
                lines = raw.decode().split("\r\n")
                assert lines[0].startswith("PATCH /orgs/1/apps/2/runs/"), lines[0]
                headers = dict(line.lower().split(": ", 1) for line in lines[1:] if line)
                assert headers["authorization"] == "bearer offline-benchmark"
                body = await reader.readexactly(int(headers["content-length"]))
                writes.append(json.loads(body))
                connection_header = b"Connection: close\r\n" if close_after_response else b""
                writer.write(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n" + connection_header + b"\r\n")
                await writer.drain()
                if close_after_response:
                    break
        except Exception as error:  # noqa: BLE001 - surface server failures to the benchmark caller
            errors.append(repr(error))
        finally:
            writer.close()
            with contextlib.suppress(ConnectionError):
                await writer.wait_closed()
            active -= 1
            if not active:
                idle.set()

    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert_dir / "cert.pem", cert_dir / "key.pem")
    server = await asyncio.start_server(handle, "127.0.0.1", 0, ssl=tls)
    port = server.sockets[0].getsockname()[1]
    config = PlatformConfig(
        host=f"127.0.0.1:{port}",
        auth=PlatformAuth(type=PlatformAuthType.BEARER, token="offline-benchmark"),
        subject=PlatformSubject(org_id="1", app_id="2"),
    )

    async def trial(mode):
        before_connections, before_writes = connections, len(writes)
        agent = Agent(name="router", model=TestModel(responses=["route_a"]), tracing_provider=PlatformTracingProvider)
        ctx = RunContext(platform_config=config, tracing_provider=PlatformTracingProvider)
        output = io.StringIO()
        scope = platform_http_session if mode == "pooled" else contextlib.nullcontext
        start = time.perf_counter()
        with patch("timbal.codegen.test.platform_http_session", scope), contextlib.redirect_stdout(output):
            await run_test(SimpleNamespace(load=lambda: agent), {"prompt": "route this"}, run_context=ctx)
        elapsed = (time.perf_counter() - start) * 1000
        # EOF processing can lag aclose by an event-loop tick. Ensure no pool
        # survives into the next sample; warmup is within a run only.
        await asyncio.wait_for(idle.wait(), 2)
        result = json.loads(output.getvalue().strip().splitlines()[-1])
        assert result["status"]["code"] == "success", result
        assert len(writes) - before_writes == 2
        root = next(span for span in writes[-1]["trace"] if span["parent_call_id"] is None)
        assert root["t1"] is not None and root["status"]["code"] == "success"
        assert not errors, errors
        return {
            "mode": mode,
            "ms": elapsed,
            "connections": connections - before_connections,
            "writes": len(writes) - before_writes,
            "final_state_saved": True,
        }

    old_context = get_run_context()
    try:
        async with server:
            await trial("fresh")
            await trial("pooled")
            rows = []
            for i in range(samples):
                for mode in ("fresh", "pooled") if i % 2 == 0 else ("pooled", "fresh"):
                    rows.append(await trial(mode))
    finally:
        set_run_context(old_context)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "httpx": httpx.__version__,
        "synthetic_connection_delay_ms": connection_delay * 1000,
        "server_keepalive": not close_after_response,
        "samples": rows,
        "summary": {
            mode: {
                "median_ms": statistics.median(row["ms"] for row in rows if row["mode"] == mode),
                "min_ms": min(row["ms"] for row in rows if row["mode"] == mode),
                "max_ms": max(row["ms"] for row in rows if row["mode"] == mode),
                "connections_per_run": sorted({row["connections"] for row in rows if row["mode"] == mode}),
                "writes_per_run": sorted({row["writes"] for row in rows if row["mode"] == mode}),
            }
            for mode in ("fresh", "pooled")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--connection-delay-ms", type=float, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--close-after-response", action="store_true")
    args = parser.parse_args()
    if args.samples < 1 or args.connection_delay_ms < 0:
        parser.error("samples must be positive and delay must be nonnegative")
    with tempfile.TemporaryDirectory(prefix="timbal-http-bench-") as directory:
        cert_dir = Path(directory)
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1,DNS:localhost",
                "-keyout",
                str(cert_dir / "key.pem"),
                "-out",
                str(cert_dir / "cert.pem"),
            ],
            check=True,
            capture_output=True,
        )
        # Keep certificate verification enabled without modifying system trust.
        with patch.dict(os.environ, {"SSL_CERT_FILE": str(cert_dir / "cert.pem"), "NO_PROXY": "127.0.0.1"}):
            result = asyncio.run(
                benchmark(args.samples, args.connection_delay_ms / 1000, cert_dir, args.close_after_response)
            )
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
