"""First-token benchmark: prompt, cache accounting, proxy URL, mocked providers."""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[3]
_SPEC = importlib.util.spec_from_file_location("bench_ttft", ROOT / "benchmarks" / "voice" / "bench_ttft.py")
assert _SPEC and _SPEC.loader
bench = importlib.util.module_from_spec(_SPEC)
sys.modules["bench_ttft"] = bench
_SPEC.loader.exec_module(bench)


def test_prompt_is_stable_and_about_7k_tokens() -> None:
    first = bench.voice_system_prompt()
    second = bench.voice_system_prompt()
    assert first == second
    tokens = bench.estimate_tokens(first)
    assert 7600 <= tokens <= 8600
    assert "phone call" in first
    assert "markdown" in first
    assert bench.system_for_run(None) == first
    fresh = bench.system_for_run("abc")
    assert fresh.startswith("Benchmark run abc.\n")
    assert fresh.endswith(first)


def test_shortlist_is_openai_groq_and_anthropic_including_bedrock_eu() -> None:
    labels = [target.label for target in bench.DEFAULT_TARGETS]
    assert labels == [
        "openai/gpt-6-luna",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.4-mini",
        "groq/openai/gpt-oss-20b",
        "groq/openai/gpt-oss-120b",
        "anthropic/claude-haiku-4-5",
        "anthropic/claude-haiku-5-5",
        "bedrock/eu.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/eu.anthropic.claude-haiku-5-5",
    ]
    assert {target.provider for target in bench.DEFAULT_TARGETS} == {"openai", "groq", "anthropic", "bedrock"}
    assert all("fireworks" not in target.label for target in bench.DEFAULT_TARGETS)


def test_missing_keys_are_named_and_a_proxy_base_url_switches_the_route() -> None:
    plans = bench.plan_targets(env={})
    assert all(plan.skip_reason for plan in plans)
    assert any("OPENAI_API_KEY" in (plan.skip_reason or "") for plan in plans)
    assert any("GROQ_API_KEY" in (plan.skip_reason or "") for plan in plans)
    assert any("ANTHROPIC_API_KEY" in (plan.skip_reason or "") for plan in plans)
    assert any("AWS_BEARER_TOKEN_BEDROCK" in (plan.skip_reason or "") for plan in plans)

    proxy = bench.plan_targets(
        env={"TIMBAL_API_KEY": "platform"},
        base_urls=bench.parse_base_urls(
            [
                "openai=https://proxy.example/orgs/1/proxies/openai-completions/v1",
                "groq=https://proxy.example/orgs/1/proxies/openai-completions/v1",
                "anthropic=https://proxy.example/orgs/1/proxies/anthropic",
            ]
        ),
        models=["gpt-5.4-mini", "gpt-oss-20b", "haiku-5-5"],
    )
    by_label = {plan.label: plan for plan in proxy}
    openai = by_label["openai/gpt-5.4-mini"]
    assert openai.route == "proxy"
    assert openai.api == "chat"
    assert openai.api_key == "platform"
    assert bench.request_url(openai) == "https://proxy.example/orgs/1/proxies/openai-completions/v1/chat/completions"
    groq = by_label["groq/openai/gpt-oss-20b"]
    assert groq.api == "chat"
    assert bench.request_url(groq).endswith("/chat/completions")
    haiku = by_label["anthropic/claude-haiku-5-5"]
    assert bench.request_url(haiku) == "https://proxy.example/orgs/1/proxies/anthropic/v1/messages"

    direct = bench.plan_targets(env={"OPENAI_API_KEY": "sk-test"}, eu=True, models=["gpt-6-luna"])
    assert direct[0].route == "direct-eu"
    assert direct[0].api == "responses"
    assert bench.request_url(direct[0]) == "https://eu.api.openai.com/v1/responses"
    body = bench.build_body(direct[0], system="static")
    assert "temperature" not in body
    assert body["prompt_cache_key"] == bench.PROMPT_CACHE_KEY

    haiku_plan = bench.plan_targets(env={"ANTHROPIC_API_KEY": "sk-ant"}, models=["haiku-5-5"])[0]
    haiku_body = bench.build_body(haiku_plan, system="static")
    assert "temperature" not in haiku_body
    assert haiku_body["system"][0]["cache_control"] == {"type": "ephemeral"}


def test_sigv4_matches_the_published_iam_example() -> None:
    headers = bench.authorization_headers(
        method="GET",
        url="https://iam.amazonaws.com/?Action=ListUsers&Version=2010-05-08",
        payload=b"",
        access_key="AKIDEXAMPLE",
        secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        service="iam",
        amz_date="20150830T123600Z",
        signed={"content-type": "application/x-www-form-urlencoded; charset=utf-8"},
    )
    assert headers["authorization"].endswith(
        "Signature=5d672d79c15b13162d9279b0855cfba6789a8edb4c82c400e06b5924a6f2b5d7"
    )


def test_usage_parsers_cover_openai_groq_and_anthropic() -> None:
    openai = {
        "usage": {
            "prompt_tokens": 7000,
            "prompt_tokens_details": {"cached_tokens": 6800},
            "completion_tokens": 4,
        }
    }
    acc = bench.UsageAcc()
    acc.observe(openai)
    assert acc.cached_tokens == 6800
    groq_miss = {"usage": {"prompt_tokens": 7000, "prompt_tokens_details": {"cached_tokens": 0}}}
    miss = bench.UsageAcc()
    miss.observe(groq_miss)
    assert miss.cached_tokens == 0
    # Groq drops the details object on a miss and still reports prompt_time.
    omitted = bench.UsageAcc()
    omitted.observe({"usage": {"prompt_tokens": 7260, "prompt_time": 0.2, "queue_time": 0.1, "completion_tokens": 4}})
    assert omitted.cached_tokens == 0
    anthropic = {
        "type": "message_delta",
        "usage": {"output_tokens": 4, "cache_read_input_tokens": 6900, "cache_creation_input_tokens": 0},
    }
    read = bench.UsageAcc()
    read.observe(anthropic)
    assert read.cached_tokens == 6900
    assert bench.text_of({"type": "response.output_text.delta", "delta": "Hi"}) == "Hi"
    assert bench.text_of({"choices": [{"delta": {"content": None, "reasoning": "think"}}]}) is None
    assert bench.text_of({"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hi"}}) == "Hi"


def _sse(events: list[dict]) -> bytes:
    lines = []
    for event in events:
        lines.append("data: " + json.dumps(event))
    lines.append("data: [DONE]")
    return ("\n".join(lines) + "\n").encode()


def _bedrock_body(event: dict) -> bytes:
    payload = json.dumps({"bytes": base64.b64encode(json.dumps(event).encode()).decode()}).encode()
    return bench.encode_event_frame(payload)


def test_mocked_direct_and_bedrock_report_cold_and_cached() -> None:
    seen = {"chat": 0, "bedrock": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"data": []})
        path = request.url.path
        key = "bedrock" if path.endswith("/invoke-with-response-stream") else "chat"
        seen[key] += 1
        cached = 0 if seen[key] == 1 else 6800
        if path.endswith("/invoke-with-response-stream"):
            assert "eu.anthropic.claude-haiku-5-5" in path
            assert request.headers["authorization"].startswith("AWS4-HMAC-SHA256")
            body = json.loads(request.content)
            assert "temperature" not in body
            assert body["system"][0]["cache_control"]["type"] == "ephemeral"
            frame = _bedrock_body(
                {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Sure"}}
            ) + _bedrock_body(
                {
                    "type": "message_delta",
                    "usage": {
                        "output_tokens": 1,
                        "input_tokens": 200,
                        "cache_read_input_tokens": cached,
                        "cache_creation_input_tokens": 0 if cached else 6800,
                    },
                }
            )
            return httpx.Response(200, content=frame)
        event = {
            "choices": [{"delta": {"content": "Sure"}, "index": 0}],
            "usage": {
                "prompt_tokens": 7100,
                "prompt_tokens_details": {"cached_tokens": cached},
                "completion_tokens": 1,
            },
        }
        # Usage often arrives on a later chunk. Send the token first.
        token = {"choices": [{"delta": {"content": "Sure"}, "index": 0}]}
        usage = {
            "choices": [],
            "usage": {
                "prompt_tokens": 7100,
                "prompt_tokens_details": {"cached_tokens": cached},
                "completion_tokens": 1,
            },
        }
        del event
        return httpx.Response(200, content=_sse([token, usage]))

    transport = httpx.MockTransport(handler)

    async def factory(plan):
        del plan
        return httpx.AsyncClient(transport=transport)

    plans = bench.plan_targets(
        env={"GROQ_API_KEY": "g-test", "AWS_ACCESS_KEY_ID": "AKID", "AWS_SECRET_ACCESS_KEY": "secret"},
        models=["gpt-oss-20b", "eu.anthropic.claude-haiku-5-5"],
    )
    assert all(plan.skip_reason is None for plan in plans)

    import asyncio

    results = asyncio.run(
        bench.run_targets(
            plans,
            samples=2,
            warmup=True,
            env={
                "GROQ_API_KEY": "g-test",
                "AWS_ACCESS_KEY_ID": "AKID",
                "AWS_SECRET_ACCESS_KEY": "secret",
            },
            client_factory=factory,
        )
    )
    by_label = {result.label: result for result in results}
    groq = by_label["groq/openai/gpt-oss-20b"]
    assert groq.error is None
    assert groq.summary()["cache_hit_rate"] == 0.5
    assert groq.summary()["cold_samples"] == 1
    assert groq.summary()["cached_samples"] == 1
    assert groq.summary()["prompt_tokens"] == 7100
    assert groq.summary()["ttft_p50_ms"] is not None
    bedrock = by_label["bedrock/eu.anthropic.claude-haiku-5-5"]
    assert bedrock.error is None, bedrock.error
    assert bedrock.route == "bedrock-eu"
    assert bedrock.summary()["cached_samples"] == 1
    assert bedrock.summary()["cold_samples"] == 1


def test_exit_code_when_nothing_ran() -> None:
    skipped = bench.TargetResult(
        label="x", provider="openai", model="m", api="chat", route="direct", base_url="", skipped="set OPENAI_API_KEY"
    )
    assert bench.exit_code([skipped]) == 2
    failed = bench.TargetResult(
        label="x", provider="openai", model="m", api="chat", route="direct", base_url="", error="HTTP 401"
    )
    assert bench.exit_code([failed]) == 1
    ok = bench.TargetResult(
        label="x",
        provider="openai",
        model="m",
        api="chat",
        route="direct",
        base_url="",
        samples=[bench.Sample(ttft_ms=10)],
    )
    assert bench.exit_code([ok, skipped]) == 0
