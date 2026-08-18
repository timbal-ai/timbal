"""Concurrent-voice-session cap: CPU sizing, env override, counter semantics."""

from __future__ import annotations

import pytest
from timbal.server import capacity


@pytest.fixture(autouse=True)
def _clean_capacity():
    capacity.reset_for_tests()
    yield
    capacity.reset_for_tests()


def _write_cgroup_v2(monkeypatch: pytest.MonkeyPatch, tmp_path, content: str) -> None:
    p = tmp_path / "cpu.max"
    p.write_text(content)
    monkeypatch.setattr(capacity, "_CGROUP_V2", p)


def _no_cgroups(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setattr(capacity, "_CGROUP_V2", missing)
    monkeypatch.setattr(capacity, "_CGROUP_V1_QUOTA", missing)
    monkeypatch.setattr(capacity, "_CGROUP_V1_PERIOD", missing)


class TestCpuDetection:
    def test_cgroup_v2_quota_beats_the_host_core_count(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The whole point: a 0.5-vCPU container must not read the host's cores."""
        _write_cgroup_v2(monkeypatch, tmp_path, "50000 100000\n")
        assert capacity.available_cpus() == pytest.approx(0.5)

    def test_cgroup_v2_unlimited_falls_through(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _write_cgroup_v2(monkeypatch, tmp_path, "max 100000\n")
        monkeypatch.setattr(capacity, "_CGROUP_V1_QUOTA", tmp_path / "nope")
        assert capacity.available_cpus() >= 1

    def test_cgroup_v1_quota(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(capacity, "_CGROUP_V2", tmp_path / "nope")
        quota = tmp_path / "quota"
        period = tmp_path / "period"
        quota.write_text("200000\n")
        period.write_text("100000\n")
        monkeypatch.setattr(capacity, "_CGROUP_V1_QUOTA", quota)
        monkeypatch.setattr(capacity, "_CGROUP_V1_PERIOD", period)
        assert capacity.available_cpus() == pytest.approx(2.0)

    def test_v1_unlimited_is_minus_one(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setattr(capacity, "_CGROUP_V2", tmp_path / "nope")
        quota = tmp_path / "quota"
        period = tmp_path / "period"
        quota.write_text("-1\n")
        period.write_text("100000\n")
        monkeypatch.setattr(capacity, "_CGROUP_V1_QUOTA", quota)
        monkeypatch.setattr(capacity, "_CGROUP_V1_PERIOD", period)
        assert capacity.available_cpus() >= 1

    def test_no_cgroup_files_still_answers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _no_cgroups(monkeypatch, tmp_path)
        assert capacity.available_cpus() >= 1


class TestAutoLimit:
    def test_scales_with_the_quota(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "auto")
        _write_cgroup_v2(monkeypatch, tmp_path, "400000 100000\n")  # 4 vCPU
        assert capacity.max_concurrent_sessions() == 8

    def test_a_fractional_task_still_admits_two(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A cap of 0 or 1 would make a reconnect race the previous session's
        teardown; the floor is what keeps a tiny task usable."""
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "AUTO")
        _write_cgroup_v2(monkeypatch, tmp_path, "25000 100000\n")  # 0.25 vCPU
        assert capacity.max_concurrent_sessions() == 2


class TestEnvOverride:
    def test_unset_means_no_cap(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """Opt-in: an upgrade must not start rejecting calls on a deployment
        that is already running more than its CPU allowance."""
        monkeypatch.delenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", raising=False)
        _write_cgroup_v2(monkeypatch, tmp_path, "25000 100000\n")
        assert capacity.max_concurrent_sessions() == 0
        for _ in range(50):
            assert capacity.acquire_session_slot()

    def test_explicit_limit_wins(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        _write_cgroup_v2(monkeypatch, tmp_path, "400000 100000\n")
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "3")
        assert capacity.max_concurrent_sessions() == 3

    def test_zero_disables_the_cap(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "0")
        assert capacity.max_concurrent_sessions() == 0
        for _ in range(50):
            assert capacity.acquire_session_slot()

    def test_garbage_stays_uncapped(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        """A typo should cost a log line, not a wave of rejections."""
        _write_cgroup_v2(monkeypatch, tmp_path, "100000 100000\n")
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "lots")
        assert capacity.max_concurrent_sessions() == 0


class TestCounter:
    def test_admits_up_to_the_limit_then_rejects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "2")
        assert capacity.acquire_session_slot()
        assert capacity.acquire_session_slot()
        assert not capacity.acquire_session_slot()
        assert capacity.active_sessions() == 2

    def test_a_released_slot_is_reusable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        assert capacity.acquire_session_slot()
        assert not capacity.acquire_session_slot()
        capacity.release_session_slot()
        assert capacity.active_sessions() == 0
        assert capacity.acquire_session_slot()

    def test_a_rejected_acquire_does_not_consume_a_slot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        assert capacity.acquire_session_slot()
        assert not capacity.acquire_session_slot()
        assert capacity.active_sessions() == 1

    def test_underflow_is_refused_not_wrapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A double release would make the box look emptier than it is —
        exactly the overcommit this cap exists to prevent."""
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        assert capacity.acquire_session_slot()
        capacity.release_session_slot()
        capacity.release_session_slot()
        assert capacity.active_sessions() == 0
