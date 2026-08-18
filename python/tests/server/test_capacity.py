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

    def test_a_nonsense_v2_quota_falls_through_to_v1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """A quota of 0 is not "unlimited" — it is an unreadable quota, and the
        v1 files are still worth asking."""
        _write_cgroup_v2(monkeypatch, tmp_path, "0 100000\n")
        quota = tmp_path / "quota"
        period = tmp_path / "period"
        quota.write_text("300000\n")
        period.write_text("100000\n")
        monkeypatch.setattr(capacity, "_CGROUP_V1_QUOTA", quota)
        monkeypatch.setattr(capacity, "_CGROUP_V1_PERIOD", period)
        assert capacity.available_cpus() == pytest.approx(3.0)


class TestAutoLimit:
    def test_scales_with_the_quota(self, monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "auto")
        _write_cgroup_v2(monkeypatch, tmp_path, "400000 100000\n")  # 4 vCPU
        assert capacity.max_concurrent_sessions() == 8

    def test_the_quota_is_split_across_workers(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """The counter is per process but the cgroup quota covers the whole
        container. Without the divisor, `--workers 4` on 4 vCPUs would admit 8
        per worker — 32 sessions on 4 CPUs, the overcommit this exists to stop."""
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "auto")
        monkeypatch.setenv("TIMBAL_SERVER_WORKERS", "4")
        _write_cgroup_v2(monkeypatch, tmp_path, "400000 100000\n")  # 4 vCPU
        assert capacity.max_concurrent_sessions() == 2

    def test_a_garbage_worker_count_is_one_worker(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "auto")
        monkeypatch.setenv("TIMBAL_SERVER_WORKERS", "many")
        _write_cgroup_v2(monkeypatch, tmp_path, "400000 100000\n")
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


class TestDefaultAuto:
    """One counter, two ceilings. Entry points that predate the cap stay
    uncapped when nothing is configured; the per-request LiveKit dial — which
    nothing predates, and which is otherwise unbounded — gets `auto`."""

    def test_a_new_path_is_capped_while_the_old_ones_are_not(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.delenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", raising=False)
        _write_cgroup_v2(monkeypatch, tmp_path, "100000 100000\n")  # 1 vCPU -> auto == 2
        assert capacity.max_concurrent_sessions() == 0
        assert capacity.max_concurrent_sessions(default_auto=True) == 2

        assert capacity.acquire_session_slot(default_auto=True)
        assert capacity.acquire_session_slot(default_auto=True)
        assert not capacity.acquire_session_slot(default_auto=True)
        # Same counter, so the box really is at two — but a transport that was
        # never capped still admits, which is what keeps this backwards
        # compatible.
        assert capacity.acquire_session_slot()
        assert capacity.active_sessions() == 3

    def test_an_explicit_limit_wins_over_the_auto_default(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        _write_cgroup_v2(monkeypatch, tmp_path, "1600000 100000\n")  # 16 vCPU -> auto == 32
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "1")
        assert capacity.max_concurrent_sessions(default_auto=True) == 1
        assert capacity.acquire_session_slot(default_auto=True)
        assert not capacity.acquire_session_slot(default_auto=True)

    def test_an_explicit_zero_still_uncaps_the_new_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """`0` is how someone says "I know, and I want it uncapped" — it has to
        beat the default, or there is no way to turn the ceiling off."""
        _write_cgroup_v2(monkeypatch, tmp_path, "100000 100000\n")
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "0")
        assert capacity.max_concurrent_sessions(default_auto=True) == 0
        for _ in range(50):
            assert capacity.acquire_session_slot(default_auto=True)


class TestBootLog:
    def test_log_capacity_resolves_without_a_session(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        """Boot-time visibility: resolution is lazy, so an operator who set
        `auto` would otherwise learn the ceiling from the first rejection."""
        monkeypatch.setenv("TIMBAL_VOICE_MAX_CONCURRENT_SESSIONS", "auto")
        _write_cgroup_v2(monkeypatch, tmp_path, "200000 100000\n")
        capacity.log_capacity()
        assert capacity.max_concurrent_sessions() == 4
