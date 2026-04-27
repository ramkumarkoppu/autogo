"""Tests for infra.gpu_lease and priority file in infra.remote_exec."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import pytest

from infra.gpu_lease import HEARTBEAT_S, STALE_TTL_S, LeaseManager, _lease_name
from infra.remote_exec import Job, _read_priorities, _write_priorities_stub


KEY = ("10.0.0.1", 0)


def _child_hold(lease_dir: str, ready_path: str, release_path: str) -> int:
    mgr = LeaseManager(driver="child", lease_dir=Path(lease_dir))
    h = mgr.try_acquire(KEY, job="child-job")
    Path(ready_path).write_text("1" if h is not None else "0")
    if h is None:
        return 1
    # Wait for parent to signal release.
    while not Path(release_path).exists():
        time.sleep(0.05)
    mgr.release(h)
    mgr.shutdown()
    return 0


def test_contention_exactly_one_winner(tmp_path: Path) -> None:
    lease_dir = tmp_path / "leases"
    ready = tmp_path / "child_ready"
    release = tmp_path / "child_release"
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child_hold, args=(str(lease_dir), str(ready), str(release)))
    p.start()
    try:
        deadline = time.time() + 5
        while not ready.exists() and time.time() < deadline:
            time.sleep(0.05)
        assert ready.read_text() == "1", "child should have acquired"

        parent = LeaseManager(driver="parent", lease_dir=lease_dir)
        h = parent.try_acquire(KEY, job="parent-job")
        assert h is None, "parent must not acquire while child holds"

        release.write_text("go")
        deadline = time.time() + 5
        while p.is_alive() and time.time() < deadline:
            time.sleep(0.05)
        p.join(timeout=2)

        h2 = parent.try_acquire(KEY, job="parent-job")
        assert h2 is not None, "parent should acquire after child releases"
        parent.release(h2)
        parent.shutdown()
    finally:
        if p.is_alive():
            p.kill()
            p.join()


def test_crash_safety_via_kill(tmp_path: Path) -> None:
    lease_dir = tmp_path / "leases"
    ready = tmp_path / "child_ready"
    release = tmp_path / "never_signaled"
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=_child_hold, args=(str(lease_dir), str(ready), str(release)))
    p.start()
    deadline = time.time() + 5
    while not ready.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert ready.read_text() == "1"

    p.kill()
    p.join()

    parent = LeaseManager(driver="parent", lease_dir=lease_dir)
    deadline = time.time() + 3
    h = None
    while time.time() < deadline:
        h = parent.try_acquire(KEY, job="after-crash")
        if h is not None:
            break
        time.sleep(0.05)
    assert h is not None, "kernel should have released the lock on process death"
    parent.release(h)
    parent.shutdown()


def test_payload_written(tmp_path: Path) -> None:
    mgr = LeaseManager(driver="unit", lease_dir=tmp_path)
    h = mgr.try_acquire(KEY, job="jA")
    assert h is not None
    data = json.loads((tmp_path / _lease_name(KEY)).read_text())
    assert data["driver"] == "unit"
    assert data["job"] == "jA"
    assert data["pid"] == os.getpid()
    assert data["heartbeat_ts"] >= data["acquired_ts"]
    mgr.release(h)
    mgr.shutdown()


def test_release_makes_reacquirable(tmp_path: Path) -> None:
    mgr = LeaseManager(driver="unit", lease_dir=tmp_path)
    h1 = mgr.try_acquire(KEY, job="first")
    assert h1 is not None
    mgr.release(h1)
    h2 = mgr.try_acquire(KEY, job="second")
    assert h2 is not None
    mgr.release(h2)
    mgr.shutdown()


def test_module_constants_sane() -> None:
    assert HEARTBEAT_S < STALE_TTL_S
    assert STALE_TTL_S >= 3 * HEARTBEAT_S  # survive two missed heartbeats


# ---- priority file ------------------------------------------------------------


def test_priorities_stub_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "job_priorities.txt"
    jobs = [Job(name="a", inner_cmd="x"), Job(name="b", inner_cmd="y")]
    _write_priorities_stub(path, jobs)
    assert _read_priorities(path) == {"a": 0, "b": 0}


def test_priorities_user_edit(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("# hi\n7 high-prio\n0 low-prio\n3\tmid\n")
    prio = _read_priorities(path)
    assert prio == {"high-prio": 7, "low-prio": 0, "mid": 3}


def test_priorities_missing_file(tmp_path: Path) -> None:
    assert _read_priorities(tmp_path / "does-not-exist.txt") == {}
    assert _read_priorities(None) == {}


def test_priorities_malformed_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "p.txt"
    path.write_text("not_a_number jobA\n\n2 jobB\njust_one_token\n3 jobC\n")
    assert _read_priorities(path) == {"jobB": 2, "jobC": 3}


@pytest.mark.parametrize("prio_map,expected_first", [
    ({"a": 0, "b": 0, "c": 0}, "a"),   # tie -> FIFO
    ({"a": 0, "b": 5, "c": 0}, "b"),   # bumped b wins
    ({"a": 0, "b": 5, "c": 9}, "c"),   # c wins
    ({"a": -1, "b": -1, "c": 0}, "c"),  # c wins over negatives
])
def test_priority_ordering(tmp_path: Path, prio_map: dict, expected_first: str) -> None:
    """Mirror _claim_next's sort to confirm behavior without spinning up SSH."""
    jobs = [Job(name=n, inner_cmd="") for n in ["a", "b", "c"]]
    order = sorted(range(len(jobs)), key=lambda i: (-prio_map.get(jobs[i].name, 0), i))
    assert jobs[order[0]].name == expected_first
