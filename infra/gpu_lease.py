"""NFS-based per-GPU leases so multiple SSH-dispatch drivers can share a cluster.

Each sub-worker `(ip, gpu_index)` maps to one lock file in `LEASE_DIR`. The
kernel `fcntl.flock` state is the source of truth (auto-released on process
death, giving free crash safety); a small JSON payload is an advisory
heartbeat for wedged-but-alive peers.
"""
from __future__ import annotations

import fcntl
import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

LEASE_DIR = Path("/nfs/cluster_leases")
STALE_TTL_S = 180
HEARTBEAT_S = 30

LeaseKey = tuple[str, int | None]


@dataclass
class LeaseHandle:
    path: Path
    fd: int
    key: LeaseKey


def _lease_name(key: LeaseKey) -> str:
    ip, gpu = key
    suffix = f"gpu{gpu}" if gpu is not None else "gpuNone"
    return f"{ip}_{suffix}.lock"


def _payload(driver: str, job: str) -> bytes:
    now = time.time()
    return json.dumps({
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "driver": driver,
        "job": job,
        "acquired_ts": now,
        "heartbeat_ts": now,
    }).encode()


class LeaseManager:
    """Per-driver lease pool. One heartbeat thread refreshes all held leases."""

    def __init__(self, driver: str = "", lease_dir: Path = LEASE_DIR) -> None:
        self.lease_dir = lease_dir
        self.lease_dir.mkdir(parents=True, exist_ok=True)
        self.driver = driver
        self._held: dict[LeaseKey, LeaseHandle] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop, name="lease-heartbeat", daemon=True)
        self._heartbeat.start()

    def try_acquire(self, key: LeaseKey, job: str = "") -> LeaseHandle | None:
        path = self.lease_dir / _lease_name(key)
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return None
        # We hold the kernel lock. Stamp the advisory payload.
        os.ftruncate(fd, 0)
        os.write(fd, _payload(self.driver, job))
        os.fsync(fd)
        handle = LeaseHandle(path=path, fd=fd, key=key)
        with self._lock:
            self._held[key] = handle
        return handle

    def release(self, handle: LeaseHandle) -> None:
        with self._lock:
            self._held.pop(handle.key, None)
        try:
            os.ftruncate(handle.fd, 0)
            fcntl.flock(handle.fd, fcntl.LOCK_UN)
        finally:
            os.close(handle.fd)

    def shutdown(self) -> None:
        self._stop.set()
        self._heartbeat.join(timeout=HEARTBEAT_S + 5)
        with self._lock:
            handles = list(self._held.values())
            self._held.clear()
        for h in handles:
            try:
                os.ftruncate(h.fd, 0)
                fcntl.flock(h.fd, fcntl.LOCK_UN)
                os.close(h.fd)
            except OSError:
                pass

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(HEARTBEAT_S):
            with self._lock:
                snapshot = list(self._held.items())
            for key, h in snapshot:
                try:
                    os.lseek(h.fd, 0, os.SEEK_SET)
                    raw = os.read(h.fd, 4096)
                    payload = json.loads(raw) if raw else {}
                    payload["heartbeat_ts"] = time.time()
                    data = json.dumps(payload).encode()
                    os.ftruncate(h.fd, 0)
                    os.lseek(h.fd, 0, os.SEEK_SET)
                    os.write(h.fd, data)
                except (OSError, json.JSONDecodeError):
                    pass
