#!/usr/bin/env -S uv run python
"""Cluster management: register worker hosts and verify SSH reachability.

Subcommands:
  add    — prepare a remote host for SSH-dispatched `docker run --rm` jobs
           (installs Docker + NVIDIA toolkit, logs into GHCR, pulls the image,
           seeds /nfs, appends [nodes."<ip>"] to cluster.toml).
  ping   — SSH-echo every node in cluster.toml and report ✓/✗.
  build  — build Dockerfile.worker locally and push it to GHCR
           (ghcr.io/ericjang/alphago-worker). The image this publishes is
           what `add` pulls and what `infra.remote_exec` runs workers from.
  pull   — `docker pull <image>` on every node in cluster.toml in parallel.
           Run after `build` to roll the new :latest onto the fleet — the
           remote_exec `docker run` path doesn't use `--pull=always`, so
           cached stale images would otherwise stick forever.
  status — show which jobs are running on each host, by reading the
           `infra.gpu_lease` flock payloads from /nfs/cluster_leases/.
           Doesn't SSH — everything lives on shared NFS, so one `ls` +
           a few file reads is enough.

cluster.toml is the source of truth consumed by
``infra.remote_exec.load_cluster(role)``.
"""

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
try:
    import tomllib  # type: ignore[import-not-found]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[no-redef]

import tyro

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
CLUSTER_TOML = PROJECT_ROOT / "cluster.toml"
GHCR_IMAGE = "ghcr.io/ericjang/alphago-worker"
SSH_KEY = "~/.ssh/id_ed25519"


@dataclass
class Node:
    type: str
    host: str
    ssh_port: int | None = None

    @property
    def ip(self) -> str:
        return self.host.split("@")[1] if "@" in self.host else self.host

    def __str__(self) -> str:
        port_str = f":{self.ssh_port}" if self.ssh_port else ""
        return f"{self.host}{port_str} ({self.type})"


def parse_node(entry: str | list[str] | dict) -> Node:
    if isinstance(entry, str):
        return Node(type="baremetal", host=entry)
    if isinstance(entry, list):
        return Node(type=entry[0], host=entry[1])
    return Node(
        type=entry.get("type", "baremetal"),
        host=entry["host"],
        ssh_port=entry.get("ssh_port"),
    )


def get_image() -> str:
    """Return the worker image tag from cluster.toml's top-level `image` key."""
    if not CLUSTER_TOML.exists():
        return f"{GHCR_IMAGE}:latest"
    with open(CLUSTER_TOML, "rb") as f:
        cfg = tomllib.load(f)
    return cfg.get("image", f"{GHCR_IMAGE}:latest")


# --- SSH helpers ---

def _ssh_base_args(port: int | None = None) -> list[str]:
    args = ["-o", "StrictHostKeyChecking=no", "-i", str(Path(SSH_KEY).expanduser())]
    if port:
        args.extend(["-p", str(port)])
    return args


def _scp_base_args(port: int | None = None) -> list[str]:
    args = ["-o", "StrictHostKeyChecking=no", "-i", str(Path(SSH_KEY).expanduser())]
    if port:
        args.extend(["-P", str(port)])
    return args


def ssh(node: Node, cmd: str, check: bool = True, env: dict[str, str] | None = None) -> int:
    env_prefix = " ".join(f"export {k}={v} &&" for k, v in env.items()) + " " if env else ""
    full = ["ssh", *_ssh_base_args(node.ssh_port), node.host, env_prefix + cmd]
    print(f"[{node.host}] {cmd}")
    r = subprocess.run(full, check=False)
    if check and r.returncode != 0:
        print(f"[{node.host}] Command failed with code {r.returncode}")
    return r.returncode


def scp(node: Node, local: Path, remote: str) -> int:
    full = ["scp", *_scp_base_args(node.ssh_port), str(local), f"{node.host}:{remote}"]
    print(f"[{node.host}] Copying {local} -> {remote}")
    return subprocess.run(full, check=False).returncode


def ssh_output(node: Node, cmd: str) -> str | None:
    full = ["ssh", *_ssh_base_args(node.ssh_port), node.host, cmd]
    r = subprocess.run(full, capture_output=True, text=True, check=False)
    return r.stdout.strip() if r.returncode == 0 else None


def host_has_gpu(node: Node) -> bool:
    return ssh(node, "nvidia-smi > /dev/null 2>&1", check=False) == 0


def detect_gpu_count(node: Node) -> int:
    output = ssh_output(node, "nvidia-smi --query-gpu=name --format=csv,noheader")
    if not output:
        return 0
    return sum(1 for line in output.splitlines() if line.strip())


def get_ghcr_credentials() -> tuple[str | None, str | None]:
    token = os.environ.get("GHCR_TOKEN")
    user = os.environ.get("GHCR_USER")
    secrets_file = PROJECT_ROOT / ".secrets"
    if (not token or not user) and secrets_file.exists():
        for line in secrets_file.read_text().splitlines():
            if line.startswith("GHCR_TOKEN="):
                token = line.split("=", 1)[1]
            elif line.startswith("GHCR_USER="):
                user = line.split("=", 1)[1]
    return user, token


def _docker(node: Node) -> str:
    """'sudo docker' for non-root hosts (group membership isn't active in the
    same SSH session that ran setup), plain 'docker' for root."""
    user = node.host.split("@")[0] if "@" in node.host else "root"
    return "docker" if user == "root" else "sudo docker"


def docker_login(node: Node) -> bool:
    user, token = get_ghcr_credentials()
    if not token or not user:
        print(f"[{node.host}] WARNING: GHCR_TOKEN/GHCR_USER not found, skipping docker login")
        return True
    r = subprocess.run(
        ["ssh", *_ssh_base_args(node.ssh_port), node.host,
         f"{_docker(node)} login ghcr.io -u " + user + " --password-stdin"],
        input=token.encode(), check=False,
    )
    return r.returncode == 0


def _append_cluster_toml_entry(ip: str, user: str, shares_nfs: bool = True,
                               ssh_port: int | None = None, num_gpu: int = 0) -> bool:
    """Append [nodes."<ip>"] block to cluster.toml if not already present."""
    CLUSTER_TOML.touch(exist_ok=True)
    existing = CLUSTER_TOML.read_text()
    section = f'[nodes."{ip}"]'
    if section in existing:
        return False
    prefix = "" if existing.endswith("\n") or not existing else "\n"
    extra = ""
    if not shares_nfs:
        extra += "shares_nfs = false\n"
    if ssh_port:
        extra += f"ssh_port = {ssh_port}\n"
    if num_gpu > 0:
        extra += f"num_gpu = {num_gpu}\n"
    entry = (
        f'{prefix}\n{section}\n'
        f'roles = ["collect"]\n'
        f'user = "{user}"\n'
        f'{extra}'
    )
    with open(CLUSTER_TOML, "a") as f:
        f.write(entry)
    return True


# --- Commands ---

@dataclass
class Add:
    """Prepare a remote host for SSH-dispatched `docker run --rm` jobs.

    Installs Docker + NVIDIA toolkit (if GPU), logs into GHCR, pulls the
    image, seeds /nfs, and appends `[nodes."<ip>"]` to cluster.toml. No
    long-running container runs — jobs spin up fresh containers per SSH call.

    Usage: ./infra/cluster.py add [--ssh-port PORT] <user@host>
    """
    host: str
    ssh_port: int | None = None


@dataclass
class Ping:
    """Verify SSH reachability of every node in cluster.toml."""
    timeout: int = 5


@dataclass
class Build:
    """Build Dockerfile.worker locally and push it to GHCR.

    The image pushed here (ghcr.io/ericjang/alphago-worker:latest by
    default, overridable via cluster.toml's top-level `image` key) is the
    one `cluster add` pulls onto each node and that `infra.remote_exec`
    runs every job in. Rebuild + push whenever the worker image's pinned
    deps or baked-in assets change.
    """
    pass


@dataclass
class Pull:
    """`docker pull <image>` on every node in cluster.toml in parallel.

    remote_exec's `docker run` does not pass `--pull=always`, so a node
    that already has the tag cached will keep using the stale copy. Run
    `cluster pull` after `cluster build` to refresh the fleet.
    """
    timeout: int = 300  # docker pull of the full worker image can be slow


@dataclass
class Status:
    """Show which jobs are running on each host by reading NFS lease files.

    Reads /nfs/cluster_leases/*.lock (written by infra.gpu_lease.LeaseManager)
    and reports per-slot state: idle (empty payload / missing file), busy
    (fresh heartbeat — a driver is actively using the GPU), or stale (old
    heartbeat — driver crashed mid-job). No SSH — the lease dir lives on
    shared NFS so every file is a local read.
    """
    lease_dir: Path = Path("/nfs/cluster_leases")
    stale_ttl_s: int = 180  # matches gpu_lease.STALE_TTL_S


def cmd_add(args: Add) -> None:
    node = Node(type="container", host=args.host, ssh_port=args.ssh_port)
    user = args.host.split("@")[0] if "@" in args.host else "root"

    print(f"=== Preparing {node.host} for SSH docker-run jobs ===")

    print("\n[1/4] Installing host dependencies (Docker, NVIDIA toolkit)...")
    script = SCRIPT_DIR / "install_container_worker.sh"
    if scp(node, script, f"~/{script.name}") != 0:
        print(f"Failed to copy install script to {node.host}")
        sys.exit(1)
    if ssh(node, f"chmod +x ~/{script.name} && ~/{script.name}") != 0:
        print(f"Host setup failed on {node.host}")
        sys.exit(1)
    secrets = PROJECT_ROOT / ".secrets"
    if secrets.exists():
        scp(node, secrets, "~/.secrets")

    print("\n[2/4] Logging in to GHCR and pulling image...")
    if not docker_login(node):
        print("docker login failed")
        sys.exit(1)
    image = get_image()
    dk = _docker(node)
    if ssh(node, f"{dk} pull {image}") != 0:
        print(f"Failed to pull {image}")
        sys.exit(1)

    print("\n[3/4] Verifying `docker run --rm` works...")
    if ssh(node, f"{dk} run --rm {image} echo ok") != 0:
        print(f"docker run test failed on {node.host}")
        sys.exit(1)

    # Probe for a shared /nfs; if absent, seed a local tree so containers
    # running with `-v /data/eric:/nfs` still see expected paths.
    nfs_probe = ssh(
        node,
        "[ ! -L /data/eric ] && test -d /data/eric/LearnAlphaGo"
        " && test -d /data/eric/game_data_root",
        check=False,
    )
    shares_nfs = (nfs_probe == 0)
    print(f"  shares_nfs={shares_nfs}")
    sudo = "" if user == "root" else "sudo "
    if shares_nfs:
        print("  ensuring /nfs resolves...")
        seed = (
            f"if [ -d /nfs ] && [ ! -L /nfs ]; then "
            f"  echo '/nfs is a real directory, leaving as-is'; "
            f"else "
            f"  {sudo}ln -sfn /data/eric /nfs; "
            f"fi"
        )
        if ssh(node, seed) != 0:
            print(f"Failed to set up /nfs on {node.host}")
            sys.exit(1)
    else:
        print("  seeding /nfs locally and symlinking /data/eric -> /nfs...")
        seed = (
            f"set -e && "
            f"{sudo}mkdir -p /nfs/LearnAlphaGo /nfs/checkpoints /nfs/game_data_root /data && "
            f"if [ -e /data/eric ] && [ ! -L /data/eric ]; then "
            f"  echo 'ERROR: /data/eric exists and is not a symlink' >&2; exit 1; "
            f"fi && {sudo}ln -sfn /nfs /data/eric"
        )
        if ssh(node, seed) != 0:
            print(f"Failed to seed /nfs on {node.host}")
            sys.exit(1)

    print(f"\n[4/4] Updating {CLUSTER_TOML.name}...")
    num_gpu = detect_gpu_count(node)
    print(f"  num_gpu={num_gpu}")
    if _append_cluster_toml_entry(node.ip, user, shares_nfs=shares_nfs,
                                  ssh_port=args.ssh_port, num_gpu=num_gpu):
        print(f'Added [nodes."{node.ip}"] to {CLUSTER_TOML}')
    else:
        print(f'[nodes."{node.ip}"] already present in {CLUSTER_TOML}, leaving unchanged')

    print(f"\n=== Node {node.host} ready ===")


def cmd_build(args: Build) -> None:
    """Build the worker image and push to GHCR.

    Two-stage build: Dockerfile.worker FROMs the local base tag
    `learnalphago-dev` (built from .devcontainer/Dockerfile — same image the
    VSCode devcontainer uses). We build the base first (cached layers make
    a no-op rebuild cheap), then layer the worker on top.
    """
    image = get_image()
    base_dockerfile = PROJECT_ROOT / ".devcontainer" / "Dockerfile"
    worker_dockerfile = PROJECT_ROOT / "Dockerfile.worker"
    base_tag = "learnalphago-dev"
    if not base_dockerfile.exists():
        print(f"ERROR: {base_dockerfile} not found")
        sys.exit(1)
    if not worker_dockerfile.exists():
        print(f"ERROR: {worker_dockerfile} not found")
        sys.exit(1)

    user, token = get_ghcr_credentials()
    if user and token:
        print("=== Logging in to ghcr.io ===")
        r = subprocess.run(
            ["docker", "login", "ghcr.io", "-u", user, "--password-stdin"],
            input=token.encode(), check=False,
        )
        if r.returncode != 0:
            print("Docker login failed")
            sys.exit(1)
    else:
        print("WARNING: GHCR_USER/GHCR_TOKEN not found; assuming `docker login` already ran")

    print(f"\n=== Building base image {base_tag} (from {base_dockerfile.relative_to(PROJECT_ROOT)}) ===")
    r = subprocess.run(
        ["docker", "build", "-f", str(base_dockerfile), "-t", base_tag, "."],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        print("Base image build failed")
        sys.exit(1)

    print(f"\n=== Building {image} ===")
    r = subprocess.run(
        ["docker", "build", "-f", str(worker_dockerfile), "-t", image, "."],
        cwd=PROJECT_ROOT,
    )
    if r.returncode != 0:
        print("Build failed")
        sys.exit(1)

    print(f"\n=== Pushing {image} ===")
    r = subprocess.run(["docker", "push", image])
    if r.returncode != 0:
        print("Push failed")
        sys.exit(1)

    print(f"\n=== Done: {image} ===")


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


_LEASE_NAME_RE = re.compile(r"^(.+)_gpu(\d+|None)\.lock$")


def cmd_status(args: Status) -> None:
    """Dump per-slot lease status by reading /nfs/cluster_leases/*.lock.
    Only shows hosts registered in cluster.toml; lease files for other IPs
    are ignored."""
    if not args.lease_dir.exists():
        print(f"No lease dir at {args.lease_dir} — has any driver run yet?")
        sys.exit(1)
    toml_entries: dict[str, dict] = {}
    if CLUSTER_TOML.exists():
        with open(CLUSTER_TOML, "rb") as f:
            toml_entries = tomllib.load(f).get("nodes", {})

    # ip -> list of (gpu_suffix, payload_dict_or_None, age_s_or_None)
    per_ip: dict[str, list[tuple[str, dict | None, float | None]]] = {}
    now = time.time()
    for lock in sorted(args.lease_dir.glob("*.lock")):
        m = _LEASE_NAME_RE.match(lock.name)
        if not m:
            continue
        ip, gpu_suffix = m.group(1), m.group(2)
        if ip not in toml_entries:
            continue
        try:
            raw = lock.read_text().strip()
        except OSError:
            continue
        payload: dict | None = None
        age: float | None = None
        if raw:
            try:
                payload = json.loads(raw)
                age = now - float(payload.get("heartbeat_ts", now))
            except (json.JSONDecodeError, ValueError, TypeError):
                payload = {"_parse_error": raw[:60]}
                age = None
        per_ip.setdefault(ip, []).append((gpu_suffix, payload, age))

    # Hosts in cluster.toml with no lease file at all: still show so users
    # notice "this host has never had a job dispatched / was just added."
    for ip in toml_entries:
        per_ip.setdefault(ip, [])

    if not per_ip:
        print("No cluster.toml nodes; nothing to report.")
        sys.exit(1)

    # Header.
    busy_total = stale_total = idle_total = 0
    for ip in sorted(per_ip):
        entry = toml_entries[ip]
        user = entry.get("user", "?")
        roles = ",".join(entry.get("roles", [])) or "-"
        num_gpu = entry.get("num_gpu")
        header = f"{user}@{ip}"
        if num_gpu:
            header += f"  num_gpu={num_gpu}"
        header += f"  roles={roles}"
        print(header)
        slots = sorted(per_ip[ip], key=lambda t: t[0])
        if not slots:
            print("  (no lease files yet — never dispatched)")
            continue
        for suffix, payload, age in slots:
            slot_lbl = f"gpu{suffix}" if suffix != "None" else "gpu-"
            if not payload:
                print(f"  {slot_lbl}  \033[90m○ idle\033[0m")
                idle_total += 1
                continue
            if "_parse_error" in payload:
                print(f"  {slot_lbl}  \033[31m? parse-error\033[0m  "
                      f"raw={payload['_parse_error']!r}")
                continue
            driver = payload.get("driver", "?") or "?"
            job = payload.get("job", "?") or "?"
            acquired = payload.get("acquired_ts")
            held = now - float(acquired) if acquired else None
            stale = age is not None and age > args.stale_ttl_s
            if stale:
                tag = "\033[33m⚠ stale\033[0m"
                stale_total += 1
            else:
                tag = "\033[32m● busy\033[0m"
                busy_total += 1
            bits = [tag, f"driver={driver}", f"job={job}"]
            if held is not None:
                bits.append(f"held={_fmt_age(held)}")
            if age is not None:
                bits.append(f"hb={_fmt_age(age)} ago")
            print(f"  {slot_lbl}  " + "  ".join(bits))
        print()
    total = busy_total + stale_total + idle_total
    print(f"{busy_total} busy / {stale_total} stale / {idle_total} idle "
          f"({total} slot{'s' if total != 1 else ''})")


def cmd_pull(args: Pull) -> None:
    """`docker pull <image>` on every node in cluster.toml in parallel.
    Mirrors cmd_ping's TOML-iteration + ✓/✗ reporting."""
    if not CLUSTER_TOML.exists():
        print(f"No {CLUSTER_TOML}")
        sys.exit(1)
    with open(CLUSTER_TOML, "rb") as f:
        cfg = tomllib.load(f)
    entries = list(cfg.get("nodes", {}).items())
    if not entries:
        print(f"No nodes in {CLUSTER_TOML}")
        sys.exit(1)

    image = get_image()
    print(f"=== Pulling {image} on {len(entries)} node(s) ===")

    def pull_one(item: tuple[str, dict]) -> tuple[str, dict, bool, str]:
        ip, entry = item
        user = entry.get("user", "root")
        port = entry.get("ssh_port")
        # Match _docker(node): plain `docker` for root, `sudo docker` otherwise.
        docker = "docker" if user == "root" else "sudo docker"
        cmd = f"{docker} pull {image}"
        argv = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={min(30, args.timeout)}",
            "-o", "BatchMode=yes",
            "-i", str(Path(SSH_KEY).expanduser()),
        ]
        if port:
            argv += ["-p", str(port)]
        argv += [f"{user}@{ip}", cmd]
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=args.timeout)
            ok = r.returncode == 0
            if ok:
                # Summarize the pull result (Status: ... line from docker pull).
                tail = [ln for ln in r.stdout.splitlines() if ln.startswith("Status:")]
                detail = tail[-1] if tail else r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            else:
                stderr = r.stderr.strip().splitlines()
                detail = stderr[-1] if stderr else f"rc={r.returncode}"
        except subprocess.TimeoutExpired:
            ok, detail = False, f"timeout after {args.timeout}s"
        return ip, entry, ok, detail

    results = list(ThreadPoolExecutor(max_workers=min(16, len(entries)))
                   .map(pull_one, entries))

    ok_count = 0
    for ip, entry, ok, detail in results:
        user = entry.get("user", "root")
        port = entry.get("ssh_port")
        tag = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        port_str = f":{port}" if port else ""
        suffix = f"  [{detail}]" if detail else ""
        print(f"  {tag}  {user}@{ip}{port_str}{suffix}")
        if ok:
            ok_count += 1

    print(f"\n{ok_count}/{len(results)} nodes pulled")
    sys.exit(0 if ok_count == len(results) else 1)


def cmd_ping(args: Ping) -> None:
    """SSH-echo every node in cluster.toml; report ✓/✗ per node."""
    if not CLUSTER_TOML.exists():
        print(f"No {CLUSTER_TOML}")
        sys.exit(1)
    with open(CLUSTER_TOML, "rb") as f:
        cfg = tomllib.load(f)
    entries = list(cfg.get("nodes", {}).items())
    if not entries:
        print(f"No nodes in {CLUSTER_TOML}")
        sys.exit(1)

    def probe(item: tuple[str, dict]) -> tuple[str, dict, bool, str]:
        ip, entry = item
        user = entry.get("user", "root")
        port = entry.get("ssh_port")
        argv = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={args.timeout}",
            "-o", "BatchMode=yes",
            "-i", str(Path(SSH_KEY).expanduser()),
        ]
        if port:
            argv += ["-p", str(port)]
        argv += [f"{user}@{ip}", "echo ok"]
        try:
            r = subprocess.run(argv, capture_output=True, text=True,
                               timeout=args.timeout + 5)
            ok = r.returncode == 0 and r.stdout.strip() == "ok"
            detail = "" if ok else (r.stderr.strip().splitlines()[-1]
                                    if r.stderr.strip() else f"rc={r.returncode}")
        except subprocess.TimeoutExpired:
            ok, detail = False, "timeout"
        return ip, entry, ok, detail

    results = list(ThreadPoolExecutor(max_workers=min(16, len(entries)))
                   .map(probe, entries))

    ok_count = 0
    for ip, entry, ok, detail in results:
        user = entry.get("user", "root")
        port = entry.get("ssh_port")
        roles = ",".join(entry.get("roles", [])) or "-"
        tag = "\033[32m✓\033[0m" if ok else "\033[31m✗\033[0m"
        port_str = f":{port}" if port else ""
        suffix = "" if ok else f"  [{detail}]"
        print(f"  {tag}  {user}@{ip}{port_str}  roles={roles}{suffix}")
        if ok:
            ok_count += 1

    print(f"\n{ok_count}/{len(results)} nodes reachable")
    sys.exit(0 if ok_count == len(results) else 1)


# --- CLI ---

Command = Add | Ping | Build | Pull | Status


def print_usage() -> None:
    print("usage: ./infra/cluster.py {add,ping,build,pull,status}")
    print()
    print("  add [--ssh-port PORT] <user@host>      Prep node for SSH docker-run jobs")
    print("  ping [--timeout SECS]                  SSH-echo each node in cluster.toml (✓/✗)")
    print("  build                                  Build & push Dockerfile.worker to GHCR")
    print("  pull [--timeout SECS]                  `docker pull` the worker image on every node")
    print("  status                                 Per-slot running-job report from NFS leases")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "add":
        if len(sys.argv) == 2 or sys.argv[2] in ("-h", "--help"):
            print("Usage: ./infra/cluster.py add [--ssh-port PORT] <user@host>")
            sys.exit(0)
        rest = sys.argv[2:]
        port: int | None = None
        filtered: list[str] = []
        i = 0
        while i < len(rest):
            if rest[i] == "--ssh-port" and i + 1 < len(rest):
                port = int(rest[i + 1])
                i += 2
            else:
                filtered.append(rest[i])
                i += 1
        if len(filtered) != 1:
            print("Error: Expected: add [--ssh-port PORT] <user@host>")
            sys.exit(1)
        cmd_add(Add(host=filtered[0], ssh_port=port))
    elif len(sys.argv) > 1 and sys.argv[1] == "build":
        cmd_build(Build())
    elif len(sys.argv) > 1 and sys.argv[1] == "status" and len(sys.argv) == 2:
        cmd_status(Status())
    elif len(sys.argv) == 1 or sys.argv[1] in ("-h", "--help"):
        print_usage()
        sys.exit(0 if len(sys.argv) > 1 else 1)
    else:
        args: Command = tyro.cli(Command)  # type: ignore[call-overload]
        if isinstance(args, Add):
            cmd_add(args)
        elif isinstance(args, Ping):
            cmd_ping(args)
        elif isinstance(args, Build):
            cmd_build(args)
        elif isinstance(args, Pull):
            cmd_pull(args)
        elif isinstance(args, Status):
            cmd_status(args)
