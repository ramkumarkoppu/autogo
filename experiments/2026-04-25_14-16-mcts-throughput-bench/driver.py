#!/usr/bin/env -S uv run python
"""Fan out the four throughput benchmarks across collect-role nodes.

One job per mode, one GPU each, all on rtx6000_ada (the only gpu_type in
cluster.toml at the time of writing) so the bars are comparable.
"""
from __future__ import annotations

import argparse
import shlex
import sys
import time
from pathlib import Path

from infra.remote_exec import Job, load_cluster, run_pool

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
DEFAULT_CKPT = "/nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt"
MODES = ["py-mcts", "cpp-mcts-seq", "cpp-batched", "cpp-batched-leaf"]


def build_jobs(checkpoint: str, num_games: int, max_moves: int,
               num_simulations: int, leaf_batch_size: int) -> list[Job]:
    results_host_dir = f"/nfs/game_data_root/experiments/{EXP_NAME}/results"
    jobs: list[Job] = []
    for mode in MODES:
        out_remote = f"experiments/{EXP_NAME}/results/{mode}.json"
        out_host = f"/nfs/game_data_root/experiments/{EXP_NAME}/results"
        # Modes 1+2 are single-game; 3+4 use num_games games in parallel.
        ng = 1 if mode in ("py-mcts", "cpp-mcts-seq") else num_games
        cmd = (
            f"uv run experiments/{EXP_NAME}/benchmark.py "
            f"--mode {mode} "
            f"--checkpoint {shlex.quote(checkpoint)} "
            f"--num-games {ng} "
            f"--max-moves {max_moves} "
            f"--num-simulations {num_simulations} "
            f"--leaf-batch-size {leaf_batch_size} "
            f"--out /nfs/game_data_root/{shlex.quote(out_remote)}"
        )
        jobs.append(Job(name=mode, inner_cmd=cmd,
                        push_files=(checkpoint,),
                        pull_dirs=(out_host,)))
    return jobs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--num-games", type=int, default=8,
                   help="Parallel games for batched modes.")
    p.add_argument("--max-moves", type=int, default=40)
    p.add_argument("--num-simulations", type=int, default=1024)
    p.add_argument("--leaf-batch-size", type=int, default=8)
    args = p.parse_args()

    Path(f"/nfs/game_data_root/experiments/{EXP_NAME}/results").mkdir(
        parents=True, exist_ok=True)

    workers, image = load_cluster("collect")
    if not workers:
        print("ERROR: no node with role 'collect' in cluster.toml", file=sys.stderr)
        return 1

    jobs = build_jobs(args.checkpoint, args.num_games, args.max_moves,
                      args.num_simulations, args.leaf_batch_size)
    print(f"=== {len(jobs)} benchmark jobs ===")
    for w in workers:
        print(f"  worker: {w.target}  num_gpu={w.num_gpu} gpu_type={w.gpu_type}")
    for j in jobs:
        print(f"  job: {j.name}")

    logs_dir = EXP_DIR / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    priorities = EXP_DIR / "job_priorities.txt"
    t0 = time.time()
    results = run_pool(workers, image, gpu=True, jobs=jobs, logs_dir=logs_dir,
                       role="collect", exp_name=EXP_NAME, per_gpu=True,
                       share_cluster=True, priorities_file=priorities)
    elapsed = time.time() - t0

    failed = [n for n, rc in results.items() if rc != 0]
    missing = [j.name for j in jobs if j.name not in results]
    print(f"\n=== Done in {elapsed:.1f}s: "
          f"{len(results) - len(failed)}/{len(jobs)} jobs OK ===")
    if missing:
        print(f"NEVER RAN: {missing}")
    if failed:
        print(f"FAILED: {failed}")
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
