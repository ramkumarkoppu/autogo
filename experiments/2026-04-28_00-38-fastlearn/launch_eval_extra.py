#!/usr/bin/env -S uv run python
"""Launch evaluation games for this experiment's trained checkpoints against
KataGo opponents (default: katago-human-9d).

For each checkpoint in [start_iter, end_iter] × each mode in --modes, plays
`num_games` games per color through run_eval_games.py. Results land in
`experiments/{EXP_NAME}/eval-katago-<...>-it{N}/` (the `eval-` prefix is what
analyze.py expects to tag these series — collect_driver.py for this experiment
never produces eval-* dirs, so there's no collision).

Defaults to 20 games / color (40 / iter / mode) for fastlearn — half what the
parent fork uses, since this experiment trains many more checkpoints.

Uses `run_pool(share_cluster=True)` so this launcher coexists with a
concurrently-running collect driver on the same cluster without double-
allocating GPUs.
"""
from __future__ import annotations

import argparse
import random
import shlex
import sys
import time
from pathlib import Path

from infra.remote_exec import Job, load_cluster, run_pool

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name

EVAL_MODES = [
    "eval-katago-human",
]

def _save_prefix(mode: str) -> str:
    """`eval-katago-human` -> `eval-katago-human` (analyze.py-compatible)."""
    return mode


def build_jobs(
    iterations: list[int],
    num_games: int,
    chunks_per_mode: int,
    modes: list[str],
    game_index_base: int,
    run_tag: str,
) -> list[Job]:
    """One chunk-per-color Job per (iter, mode, color, chunk).

    Total games per (iter, mode): 2 * num_games (num_games black + num_games
    white), split across `chunks_per_mode` chunks per color.

    `game_index_base` shifts every chunk's --game_index_offset by a constant
    so a second concurrent launch can stack new games on top of an already-
    running batch without colliding on save-dir game indices. `run_tag` is
    folded into the job names and seed namespace so two pools targeting the
    same iters/modes don't duplicate-name each other.
    """
    chunk_size = num_games // chunks_per_mode
    remainder = num_games - chunk_size * chunks_per_mode
    # Stable per-tag seed offset so a second pool's seeds don't collide with
    # the first's. 0 for the empty/default tag preserves prior seed math.
    tag_seed = 0 if not run_tag else (abs(hash(run_tag)) % 1_000) * 10_000
    jobs: list[Job] = []
    for iteration in iterations:
        checkpoint = f"/nfs/checkpoints/{EXP_NAME}/iter{iteration}_best.pt"
        for mode_id, mode in enumerate(modes):
            save_dir = f"experiments/{EXP_NAME}/{_save_prefix(mode)}-it{iteration}"
            host_save_dir = f"/nfs/game_data_root/{save_dir}"
            for color_id, (color, name_suffix, color_offset) in enumerate([
                ("black", "-b", 0),
                ("white", "-w", num_games),
            ]):
                offset = game_index_base + color_offset
                for chunk_idx in range(chunks_per_mode):
                    games = chunk_size + (1 if chunk_idx < remainder else 0)
                    if games == 0:
                        continue
                    seed = (9_000_000
                            + tag_seed
                            + iteration * 100_000
                            + mode_id * 1_000
                            + color_id * 100
                            + chunk_idx)
                    tag_suffix = f"-{run_tag}" if run_tag else ""
                    name = f"eval-it{iteration}-{mode}-c{chunk_idx}{name_suffix}{tag_suffix}"
                    cmd = (
                        f"uv run experiments/{EXP_NAME}/run_eval_games.py "
                        f"--mode {mode} "
                        f"--color {color} "
                        f"--checkpoint {shlex.quote(checkpoint)} "
                        f"--num_games {games} "
                        f"--save-name {shlex.quote(save_dir)} "
                        f"--seed {seed} "
                        f"--game_index_offset {offset}"
                    )
                    jobs.append(Job(
                        name=name,
                        inner_cmd=cmd,
                        push_files=(checkpoint,),
                        pull_dirs=(host_save_dir,),
                    ))
                    offset += games
    return jobs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start-iter", type=int, default=0)
    p.add_argument("--end-iter", type=int, default=10,
                   help="Inclusive upper bound on iteration index.")
    p.add_argument("--num-games", type=int, default=20,
                   help="Games per color per (iter, mode). Default 20 for fastlearn.")
    p.add_argument("--chunks-per-mode", type=int, default=4,
                   help="Chunks per color; each chunk is one run_eval_games.py job.")
    p.add_argument("--modes", type=str, default=",".join(EVAL_MODES),
                   help="Comma-separated eval modes (see run_eval_games.py MODE_OPPONENT).")
    p.add_argument("--shuffle-seed", type=int, default=0,
                   help="Seed for the job-order shuffle. Jobs are shuffled "
                        "before dispatch so games from every (iter, mode, color) "
                        "combination arrive interleaved rather than iteration-"
                        "by-iteration.")
    p.add_argument("--game-index-base", type=int, default=0,
                   help="Constant added to every chunk's --game_index_offset. "
                        "Set to (prior --num-games * 2) when stacking a second "
                        "pool on top of an already-running batch so the new "
                        "games claim fresh slots in each save dir.")
    p.add_argument("--run-tag", type=str, default="",
                   help="Suffix added to job names, priorities file, and logs "
                        "dir so a second concurrent launcher doesn't collide "
                        "with the first. Required when --game-index-base != 0.")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()
    if args.game_index_base != 0 and not args.run_tag:
        print("ERROR: --run-tag is required when --game-index-base != 0",
              file=sys.stderr)
        return 1

    iterations = list(range(args.start_iter, args.end_iter + 1))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    iterations = [
        i for i in iterations
        if Path(f"/nfs/checkpoints/{EXP_NAME}/iter{i}_best.pt").exists()
    ]
    if not iterations:
        print("ERROR: no iter{N}_best.pt checkpoints found in "
              f"/nfs/checkpoints/{EXP_NAME}/", file=sys.stderr)
        return 1

    Path(f"/nfs/game_data_root/experiments/{EXP_NAME}").mkdir(parents=True, exist_ok=True)

    workers, image = load_cluster("collect")
    if not workers:
        print("ERROR: no node with role 'collect' in cluster.toml", file=sys.stderr)
        return 1

    jobs = build_jobs(
        iterations=iterations,
        num_games=args.num_games,
        chunks_per_mode=args.chunks_per_mode,
        modes=modes,
        game_index_base=args.game_index_base,
        run_tag=args.run_tag,
    )
    random.Random(args.shuffle_seed).shuffle(jobs)

    total_gpus = sum(max(w.num_gpu, 1) for w in workers)
    print(f"=== eval-extra: {len(jobs)} jobs across {len(workers)} hosts / "
          f"{total_gpus} gpu-workers ===")
    print(f"iterations={iterations}")
    print(f"modes={modes}")
    print(f"image={image}  gpu={'no' if args.no_gpu else 'yes'}  "
          f"share_cluster=True")
    for w in workers:
        gpu_tag = f" gpu_type={w.gpu_type}" if w.gpu_type else ""
        print(f"  worker: {w.target}  num_gpu={w.num_gpu}{gpu_tag}")

    tag = f"-{args.run_tag}" if args.run_tag else ""
    logs_dir = EXP_DIR / "logs" / f"eval_extra{tag}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    priorities_file = EXP_DIR / f"job_priorities-eval-extra{tag}.txt"
    start_ts = time.time()
    results = run_pool(
        workers, image, not args.no_gpu, jobs, logs_dir,
        role="collect", exp_name=EXP_NAME,
        per_gpu=True, share_cluster=True,
        priorities_file=priorities_file,
    )
    elapsed = time.time() - start_ts

    failed = [n for n, rc in results.items() if rc != 0]
    missing = [j.name for j in jobs if j.name not in results]
    print(f"\n=== Done in {elapsed/60:.1f} min: "
          f"{len(results) - len(failed)}/{len(jobs)} jobs OK ===")
    if missing:
        print(f"NEVER RAN: {missing}")
    if failed:
        print(f"FAILED: {failed}")
    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
