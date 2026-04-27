#!/usr/bin/env -S uv run python
"""League gauntlet collect driver for learngo-decen-collect-v7.

Each iteration N plays the current checkpoint against up to the last K
iterations on both sides, plus a pure selfplay set:

  - as-black-it{N}/vs-it{M}/  iter N (B) vs iter M (W),  M ∈ last K iters
  - as-white-it{N}/vs-it{M}/  iter M (B) vs iter N (W),  M ∈ last K iters
  - selfplay-it{N}/           iter N (B) vs iter N (W)

Defaults: K=3 opponents (6 gauntlet matchups) + 1 selfplay = 7 matchups,
50 games each (350 games / iter at K=3), sharded into ~16 jobs so workers
stay loaded. Selfplay games feed training data only; they're excluded from
update_league.py's win-rate aggregation since iter-vs-self is symmetric and
doesn't measure relative strength.

iter 0 has no prior iters; the gauntlet falls back to iter 0 vs iter 0 (so
the league bootstrap has data) and the selfplay set is collected as usual.

`league_state.json` (next to this file) tracks `best_black_iter` /
`best_white_iter` and the per-iter aggregate win rates against the gauntlet.
After collection, run `update_league.py` to recompute standings from the
just-collected NPZs.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

from infra.remote_exec import Job, load_cluster, run_pool

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
LEAGUE_STATE = EXP_DIR / "league_state.json"


def _ckpt(iteration: int) -> str:
    return f"/nfs/checkpoints/{EXP_NAME}/iter{iteration}_best.pt"


def _opponents(iteration: int, last_k: int) -> list[int]:
    """Last K iterations (exclusive of iteration). Falls back to [iteration]
    when there are no prior iters so iter 0 still produces selfplay data."""
    return list(range(max(0, iteration - last_k), iteration)) or [iteration]


def _chunk_distribution(num_matchups: int, target_jobs: int) -> list[int]:
    """Spread `target_jobs` over `num_matchups`; each matchup gets ≥ 1 chunk.

    Returns chunk counts per matchup. Sum is `target_jobs` when
    target_jobs ≥ num_matchups; otherwise the floor `num_matchups` (every
    matchup is represented even if that overshoots the target).
    """
    if target_jobs <= num_matchups:
        return [1] * num_matchups
    base = target_jobs // num_matchups
    remainder = target_jobs - base * num_matchups
    return [base + (1 if i < remainder else 0) for i in range(num_matchups)]


def _matchups(iteration: int, num_games: int, selfplay_games: int,
              last_k: int) -> list[dict]:
    """Enumerate every matchup for this iteration.

    Order: as-black gauntlet (per opponent) → as-white gauntlet (per
    opponent) → selfplay-it{N}. Selfplay is appended only when
    `selfplay_games > 0`. Each entry carries everything `build_jobs` needs
    to shard the matchup into chunks (save sub-dir, ckpts, total game count,
    name slug, and a `seed_id` so chunk seeds across matchups don't collide).
    """
    cur_ckpt = _ckpt(iteration)
    opponents = _opponents(iteration, last_k)
    out: list[dict] = []
    for side_id, side in enumerate(("as-black", "as-white")):
        for opp in opponents:
            opp_ckpt = _ckpt(opp)
            black_ckpt, white_ckpt = (
                (cur_ckpt, opp_ckpt) if side == "as-black"
                else (opp_ckpt, cur_ckpt)
            )
            out.append({
                "save_subdir": f"{side}-it{iteration}/vs-it{opp}",
                "black_ckpt": black_ckpt,
                "white_ckpt": white_ckpt,
                "num_games": num_games,
                "name_slug": f"{side}-vs-it{opp}",
                "seed_id": side_id * 1_000_000 + opp * 1_000,
            })
    if selfplay_games > 0:
        out.append({
            "save_subdir": f"selfplay-it{iteration}",
            "black_ckpt": cur_ckpt,
            "white_ckpt": cur_ckpt,
            "num_games": selfplay_games,
            "name_slug": "selfplay",
            "seed_id": 2_000_000,  # disjoint from any side_id*1M+opp*1K above
        })
    return out


def build_jobs(iteration: int, num_games: int, num_jobs: int,
               last_k: int, selfplay_games: int) -> list[Job]:
    """Build per-(matchup, chunk) Jobs.

    Matchups (see `_matchups`): K gauntlet opponents on each side plus one
    iter-vs-self selfplay set. The total `num_jobs` shards are spread across
    matchups so every matchup gets at least one chunk (see
    `_chunk_distribution`). Within a matchup, games split evenly with the
    remainder going to low-index chunks. Seeds deterministic in
    (iteration, matchup.seed_id, chunk_idx); game indices unique within each
    matchup's save dir.
    """
    matchups = _matchups(iteration, num_games, selfplay_games, last_k)
    chunks_per = _chunk_distribution(len(matchups), num_jobs)

    jobs: list[Job] = []
    for m, n_chunks in zip(matchups, chunks_per):
        save_dir = f"experiments/{EXP_NAME}/{m['save_subdir']}"
        host_save_dir = f"/nfs/game_data_root/{save_dir}"
        chunk_size = m["num_games"] // n_chunks
        remainder = m["num_games"] - chunk_size * n_chunks
        offset = 0
        for chunk_idx in range(n_chunks):
            games = chunk_size + (1 if chunk_idx < remainder else 0)
            if games == 0:
                continue
            seed = iteration * 10_000_000 + m["seed_id"] + chunk_idx
            cmd = (
                f"uv run experiments/{EXP_NAME}/run_games.py "
                f"--black-checkpoint {shlex.quote(m['black_ckpt'])} "
                f"--white-checkpoint {shlex.quote(m['white_ckpt'])} "
                f"--num_games {games} "
                f"--save-name {shlex.quote(save_dir)} "
                f"--seed {seed} "
                f"--game_index_offset {offset}"
            )
            # Dedup so iter-vs-self only rsyncs one file.
            push = tuple(dict.fromkeys((m["black_ckpt"], m["white_ckpt"])))
            jobs.append(Job(
                name=f"it{iteration}-{m['name_slug']}-c{chunk_idx}",
                inner_cmd=cmd,
                push_files=push,
                pull_dirs=(host_save_dir,),
            ))
            offset += games
    return jobs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    p.add_argument("--num-games", type=int, default=50,
                   help="Games per matchup (per opponent per side).")
    p.add_argument("--selfplay-games", type=int, default=50,
                   help="Games of iter-vs-self saved to selfplay-it{N}/. "
                        "Set 0 to skip the selfplay matchup.")
    p.add_argument("--num-jobs", type=int, default=12,
                   help="Target shard count per iter; spread across matchups "
                        "(gauntlet + selfplay; every matchup gets ≥ 1 chunk).")
    p.add_argument("--last-k", type=int, default=3,
                   help="Number of prior iterations to play as opponents.")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    opponents = _opponents(args.iteration, args.last_k)
    Path(f"/nfs/game_data_root/experiments/{EXP_NAME}").mkdir(parents=True, exist_ok=True)

    workers, image = load_cluster("collect")
    if not workers:
        print("ERROR: no node with role 'collect' in cluster.toml", file=sys.stderr)
        return 1

    jobs = build_jobs(
        iteration=args.iteration,
        num_games=args.num_games,
        num_jobs=args.num_jobs,
        last_k=args.last_k,
        selfplay_games=args.selfplay_games,
    )

    total_gpus = sum(max(w.num_gpu, 1) for w in workers)
    num_matchups = 2 * len(opponents) + (1 if args.selfplay_games > 0 else 0)
    total_games = 2 * len(opponents) * args.num_games + args.selfplay_games
    print(f"=== Iter {args.iteration} collect: {len(jobs)} jobs across "
          f"{len(workers)} hosts / {total_gpus} gpu-workers ===")
    print(f"opponents (last K={args.last_k}): {opponents}")
    print(f"matchups: {2*len(opponents)} gauntlet + "
          f"{1 if args.selfplay_games > 0 else 0} selfplay = {num_matchups}")
    print(f"per-matchup: {args.num_games} gauntlet / {args.selfplay_games} selfplay  "
          f"(target {args.num_jobs} jobs over {num_matchups} matchups "
          f"-> {len(jobs)} actual jobs, {total_games} games / iter)")
    print(f"image={image}  gpu={'no' if args.no_gpu else 'yes'}")
    for w in workers:
        gpu_tag = f" gpu_type={w.gpu_type}" if w.gpu_type else ""
        print(f"  worker: {w.target}  num_gpu={w.num_gpu}{gpu_tag}")

    logs_dir = EXP_DIR / "logs" / f"it{args.iteration}"
    logs_dir.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()
    results = run_pool(workers, image, not args.no_gpu, jobs, logs_dir,
                       role="collect", exp_name=EXP_NAME, per_gpu=True,
                       share_cluster=True)
    end_ts = time.time()

    failed = [n for n, rc in results.items() if rc != 0]
    missing = [j.name for j in jobs if j.name not in results]
    print(f"\n=== Done: {len(results) - len(failed)}/{len(jobs)} jobs OK ===")
    if missing:
        print(f"NEVER RAN: {missing}")
    if failed:
        print(f"FAILED: {failed}")

    timing_dir = EXP_DIR / "timing"
    timing_dir.mkdir(exist_ok=True)
    stats_path = timing_dir / f"collect-it{args.iteration}.json"
    tmp = stats_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump({
            "iteration": args.iteration,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "elapsed_seconds": round(end_ts - start_ts, 1),
            "num_jobs": len(jobs),
            "num_ok": len(results) - len(failed),
            "num_failed": len(failed),
            "num_missing": len(missing),
            "modes": (["as-black", "as-white"]
                      + (["selfplay"] if args.selfplay_games > 0 else [])),
            "opponents": opponents,
            "last_k": args.last_k,
            "num_games_per_matchup": args.num_games,
            "selfplay_games": args.selfplay_games,
            "target_num_jobs": args.num_jobs,
            "total_games": total_games,
        }, f, indent=2)
    tmp.replace(stats_path)

    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
