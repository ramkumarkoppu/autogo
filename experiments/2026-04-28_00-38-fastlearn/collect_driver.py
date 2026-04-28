#!/usr/bin/env -S uv run python
"""Champion-only collect driver.

Each iteration N plays the current iter-N checkpoint against the reigning
league champions:

  - as-black-it{N}/vs-it{best_white}/  iter N (B) vs current best-white (W)
  - as-white-it{N}/vs-it{best_black}/  current best-black (B) vs iter N (W)
  - selfplay-it{N}/                    iter N (B) vs iter N (W)

`league_state.json` (next to this script) is the source of truth for the
current champions. iter 0 has no league yet; we skip the gauntlet matchups
and only collect the selfplay set (`update_league.py` then bootstraps iter 0
as the champion of both colors).

Defaults: 50 games per matchup → 150 games / iter (50 selfplay-only on
iter 0), sharded into ~12 jobs (every matchup gets ≥ 1 chunk) so the cluster
stays loaded. Selfplay games are ignored by `update_league.py`'s win-rate
aggregation (iter-vs-self is symmetric).
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


def _load_champions() -> tuple[int | None, int | None]:
    """Return (best_black_iter, best_white_iter) from league_state.json, or
    (None, None) if no league state exists yet (iter 0 bootstrap)."""
    if not LEAGUE_STATE.exists():
        return None, None
    state = json.loads(LEAGUE_STATE.read_text())
    return state.get("best_black_iter"), state.get("best_white_iter")


def _chunk_distribution(num_matchups: int, target_jobs: int) -> list[int]:
    """Spread `target_jobs` over `num_matchups`; each matchup gets ≥ 1 chunk."""
    if target_jobs <= num_matchups:
        return [1] * num_matchups
    base = target_jobs // num_matchups
    remainder = target_jobs - base * num_matchups
    return [base + (1 if i < remainder else 0) for i in range(num_matchups)]


def _matchups(iteration: int, num_games: int, selfplay_games: int,
              best_black: int | None, best_white: int | None) -> list[dict]:
    """Enumerate every matchup for this iteration.

    Order: as-black vs best-white → as-white vs best-black → selfplay-it{N}.
    Gauntlet matchups are skipped when the league is empty (iter 0 bootstrap).
    """
    cur_ckpt = _ckpt(iteration)
    out: list[dict] = []
    if best_white is not None and num_games > 0:
        out.append({
            "save_subdir": f"as-black-it{iteration}/vs-it{best_white}",
            "black_ckpt": cur_ckpt,
            "white_ckpt": _ckpt(best_white),
            "num_games": num_games,
            "name_slug": f"as-black-vs-it{best_white}",
            "seed_id": 1_000 + best_white,
        })
    if best_black is not None and num_games > 0:
        out.append({
            "save_subdir": f"as-white-it{iteration}/vs-it{best_black}",
            "black_ckpt": _ckpt(best_black),
            "white_ckpt": cur_ckpt,
            "num_games": num_games,
            "name_slug": f"as-white-vs-it{best_black}",
            "seed_id": 1_000_000 + best_black,
        })
    if selfplay_games > 0:
        out.append({
            "save_subdir": f"selfplay-it{iteration}",
            "black_ckpt": cur_ckpt,
            "white_ckpt": cur_ckpt,
            "num_games": selfplay_games,
            "name_slug": "selfplay",
            "seed_id": 2_000_000,
        })
    return out


def build_jobs(iteration: int, num_games: int, num_jobs: int,
               selfplay_games: int, best_black: int | None,
               best_white: int | None) -> list[Job]:
    """Build per-(matchup, chunk) Jobs.

    The `num_jobs` shards spread across matchups so every matchup gets at
    least one chunk (see `_chunk_distribution`). Within a matchup, games
    split evenly with the remainder going to low-index chunks. Seeds are
    deterministic in (iteration, matchup.seed_id, chunk_idx); game indices
    are unique within each matchup's save dir.
    """
    matchups = _matchups(iteration, num_games, selfplay_games,
                         best_black, best_white)
    if not matchups:
        return []
    chunks_per = _chunk_distribution(len(matchups), num_jobs)

    jobs: list[Job] = []
    for m, n_chunks in zip(matchups, chunks_per):
        base_save_dir = f"experiments/{EXP_NAME}/{m['save_subdir']}"
        chunk_size = m["num_games"] // n_chunks
        remainder = m["num_games"] - chunk_size * n_chunks
        offset = 0
        for chunk_idx in range(n_chunks):
            games = chunk_size + (1 if chunk_idx < remainder else 0)
            if games == 0:
                continue
            # Per-chunk subdir so each job's pull_dir is disjoint from its
            # siblings on the same host. Without this, sibling chunks share a
            # remote scratch dir and one chunk's post-job `rm -rf` deletes
            # files mid-rsync from another, returning rc=23 and triggering a
            # spurious whole-job retry. Training/league walkers use rglob so
            # the extra nesting is transparent.
            save_dir = f"{base_save_dir}/c{chunk_idx}"
            host_save_dir = f"/nfs/game_data_root/{save_dir}"
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
                   help="Games per gauntlet matchup (vs each champion).")
    p.add_argument("--selfplay-games", type=int, default=50,
                   help="Games of iter-vs-self saved to selfplay-it{N}/. "
                        "Set 0 to skip the selfplay matchup.")
    p.add_argument("--num-jobs", type=int, default=20,
                   help="Target shard count per iter; spread across matchups "
                        "(every matchup gets ≥ 1 chunk).")
    p.add_argument("--no-gpu", action="store_true")
    args = p.parse_args()

    best_black, best_white = _load_champions()
    if args.iteration == 0:
        # Iter 0 has no league yet; only collect selfplay so the bootstrap
        # has data and update_league.py can crown iter 0.
        best_black = best_white = None
    else:
        if best_black is None or best_white is None:
            print(f"ERROR: iter {args.iteration} > 0 but league_state.json missing "
                  f"or has no champions yet. Run update_league.py for prior iters first.",
                  file=sys.stderr)
            return 1

    Path(f"/nfs/game_data_root/experiments/{EXP_NAME}").mkdir(parents=True, exist_ok=True)

    workers, image = load_cluster("collect")
    if not workers:
        print("ERROR: no node with role 'collect' in cluster.toml", file=sys.stderr)
        return 1

    jobs = build_jobs(
        iteration=args.iteration,
        num_games=args.num_games,
        num_jobs=args.num_jobs,
        selfplay_games=args.selfplay_games,
        best_black=best_black,
        best_white=best_white,
    )
    if not jobs:
        print(f"ERROR: no jobs built for iter {args.iteration}", file=sys.stderr)
        return 1

    total_gpus = sum(max(w.num_gpu, 1) for w in workers)
    has_gauntlet = best_black is not None and best_white is not None
    num_matchups = (2 if has_gauntlet else 0) + (1 if args.selfplay_games > 0 else 0)
    total_games = (2 * args.num_games if has_gauntlet else 0) + args.selfplay_games
    print(f"=== Iter {args.iteration} collect: {len(jobs)} jobs across "
          f"{len(workers)} hosts / {total_gpus} gpu-workers ===")
    if has_gauntlet:
        print(f"champions: best_black=iter{best_black}  best_white=iter{best_white}")
        print(f"matchups: 2 gauntlet (vs champs) + "
              f"{1 if args.selfplay_games > 0 else 0} selfplay = {num_matchups}")
    else:
        print(f"iter 0 bootstrap: selfplay only ({args.selfplay_games} games)")
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
            "modes": ((["as-black", "as-white"] if has_gauntlet else [])
                      + (["selfplay"] if args.selfplay_games > 0 else [])),
            "best_black": best_black,
            "best_white": best_white,
            "num_games_per_matchup": args.num_games,
            "selfplay_games": args.selfplay_games,
            "target_num_jobs": args.num_jobs,
            "total_games": total_games,
        }, f, indent=2)
    tmp.replace(stats_path)

    return 1 if (failed or missing) else 0


if __name__ == "__main__":
    sys.exit(main())
