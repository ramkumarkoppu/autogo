#!/usr/bin/env -S uv run python
"""Update `league_state.json` after iter N's gauntlet collect.

Walks /nfs/game_data_root/experiments/<EXP>/{as-black,as-white}-it{N}/vs-it*/
NPZs, aggregates iter N's wins across every opponent on each side, and
updates the league standings. iter N becomes the new best_{color} only if
its aggregate win rate strictly exceeds the recorded aggregate win rate of
the current best_{color} (measured when *its* iteration ran).

Note: the comparison isn't strictly fair across iterations — different iters
may face different opponent sets — but it matches the user-specified rule
("if it wins more often, it dethrones") and biases toward keeping earlier
champions on ties.

iter 0 bootstraps the state: it's the league best for both colors by
default, with its self-vs-self win rates as the baselines.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(f"/nfs/game_data_root/experiments/{EXP_NAME}")
STATE_FILE = EXP_DIR / "league_state.json"

_OPP_RE = re.compile(r"^vs-it(\d+)$")


def _walk_side(side_root: Path) -> dict[int, tuple[int, int]]:
    """Return {opponent_iter: (black_wins, total_games)} for one side dir."""
    out: dict[int, tuple[int, int]] = {}
    if not side_root.exists():
        return out
    for opp_dir in sorted(side_root.iterdir()):
        m = _OPP_RE.match(opp_dir.name)
        if not m or not opp_dir.is_dir():
            continue
        opp = int(m.group(1))
        wins = total = 0
        for npz in sorted(opp_dir.rglob("*.npz")):
            d = dict(np.load(npz, allow_pickle=True))
            if "winner" not in d:
                continue
            winner = int(d["winner"])
            if winner == 1:
                wins += 1
            if winner in (1, 2):
                total += 1
        out[opp] = (wins, total)
    return out


def _load() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"best_black_iter": None, "best_white_iter": None,
            "history": [], "by_iter": {}}


def _save(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2))
    tmp.replace(STATE_FILE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--iteration", type=int, required=True)
    args = p.parse_args()
    n = args.iteration

    state = _load()

    as_black = _walk_side(GAME_DATA_DIR / f"as-black-it{n}")  # iter N is black
    as_white = _walk_side(GAME_DATA_DIR / f"as-white-it{n}")  # iter N is white
    if not as_black or not as_white:
        print(f"ERROR: missing matchup dirs for iter {n} "
              f"(as-black={list(as_black)}, as-white={list(as_white)})",
              file=sys.stderr)
        sys.exit(1)

    bb_wins = sum(w for w, _ in as_black.values())
    bb_total = sum(t for _, t in as_black.values())
    ww_wins = sum(w for w, _ in as_white.values())
    ww_total = sum(t for _, t in as_white.values())
    if bb_total == 0 or ww_total == 0:
        print(f"ERROR: zero-game side for iter {n} "
              f"(as-black games={bb_total}, as-white games={ww_total})",
              file=sys.stderr)
        sys.exit(1)

    as_black_wr = bb_wins / bb_total                     # iter N is black
    as_white_wr = (ww_total - ww_wins) / ww_total        # iter N is white

    by_opponent: dict[str, dict] = {}
    for opp, (w, t) in as_black.items():
        by_opponent.setdefault(str(opp), {})["as_black"] = {
            "iter_n_wins": w, "games": t,
            "iter_n_wr": round(w / t, 4) if t else None,
        }
    for opp, (w, t) in as_white.items():
        by_opponent.setdefault(str(opp), {})["as_white"] = {
            "iter_n_wins": t - w, "games": t,
            "iter_n_wr": round((t - w) / t, 4) if t else None,
        }

    state["by_iter"][str(n)] = {
        "as_black_wr": round(as_black_wr, 4),
        "as_white_wr": round(as_white_wr, 4),
        "as_black_games": bb_total,
        "as_white_games": ww_total,
        "opponents": sorted({int(o) for o in by_opponent}),
        "by_opponent": by_opponent,
    }

    if state["best_black_iter"] is None:
        if n != 0:
            print(f"ERROR: empty league state but iter={n}", file=sys.stderr)
            sys.exit(1)
        new_black = new_white = 0
    else:
        prev_black = int(state["best_black_iter"])
        prev_white = int(state["best_white_iter"])
        prev_black_wr = state["by_iter"][str(prev_black)]["as_black_wr"]
        prev_white_wr = state["by_iter"][str(prev_white)]["as_white_wr"]
        new_black = n if as_black_wr > prev_black_wr else prev_black
        new_white = n if as_white_wr > prev_white_wr else prev_white

    state["best_black_iter"] = new_black
    state["best_white_iter"] = new_white
    state["history"].append({"iter": n, "best_black": new_black, "best_white": new_white})
    _save(state)

    opps = sorted(by_opponent, key=int)
    print(f"iter{n}: as_black_wr={as_black_wr:.3f} ({bb_wins}/{bb_total}) "
          f"as_white_wr={as_white_wr:.3f} ({ww_total-ww_wins}/{ww_total}) "
          f"opponents={opps}")
    print(f"league: best_black=iter{new_black}  best_white=iter{new_white}  "
          f"-> {STATE_FILE.name}")


if __name__ == "__main__":
    main()
