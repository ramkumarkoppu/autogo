#!/usr/bin/env -S uv run python
"""Update `league_state.json` after iter N's champion-only collect.

Walks /nfs/game_data_root/experiments/<EXP>/{as-black,as-white}-it{N}/vs-it*/
NPZs to compute iter N's win rate as black against the reigning best-white
champion and as white against the reigning best-black champion. iter N
dethrones the current champions (becoming the new champion for *both*
colors) only if BOTH win rates strictly exceed 0.55.

iter 0 bootstraps the league: it's the champion for both colors with no
gauntlet games required (collect_driver only writes selfplay for iter 0).
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

PROMOTION_THRESHOLD = 0.55

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

    if n == 0:
        if state["best_black_iter"] is not None or state["best_white_iter"] is not None:
            print(f"WARN: league_state.json already initialised "
                  f"(best_black={state['best_black_iter']}, "
                  f"best_white={state['best_white_iter']}); re-bootstrapping iter 0.",
                  file=sys.stderr)
        state["best_black_iter"] = 0
        state["best_white_iter"] = 0
        state["by_iter"]["0"] = {
            "as_black_wr": None, "as_white_wr": None,
            "as_black_games": 0, "as_white_games": 0,
            "challenged_black": None, "challenged_white": None,
            "promoted": True,
            "by_opponent": {},
        }
        state["history"].append({
            "iter": 0, "best_black": 0, "best_white": 0,
            "as_black_wr": None, "as_white_wr": None, "promoted": True,
        })
        _save(state)
        print(f"iter0: bootstrapped league (champion for both colors)")
        return

    prev_black = state["best_black_iter"]
    prev_white = state["best_white_iter"]
    if prev_black is None or prev_white is None:
        print(f"ERROR: iter {n} > 0 but no champions yet — bootstrap iter 0 first.",
              file=sys.stderr)
        sys.exit(1)

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

    as_black_wr = bb_wins / bb_total                   # iter N is black, winner==1 → win
    as_white_wr = (ww_total - ww_wins) / ww_total      # iter N is white, winner==2 → win

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

    promoted = (as_black_wr > PROMOTION_THRESHOLD
                and as_white_wr > PROMOTION_THRESHOLD)
    new_black = n if promoted else prev_black
    new_white = n if promoted else prev_white

    state["by_iter"][str(n)] = {
        "as_black_wr": round(as_black_wr, 4),
        "as_white_wr": round(as_white_wr, 4),
        "as_black_games": bb_total,
        "as_white_games": ww_total,
        "challenged_black": prev_black,
        "challenged_white": prev_white,
        "promoted": promoted,
        "by_opponent": by_opponent,
    }
    state["best_black_iter"] = new_black
    state["best_white_iter"] = new_white
    state["history"].append({
        "iter": n, "best_black": new_black, "best_white": new_white,
        "as_black_wr": round(as_black_wr, 4),
        "as_white_wr": round(as_white_wr, 4),
        "promoted": promoted,
    })
    _save(state)

    verdict = ("PROMOTED to champion" if promoted
               else f"stays under champs (need >{PROMOTION_THRESHOLD:.2f} on both sides)")
    print(f"iter{n}: as_black_wr={as_black_wr:.3f} ({bb_wins}/{bb_total}) vs iter{prev_white}  "
          f"as_white_wr={as_white_wr:.3f} ({ww_total-ww_wins}/{ww_total}) vs iter{prev_black}  "
          f"-> {verdict}")
    print(f"league: best_black=iter{new_black}  best_white=iter{new_white}  "
          f"-> {STATE_FILE.name}")


if __name__ == "__main__":
    main()
