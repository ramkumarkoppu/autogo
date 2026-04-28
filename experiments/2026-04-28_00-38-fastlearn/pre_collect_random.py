"""Pre-collect iter0 training data: random vs random selfplay.

Writes 1000 games to /nfs/game_data_root/experiments/<EXP>/random-it0/, which
is the single dir referenced by dataset-it0.txt. iter0's `train.py` runs from
scratch on this — no MCTS visits in the NPZs, so the dataset loader falls
back to label-smoothed one-hot policy targets and the value head learns from
the (random-vs-random) game outcomes.

Runs locally on the controller; the random agent has no GPU/inference cost
so dispatching to the cluster isn't worth the round-trip. Saves into the
shared NFS dir so the train-role node sees it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXP_NAME = Path(__file__).resolve().parent.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num_games", type=int, default=5_000)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    save_name = f"experiments/{EXP_NAME}/random-it0"
    from alpha_go.self_play import main as self_play_main
    sys.argv = [
        "self_play",
        "--black", "random",
        "--white", "random",
        "--board_size", "9",
        "--num_games", str(args.num_games),
        "--num_workers", str(args.num_workers),
        "--save-name", save_name,
        "--seed", str(args.seed),
    ]
    print(f"=== Pre-collect {args.num_games} random-vs-random games -> {save_name} ===",
          flush=True)
    self_play_main()


if __name__ == "__main__":
    main()
