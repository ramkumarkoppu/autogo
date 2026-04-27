# learngo-decen-collect-v7

Self-play AlphaGo Zero–style training loop on 9×9 Go, dispatched across an
SSH cluster of one or more GPU workers (NFS-shared or not). Iter0 is trained from scratch on 10000 random-vs-random
games and the loop bootstraps itself from selfplay thereafter.

## How it works

Each iteration is three steps:

1. **Collect (gauntlet + selfplay)** — `collect_driver.py` plays iter N
   against the last `K=3` iterations on both sides AND a pure selfplay set
   (no league-state lookup; the matchup list is fully determined by N and K):
     - `as-black-it{N}/vs-it{M}/` — iter N (B) vs iter M (W) for M ∈ last K
     - `as-white-it{N}/vs-it{M}/` — iter M (B) vs iter N (W) for M ∈ last K
     - `selfplay-it{N}/`         — iter N (B) vs iter N (W)
   Defaults: K=3 gauntlet opponents × 2 sides + 1 selfplay = 7 matchups
   × 50 games / matchup ⇒ 350 games / iter, sharded into ~12 jobs (chunk
   counts spread across matchups, every matchup ≥ 1 chunk) so the cluster
   stays loaded. iter 0 has no prior iters, so the gauntlet falls back to
   iter 0 vs iter 0 (giving the league bootstrap baseline) and the
   selfplay set is collected as usual.
2. **Update league** — `update_league.py` aggregates iter N's wins across
   every gauntlet opponent on each side and updates the champions in
   `league_state.json`. Selfplay games are ignored here (iter-vs-self is
   symmetric and doesn't measure relative strength). iter N becomes the
   new best_{color} only if its aggregate gauntlet win rate strictly
   exceeds the recorded aggregate win rate of the current champion
   (measured when *its* iteration ran).
3. **Train** — `train.py` loads every NPZ in `dataset-it{N+1}.txt` into RAM
   and trains the next-iter checkpoint with policy CE (teacher-masked) +
   value BCE.

Architecture: MuP-parameterized GoResNet (`channels=256`, `n_blocks=14`,
~18M params). PCR 95/5 1024/2048 sims; LR 1.5e-3 cosine; weight decay 5e-3;
batch 512; 15-min time budget per iter or train policy acc ≥ 95%.

## Decentralized collect workers

Workers may share `/nfs` with the controller or not. `cluster.toml` nodes
carry a `shares_nfs` bool (default true); `infra.remote_exec.Job` carries
`push_files` and `pull_dirs`. For non-NFS workers, `run_pool` rsyncs each
`push_files` entry over before the docker run and, on `rc == 0`, rsyncs each
`pull_dirs` entry back and `rm -rf`s the remote scratch copy. By the time
`collect_driver.py` returns, the controller's `/nfs/game_data_root/...`
already has every successful chunk's NPZs.

`collect_driver.py::build_jobs` tags every job with
`push_files=(host_ckpt,)` and `pull_dirs=(host_save_dir,)` — paths under
`/nfs/...` resolve identically on NFS hosts and on non-NFS hosts (where
`infra/cluster.py add` symlinks `/data/eric → /nfs` and seeds the directory
layout).

## Data dirs per iteration

- `experiments/<EXP>/random-it0/`               — bootstrap random-vs-random
                                                   games (1000 games; only
                                                   referenced by dataset-it0.txt)
- `experiments/<EXP>/as-black-it{N}/vs-it{M}/`  — iter N (B) vs iter M (W)
- `experiments/<EXP>/as-white-it{N}/vs-it{M}/`  — iter M (B) vs iter N (W)
- `experiments/<EXP>/selfplay-it{N}/`           — iter N (B) vs iter N (W)

`dataset-it0.txt` points at `random-it0/` and is used only for the iter0
train. `dataset-it1.txt` is built fresh from `as-black-it0/` + `as-white-it0/`
+ `selfplay-it0/` (random-it0 is intentionally dropped); `dataset-it{N+1}.txt`
for N ≥ 1 carries forward every path from the previous iteration's dataset
and appends those same three dirs for iter N. The dataset loader recurses
via `rglob`, so listing the gauntlet parent dir picks up every opponent's
sub-dir NPZs.

## League state

`league_state.json` (next to the scripts) is the source of truth for the
champions. Schema:

```jsonc
{
  "best_black_iter": 1,        // current black champion
  "best_white_iter": 0,        // current white champion
  "history": [                 // one append per update_league.py call
    {"iter": 0, "best_black": 0, "best_white": 0},
    {"iter": 1, "best_black": 1, "best_white": 0}
  ],
  "by_iter": {                 // aggregate win rates across the gauntlet
    "0": {"as_black_wr": 0.50, "as_white_wr": 0.50,
          "as_black_games": 50, "as_white_games": 50,
          "opponents": [0],
          "by_opponent": {
            "0": {"as_black": {"iter_n_wins": 25, "games": 50, "iter_n_wr": 0.50},
                  "as_white": {"iter_n_wins": 25, "games": 50, "iter_n_wr": 0.50}}
          }},
    "1": {"as_black_wr": 0.62, "as_white_wr": 0.45,
          "as_black_games": 50, "as_white_games": 50,
          "opponents": [0],
          "by_opponent": {
            "0": {"as_black": {"iter_n_wins": 31, "games": 50, "iter_n_wr": 0.62},
                  "as_white": {"iter_n_wins": 22, "games": 50, "iter_n_wr": 0.44}}
          }}
  }
}
```

## Adding a collect worker (NFS or not)

```bash
# Installs docker + NVIDIA toolkit, auto-detects whether /data/eric/{LearnAlphaGo,
# game_data_root} exists; if not, seeds a local /nfs layout and symlinks
# /data/eric -> /nfs. Appends `[nodes."<ip>"]` to cluster.toml with the
# right `shares_nfs` flag.
uv run ./infra/cluster.py add root@31.22.104.54
./infra/cluster.py ping   # ✓ on new node
```

From then on, the usual `collect_driver.py` dispatch treats it interchangeably
with NFS workers.

## Usage

```bash
EXP=experiments/2026-04-26_22-32-train-fromscratch

# Loop: pre-collect random-vs-random -> train iter0 -> collect -> train, for
# 5 iterations. On first launch run_iteration.sh runs pre_collect_random.py
# (1000 random games -> random-it0/) and trains iter0 from dataset-it0.txt.
bash $EXP/run_iteration.sh 0 5
```

To run just one iteration manually:

```bash
# Seed iter0's training data: 1000 random-vs-random games.
uv run $EXP/pre_collect_random.py

# Train iter0 from random-vs-random data (no resume — fresh init).
uv run $EXP/train.py --dataset-txt $EXP/dataset-it0.txt --iteration 0

# One iteration of SSH-parallel collection (reads cluster.toml).
# Non-NFS workers get the checkpoint pushed over ssh; their NPZs are
# rsynced back before this command returns.
uv run $EXP/collect_driver.py --iteration 0

# Update league standings from the just-collected as-black/as-white NPZs.
uv run $EXP/update_league.py --iteration 0

# Train iter1 from iter0's checkpoint and the just-collected data.
uv run $EXP/train.py --dataset-txt $EXP/dataset-it1.txt --iteration 1 \
    --resume-from /nfs/checkpoints/$(basename $EXP)/iter0_best.pt
```
