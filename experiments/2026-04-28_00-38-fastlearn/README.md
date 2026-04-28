# fastlearn — speed up holdout policy_acc rise vs champion baseline

Forked from `experiments/2026-04-27_16-31-train-fromscratch-champion`. Goal:
make holdout policy_acc rise faster than the parent's baseline.

Parent baseline (held-out policy accuracy on selfplay-it40 from
2026-04-15_08-00-learngo-decen-collect-v7):
- iter0: 0.0754
- iter1: 0.1014
- iter2: 0.2132
- iter10: 0.3034 (Phase A target to match-or-beat with hparam tuning)

## Two-phase workflow

**Phase A — hparam tuning (training-only, dataset-it10).**
Reuse parent's `dataset-it10.txt` (10 iterations of accumulated selfplay +
gauntlet). Train from scratch, measure holdout policy_acc. Goal: find
well-tuned LR/schedule/architecture. Logged to `results-phaseA.tsv`.

**Phase B — collect-then-train one iteration.**
The best Phase A checkpoint is treated as iter 0. Each Phase B run:
1. Collect with it0 (selfplay and/or other matchups — strategy is part of
   what we tune).
2. Train it1 from the collected data (initialised from it0).
3. Report holdout policy_acc of it1.

There is no it2 in this version. Single metric per run:
holdout_policy_acc (it1). Logged to `results.tsv`.

Each iter step (train or collect) is time-boxed to ~3 min × num_gpus
(cluster has 13 GPUs total → ~39 min per step).

## Layout

- `train.py` — single-iteration trainer, identical interface to parent.
- `collect_driver.py` / `run_games.py` / `update_league.py` /
  `pre_collect_random.py` — collection harness, identical to parent.
- `run_pipeline.sh` — orchestrates one Phase B (collect + train) run.
- `results-phaseA.tsv` — Phase A hparam log.
- `results.tsv` — Phase B pipeline log (commit / holdout_policy_acc_it1 /
  status / description).
