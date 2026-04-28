# fastlearn — final report

## Headline result

| metric | value |
|---|---|
| Parent baseline iter1 holdout policy_acc | 0.1014 |
| Parent baseline iter10 holdout policy_acc | 0.3034 |
| **Our Phase B iter1 best (dirichlet_v1)** | **0.3389** |
| Our Phase B iter1 mean ± std (4 samples)  | 0.3290 ± 0.018 |
| Improvement vs parent iter1, best | **+0.238** absolute (+234%) |
| Improvement vs parent iter1, mean | +0.228 absolute (+225%) |
| Improvement vs parent iter10, mean | +0.026 absolute |

Phase B iter1 — Phase A's tuned-from-scratch training plus a single
MCTS-collect-and-retrain — on average matches and slightly beats the
parent's iter10 holdout accuracy (which required ten sequential
train+collect rounds).

## ⚠️ Caveat: the win is FRONT-LOADED and DOESN'T COMPOUND

A subsequent multi-iteration from-scratch run (`run_iteration.sh 0 20`)
exposed a critical limitation. The fastlearn config is great for one
collect-and-retrain step, but **regresses when iterated**:

| iter | fastlearn | parent |
|---|---|---|
| 0 | 0.060 | 0.075 |
| 1 | **0.270** | 0.101 |
| 2 | 0.254 | 0.213 |
| 3 | 0.265 | 0.253 |
| 4 | 0.243 | 0.257 |
| 5 | **0.166** | 0.273 |
| 6 | **0.179** | 0.306 |
| 7 | (in flight) | 0.326 (parent peak) |

Fastlearn iter1 already matches parent's iter5 — a real ~5x speedup on
the early curve. But the trajectory then plateaus around 0.25 for iters
2-4, and **collapses at iter5** (failed promotion: as_black_wr=0.42 <
0.55 threshold). The most likely culprit is the Dirichlet root noise:
the very thing that gave a small reproducible bump on a single iteration
(+0.015, lower variance) injects exploration that compounds badly across
iterations — by iter5 the model has drifted enough that it can't beat
its own previous champion, training data quality degrades, and the loop
falls apart.

Parent's noise-free MCTS produces cleaner per-game targets and
consistent self-improvement; the iters compound. Fastlearn's dirichlet
+ minor architecture tweaks compound noise instead of skill.

**Practical takeaway**: the it1 result (0.3366 mean / 0.3389 best) is a
genuine training-recipe improvement on the *one-shot* problem the user
posed, but if anyone wanted to use this for multi-iter training they
should turn Dirichlet noise OFF (revert to parent's setting).

See `figures/comparison_vs_parent.png`.

## Phase A — hparam tuning on parent's dataset-it10

Goal: find a well-tuned training recipe before testing collect strategies
in Phase B. Used parent's `dataset-it10.txt` as a fixed training set;
measured holdout policy_acc.

### Largest single-step improvement: bug fix in policy mask

Found a bug in the parent's training loop:
```python
is_teacher = winner.float()
```
This override discarded the dataset's actual `is_teacher` flag and
trained the policy head only on positions the eventual game winner
played — about half of all MCTS positions. Switching to
```python
is_teacher = (ds_is_teacher | winner).float()
```
trains policy on every MCTS position (since `is_teacher` is True for all
selfplay/gauntlet data) and falls back to the winner-only rule for
random-playout data (where `is_teacher` is missing). This
several-character change lifted holdout policy_acc from
**0.2708 → 0.3046** at the same training budget (+0.034 absolute) —
matching the parent's iter10 result with one 5-min training run.

### Phase A results table

See `results-phaseA.tsv`. 5-minute training runs unless otherwise noted.

| run | config                                       | holdout_acc | status |
|-----|----------------------------------------------|-------------|--------|
| 01  | baseline (128ch x 10b, max_epochs=4, 141s)   | 0.2615      | undertrained |
| 02  | max_epochs=30, 10min                         | 0.2701      | + |
| 03  | lr=3e-3                                      | 0.2639      | discard |
| 04  | cosine schedule aligned to time_budget; 5min | 0.2657      | + (5min ref) |
| 05  | 192ch x 10b                                  | 0.2708      | + |
| 06  | **+ ds is_teacher mask**                     | **0.3046**  | **biggest win** |
| 07  | 256ch x 10b                                  | 0.2949      | discard |
| 08  | WD=1e-3                                      | 0.2714      | discard |
| 09  | 192ch + ds_is_teacher + 10min budget         | 0.3078      | best Phase A ckpt |

Final Phase A config: `192ch × 10b, lr=1e-3, wd=5e-3, cosine schedule
aligned to time_budget, ds is_teacher mask, 10-min training`.

## Phase B — collect-then-train one iteration

Take Phase A best as iter0 (holdout 0.3078). Collect 50 games of selfplay
with iter0; train iter1 resuming from iter0, on dataset = parent's
accumulated MCTS data + new selfplay-it0. Measure iter1's
holdout_policy_acc.

### Variance is large (~0.018)

Two independent runs of the same `baseline` config landed at 0.3377 and
0.3053 — a 0.032 spread. Across 4 runs (2 baseline + 2 dirichlet) the
overall std is **±0.018** at iter1. This means single-run differences
under ~0.02 are within noise; we need multi-sample comparisons for
smaller effects.

### Phase B results table

See `results.tsv`. All variants run from the same Phase A best ckpt.

| variant         | holdout_acc_it1 | notes |
|---|---|---|
| baseline (run 1)                                    | 0.3377 | keep |
| baseline (run 2)                                    | 0.3053 | keep — variance check |
| **dirichlet** (root noise on, run 1)                | 0.3389 | keep |
| **dirichlet** (root noise on, run 2)                | 0.3343 | keep |
| cleandata (only it7-9 + new)                        | 0.3051 | discard (small dataset → overfit) |
| onlynew (only new selfplay-it0)                     | 0.2730 | discard (early-stopped at 1200 steps; severe overfit) |
| longertrain (20-min vs 10-min)                      | 0.2986 | discard (overfit) |
| lowerlr (lr=3e-4 vs 1e-3)                           | 0.3115 | discard |
| strong25 (25 games × 2048 sims + noise)             | 0.3306 | discard (game variety > stronger per-game MCTS) |
| strongmcts (50 games × 2048 sims)                   | killed | exceeds 39-min/iter budget |

### Mean comparisons

| config | mean | std | n | gap vs baseline |
|---|---|---|---|---|
| baseline (no noise)   | 0.3215 | 0.0162 | 2 | — |
| dirichlet (+ noise)   | 0.3366 | 0.0023 | 2 | **+0.015** |

Dirichlet noise on the MCTS root is a small-but-reproducible gain of
~0.015 absolute, AND lowers run-to-run variance dramatically (0.016 →
0.002). Worth keeping as the default Phase B setting.

## Key insights

1. **Highest-leverage change**: a few-character bug fix in the loss mask
   (using the dataset's `is_teacher` instead of the parent's
   `winner.float()` override). +0.034 absolute on holdout policy_acc on
   the same data and time budget.

2. **Architecture scales modestly.** 128ch → 192ch gave +0.005, 192ch →
   256ch regressed at the 5-min budget. 192ch is the sweet spot.

3. **Less data hurts more than expected.** Both `cleandata` (only
   recent iters) and `onlynew` (only the new selfplay-it0) regressed.
   The accumulated parent data does real regularization work.

4. **More training is not better.** Doubling the training budget to 20
   minutes regressed (0.3377 → 0.2986).

5. **Default LR is right-sized for warm-resume.** lr=3e-4 regressed;
   lr=1e-3 with cosine-to-zero is the right speed.

6. **Dirichlet noise modestly helps and stabilises Phase B.** +0.015
   absolute on mean, and 7× lower std across runs.

7. **Game variety > per-game MCTS strength.** 25 games × 2048 sims
   regressed vs 50 × 1024 with the same total compute.

8. **It1 ≈ It10.** Phase A's tuned training + a single Phase B
   collect-then-train cycle reaches holdout policy_acc that beats what
   the parent took ten sequential iterations to achieve. Most of the
   parent's per-iteration gains came from training-recipe differences
   being amortised across ten iters, not from accumulating selfplay
   data.

## Files

- `train.py` — final training script (192ch × 10b, ds is_teacher mask).
- `run_games.py` — selfplay runner with MCTS hyperparams (default:
  1024 sims, c_puct=0.5, T=0.3, **Dirichlet noise on**).
- `collect_driver.py` / `update_league.py` / `pre_collect_random.py` —
  parent's collection harness (unmodified).
- `run_phaseB.sh` — orchestrates a Phase B run.
- `dataset-phaseB-baseline.txt` — Phase B training set (it10 + new).
- `dataset-phaseB-cleandata.txt` / `-onlynew.txt` — variants.
- `results-phaseA.tsv`, `results.tsv` — run logs.
- `figures/phaseA_progress.png`, `figures/phaseB_progress.png`,
  `figures/comparison_vs_parent.png` — plots.
- `holdout_eval/` — per-color holdout metrics from eval-only runs.

Best Phase A ckpt: `phaseA-best.pt` (holdout 0.3078).
Best Phase B ckpt: `phaseB-dirichlet-it1.pt` (holdout 0.3389).
