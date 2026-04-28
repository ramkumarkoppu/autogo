# mcts-throughput-bench

Measure self-play MCTS simulations/sec on a 19x19 board with progressively
more optimization. Each mode plays the same 19x19 model against itself with
1024 MCTS sims/move and resignation disabled. The headline metric is
sims/sec — moves/sec confounds throughput with sims/move.

Modes
-----
1. **py-mcts** — single-thread Python MCTS, single-board NN inference per leaf.
2. **cpp-mcts-seq** — single-thread C++ MCTS, single-board Python NN inference
   per leaf (one game).
3. **cpp-batched** — `N` games in parallel threads, each running C++ MCTS
   sequentially, all sharing one `LocalBatchedInferenceEngine` (cross-game
   batching).
4. **cpp-batched-leaf** — `N` games in parallel threads, each running C++
   MCTS with leaf-parallel virtual-loss batching, all sharing one
   `LocalBatchedInferenceEngine` (within-game leaf batches mixed with
   cross-game batches at the engine).

Setup
-----
- Board: 19x19, komi 7.5
- Model: `/nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt`
  (`SizeInvariantGoResNet`, auto-detected by checkpoint fingerprint)
- num_simulations = 1024, c_puct = 0.5, temperature = 0.3, no Dirichlet noise
- resign_threshold = 0 (disabled)
- max_moves per game = 40 (caps each game so wall time is bounded)
- N = 8 parallel games for modes 3 + 4

Run
---
```bash
bash experiments/2026-04-25_14-16-mcts-throughput-bench/launcher.sh
uv run python experiments/2026-04-25_14-16-mcts-throughput-bench/analyze.py
```

The launcher dispatches one job per mode to four collect-role nodes.
Each job writes a JSON to `results/<mode>.json`; `analyze.py` reads them and
plots `figures/simulations_per_sec.png`.

Files
-----
- `benchmark.py` — runs one mode and prints/writes the throughput JSON.
- `driver.py` — fans out the four jobs across the cluster.
- `launcher.sh` — one-shot entrypoint for the driver.
- `analyze.py` — produces the bar chart + report.
