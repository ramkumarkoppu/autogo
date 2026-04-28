# 2026-04-25_14-16-mcts-throughput-bench

## Goal
Quantify MCTS simulations/sec at each level of optimization on a 19x19 board.

## Setup
- Self-play (agent vs itself), 1024 sims/move, c_puct=0.5, temperature=0.3.
- Resignation disabled. Komi 7.5. max_moves=40 per game.
- Model: SizeInvariantGoResNet 18M, /nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt
- Modes 3-4: 8 parallel game threads sharing one `LocalBatchedInferenceEngine` (batch_size=64, timeout=2ms).
- Mode 4 leaf_batch_size=8 (virtual-loss leaf parallel inside C++ MCTS).

## Results

| # | mode | num_games | total_moves | elapsed_s | moves/sec | sims/sec |
|---|------|----------:|------------:|----------:|----------:|---------:|
| py | py-mcts | 1 | 40 | 849.5 | 0.05 | **48** |
| cpp | cpp-mcts-seq | 1 | 40 | 329.1 | 0.12 | **124** |
| cpp | cpp-batched | 8 | 320 | 1079.7 | 0.30 | **303** |
| cpp | cpp-batched-leaf | 8 | 320 | 351.0 | 0.91 | **933** |

End-to-end speedup (mode 4 / mode 1, sims/sec): **19.4x**

Figures:
- `figures/simulations_per_sec.png`

## Key findings
- TBD (fill in after run completes)
