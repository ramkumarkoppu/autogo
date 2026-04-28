#!/usr/bin/env bash
# Inner script: runs all four benchmark modes sequentially on the same GPU.
# Invoked inside the docker container by infra.remote_exec.
set -euo pipefail
EXP="2026-04-25_14-16-mcts-throughput-bench"
CKPT="/nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt"
OUT_DIR="/nfs/game_data_root/experiments/${EXP}/results"
mkdir -p "${OUT_DIR}"

run_mode() {
    local mode="$1"; local num_games="$2"
    echo "=== ${mode} ==="
    uv run experiments/${EXP}/benchmark.py \
        --mode "${mode}" \
        --checkpoint "${CKPT}" \
        --num-games "${num_games}" \
        --max-moves 40 \
        --num-simulations 1024 \
        --leaf-batch-size 8 \
        --out "${OUT_DIR}/${mode}.json"
}

run_mode py-mcts 1
run_mode cpp-mcts-seq 1
run_mode cpp-batched 8
run_mode cpp-batched-leaf 8
echo "=== ALL MODES DONE ==="
