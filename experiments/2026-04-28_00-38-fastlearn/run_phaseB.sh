#!/usr/bin/env bash
# Phase B: collect selfplay games with the it0 checkpoint, then train it1.
#
# Args:
#   $1 = path to Phase-A-best ckpt (will be staged as iter0_best.pt)
#   $2 = run tag (used in log dirs and dataset filename)
#
# Knobs (env-overridable):
#   PHASEB_NUM_GAMES   = 50  (selfplay games to collect)
#   PHASEB_NUM_JOBS    = 12  (collect shards across cluster)
#   PHASEB_DATASET     = path to dataset txt for it1 training. Default
#                          dataset-phaseB-baseline.txt = dataset-it10 +
#                          new selfplay-it0.
#   PHASEB_TIME_BUDGET = 600 (seconds for the it1 training run)
#
# Reports a single ===PHASEB_RESULT=== JSON line at the end. Logs land in
# logs/phaseB/<tag>/.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"

CKPT_SRC="${1:?Usage: run_phaseB.sh <phaseA-ckpt> <tag>}"
TAG="${2:?Usage: run_phaseB.sh <phaseA-ckpt> <tag>}"

PHASEB_NUM_GAMES="${PHASEB_NUM_GAMES:-50}"
PHASEB_NUM_JOBS="${PHASEB_NUM_JOBS:-12}"
PHASEB_DATASET="${PHASEB_DATASET:-${EXP_DIR}/dataset-phaseB-baseline.txt}"
PHASEB_TIME_BUDGET="${PHASEB_TIME_BUDGET:-600}"

LOG_DIR="${EXP_DIR}/logs/phaseB/${TAG}"
mkdir -p "${LOG_DIR}"
CKPT_DIR="/nfs/checkpoints/${EXP_NAME}"
mkdir -p "${CKPT_DIR}"

# Stage the Phase A ckpt as iter0_best.pt for the collect_driver.
cp -f "${CKPT_SRC}" "${CKPT_DIR}/iter0_best.pt"

# Bootstrap league_state.json: iter0 is champion of both colors so that on
# iter 0 collect_driver only runs selfplay (it skips gauntlet matchups for
# iter==0 to avoid recursion against the same model).
cat > "${EXP_DIR}/league_state.json" <<EOF
{
  "best_black_iter": 0,
  "best_white_iter": 0,
  "history": [
    {"iter": 0, "best_black": 0, "best_white": 0,
     "as_black_wr": null, "as_white_wr": null, "promoted": true}
  ],
  "by_iter": {}
}
EOF

echo "############### Phase B [${TAG}]: collect iter0 selfplay (${PHASEB_NUM_GAMES} games, ${PHASEB_NUM_JOBS} jobs) ###############"
uv run "${EXP_DIR}/collect_driver.py" \
    --iteration 0 \
    --num-games 0 \
    --selfplay-games "${PHASEB_NUM_GAMES}" \
    --num-jobs "${PHASEB_NUM_JOBS}" \
    2>&1 | tee "${LOG_DIR}/collect.log"

echo "############### Phase B [${TAG}]: train iter1 from ${PHASEB_DATASET} ###############"
uv run python -m infra.remote_exec \
    --role train \
    --exp-name "${EXP_NAME}" \
    --log-path "${LOG_DIR}/train-it1.log" \
    -- uv run "${EXP_DIR}/train.py" \
        --dataset-txt "${PHASEB_DATASET}" \
        --iteration 1 \
        --resume-from "${CKPT_DIR}/iter0_best.pt" \
        --time-budget "${PHASEB_TIME_BUDGET}"

# Extract holdout_policy_acc into the standard ===PHASEB_RESULT=== line.
RESULT_LINE=$(grep -A 1 "===RESULT===" "${LOG_DIR}/train-it1.log" | tail -1)
echo ""
echo "===PHASEB_RESULT==="
echo "tag=${TAG}"
echo "${RESULT_LINE}"
