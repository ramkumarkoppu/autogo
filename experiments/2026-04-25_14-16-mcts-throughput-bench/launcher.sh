#!/usr/bin/env bash
# Run all 4 benchmark modes sequentially on a single remote collect-role node.
# Same GPU for every mode → cleaner moves/sec comparison; only one lease needed.
set -euo pipefail
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_NAME="$(basename "${EXP_DIR}")"
cd /workspace
exec uv run python -m infra.remote_exec \
    --role collect \
    --share-cluster \
    --log-path "${EXP_DIR}/logs/run.log" \
    --exp-name "${EXP_NAME}" \
    -- bash "experiments/${EXP_NAME}/run_all_modes.sh"
