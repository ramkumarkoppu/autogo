#!/usr/bin/env bash
# Run N iterations autonomously (fastlearn version):
#   bash run_iteration.sh <start_iter> <end_iter>
#
# Iter N == train iter N's checkpoint and play it through one collect cycle:
#     [bootstrap] pre_collect_random.py + train iter0   (if iter0_best.pt missing)
#     for ITER in [start..end]:
#         collect_driver.py --iteration ITER             # plays games using iter ITER
#         update_league.py  --iteration ITER             # promotes iter ITER vs reigning champs
#         train.py          --iteration (ITER+1) --resume-from iter${ITER}_best.pt
#
# Adapted from experiments/2026-04-27_16-31-train-fromscratch-champion/run_iteration.sh
# with the collect step UNCOMMENTED (parent had it commented out).
#
# fastlearn defaults: 192ch x 10b, ds is_teacher mask, Dirichlet noise on,
# 1024 sims, 50 games / matchup, 10-min training. See report.md for the
# full hyperparameter rationale.
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
WORKSPACE_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
START=${1:?Usage: run_iteration.sh <start_iter> <end_iter>}
END=${2:?Usage: run_iteration.sh <start_iter> <end_iter>}

mkdir -p "${EXP_DIR}/logs"

# On interrupt, sweep leftover docker containers on every cluster node.
# docker run --rm only cleans up on clean exit; an SSH parent killed mid-run
# can orphan the container and double-book GPUs on the next invocation.
cleanup_remote_containers() {
    echo ""
    echo "############### Cleanup: killing remote containers ###############" >&2
    local image
    image=$(uv run python -c "
import tomllib
cfg = tomllib.load(open('${WORKSPACE_ROOT}/cluster.toml', 'rb'))
print(cfg.get('image', 'ghcr.io/ericjang/alphago-worker:latest'))
")
    while read -r user ip port; do
        local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -i $HOME/.ssh/id_ed25519"
        [ -n "$port" ] && ssh_opts="$ssh_opts -p $port"
        local dk="docker"
        [ "$user" != "root" ] && dk="sudo docker"
        (
            ids=$(ssh $ssh_opts "${user}@${ip}" "$dk ps -q --filter ancestor=${image}" 2>/dev/null || true)
            if [ -n "$ids" ]; then
                echo "[${user}@${ip}] killing: $ids" >&2
                ssh $ssh_opts "${user}@${ip}" "$dk rm -f $ids" >/dev/null 2>&1 || true
            fi
        ) &
    done < <(uv run python -c "
import tomllib
cfg = tomllib.load(open('${WORKSPACE_ROOT}/cluster.toml', 'rb'))
for ip, e in cfg.get('nodes', {}).items():
    print(e.get('user', 'root'), ip, e.get('ssh_port') or '')
")
    wait
    echo "############### Cleanup done ###############" >&2
}
trap 'cleanup_remote_containers; exit 130' INT TERM

train_remote() {
    local iteration=$1 resume=$2 ds=$3
    mkdir -p "${EXP_DIR}/logs/it${iteration}"
    uv run python -m infra.remote_exec \
        --role train \
        --exp-name "${EXP_NAME}" \
        --log-path "${EXP_DIR}/logs/it${iteration}/train.log" \
        -- uv run "${EXP_DIR}/train.py" \
            --dataset-txt "${ds}" \
            --iteration "${iteration}" \
            --resume-from "${resume}"
}

START_CKPT="/nfs/checkpoints/${EXP_NAME}/iter${START}_best.pt"
if [ ! -f "$START_CKPT" ]; then
    if [ "$START" -ne 0 ]; then
        echo "ERROR: starting iter ${START} > 0 but ${START_CKPT} missing" >&2
        exit 1
    fi
    # Reuse parent's random-it0 (saves 5K random-vs-random games' worth of work).
    PARENT_RANDOM_DIR="/nfs/game_data_root/experiments/2026-04-27_16-31-train-fromscratch-champion/random-it0"
    if [ ! -d "$PARENT_RANDOM_DIR" ] || [ -z "$(ls -A "$PARENT_RANDOM_DIR" 2>/dev/null)" ]; then
        echo "############### Bootstrap: pre-collect random-vs-random games ###############"
        uv run "${EXP_DIR}/pre_collect_random.py"
    else
        echo "Using parent's random-it0 ($(ls "$PARENT_RANDOM_DIR" | wc -l) files), skipping pre-collect"
    fi
    echo "############### Bootstrap: train iter0 from random data ###############"
    train_remote 0 "" "${EXP_DIR}/dataset-it0.txt"
else
    echo "$START_CKPT already exists, proceeding with collect"
fi

for ITER in $(seq "$START" "$END"); do
    echo ""
    echo "############### Iter ${ITER}: collection ###############"
    mkdir -p "${EXP_DIR}/logs/it${ITER}"
    uv run "${EXP_DIR}/collect_driver.py" --iteration "${ITER}" \
        2>&1 | tee -a "${EXP_DIR}/logs/it${ITER}/collect.log"

    echo "############### Iter ${ITER}: updating league standings ###############"
    uv run "${EXP_DIR}/update_league.py" --iteration "${ITER}" \
        | tee -a "${EXP_DIR}/logs/it${ITER}/league.log"

    NEXT=$((ITER + 1))
    DS="${EXP_DIR}/dataset-it${NEXT}.txt"
    PREV_DS="${EXP_DIR}/dataset-it${ITER}.txt"
    if [ ! -f "$DS" ]; then
        if [ "$ITER" -gt 0 ] && [ ! -f "$PREV_DS" ]; then
            echo "ERROR: prior dataset ${PREV_DS} missing; can't build ${DS}" >&2
            exit 1
        fi
        {
            echo "# ${EXP_NAME} iter${NEXT} dataset (auto-generated)."
            # Carry forward champion cross-play paths from the previous
            # iteration's dataset (stripping comments/blanks). Skip for
            # ITER=0 so iter1+ doesn't re-include random-it0.
            if [ "$ITER" -gt 0 ]; then
                grep -vE "^(#|$)" "$PREV_DS"
            fi
            # iter0 only produced selfplay; iter1+ also produced as-black/
            # as-white champion gauntlets.
            if [ "$ITER" -gt 0 ]; then
                echo "experiments/${EXP_NAME}/as-black-it${ITER}"
                echo "experiments/${EXP_NAME}/as-white-it${ITER}"
            fi
            echo "experiments/${EXP_NAME}/selfplay-it${ITER}"
        } > "$DS"
    fi

    echo ""
    echo "############### Iter ${NEXT}: training ###############"
    RESUME="/nfs/checkpoints/${EXP_NAME}/iter${ITER}_best.pt"
    train_remote "${NEXT}" "${RESUME}" "${DS}"
done
echo ""
echo "############### Done. Last trained iter: ${END} ###############"
