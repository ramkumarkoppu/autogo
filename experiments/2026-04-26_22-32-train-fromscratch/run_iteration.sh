#!/usr/bin/env bash
# Run N iterations autonomously: collect-it{ITER} then train-it{ITER+1}.
# Collection is round-robin across `collect`-role nodes; training is one job
# on a `train`-role node. Both dispatched via infra.remote_exec.
# Usage: bash run_iteration.sh <start_iter> <end_iter>
set -euo pipefail

EXP_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$EXP_DIR")"
WORKSPACE_ROOT="$(cd "${EXP_DIR}/../.." && pwd)"
START=${1:?Usage: run_iteration.sh <start_iter> <end_iter>}
END=${2:?Usage: run_iteration.sh <start_iter> <end_iter>}

mkdir -p "${EXP_DIR}/logs"

# On interrupt, sweep leftover docker containers on every collect-role node.
# docker run --rm only cleans up on clean exit; an SSH parent killed mid-run
# can orphan the container and double-book GPUs on the next invocation.
cleanup_remote_containers() {
    echo ""
    echo "############### Cleanup: killing remote containers ###############" >&2
    local image
    image=$(python3 -c "
import tomllib
cfg = tomllib.load(open('${WORKSPACE_ROOT}/cluster.toml', 'rb'))
print(cfg.get('image', 'ghcr.io/ericjang/alphago-worker:latest'))
")
    # Emit one 'user ip port' triple per cluster node. No port prints an
    # empty field so the read below sees three tokens consistently.
    while read -r user ip port; do
        local ssh_opts="-o StrictHostKeyChecking=no -o ConnectTimeout=5 -i $HOME/.ssh/id_ed25519"
        [ -n "$port" ] && ssh_opts="$ssh_opts -p $port"
        local dk="docker"
        [ "$user" != "root" ] && dk="sudo docker"
        # Background each cleanup so slow hosts don't block the rest.
        (
            ids=$(ssh $ssh_opts "${user}@${ip}" "$dk ps -q --filter ancestor=${image}" 2>/dev/null || true)
            if [ -n "$ids" ]; then
                echo "[${user}@${ip}] killing: $ids" >&2
                ssh $ssh_opts "${user}@${ip}" "$dk rm -f $ids" >/dev/null 2>&1 || true
            fi
        ) &
    done < <(python3 -c "
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
        --share-cluster \
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
    RANDOM_DATA_DIR="/nfs/game_data_root/experiments/${EXP_NAME}/random-it0"
    if [ ! -d "$RANDOM_DATA_DIR" ] || [ -z "$(ls -A "$RANDOM_DATA_DIR" 2>/dev/null)" ]; then
        echo "############### Bootstrap: pre-collect random-vs-random games ###############"
        uv run "${EXP_DIR}/pre_collect_random.py"
    else
        echo "$RANDOM_DATA_DIR already populated, skipping pre-collect"
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
        > "${EXP_DIR}/logs/it${ITER}/collect.log" 2>&1

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
            # Carry forward league cross-play paths from the previous
            # iteration's dataset (stripping comments/blanks). Skip for
            # ITER=0 so iter1+ never re-includes random-it0 — the random-
            # vs-random bootstrap data is only meant for the iter0 train.
            if [ "$ITER" -gt 0 ]; then
                grep -vE "^(#|$)" "$PREV_DS"
            fi
            # Append the gauntlet + selfplay sets collected in this iteration.
            echo "experiments/${EXP_NAME}/as-black-it${ITER}"
            echo "experiments/${EXP_NAME}/as-white-it${ITER}"
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
