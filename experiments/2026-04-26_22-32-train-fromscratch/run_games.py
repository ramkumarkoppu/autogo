"""Cross-play games for learngo-decen-collect-v7 (local, no Ray).

Plays `--num_games` games of (black-checkpoint vs white-checkpoint) with both
sides marked as teachers (their MCTS visit counts become training targets).
Pass the same path for both checkpoints to get selfplay; pass distinct paths
for league cross-play (e.g. iter N vs current best_white).
"""
from __future__ import annotations
import argparse, subprocess, sys
from alpha_go.agents.base import register_agent
from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator

NUM_SIMULATIONS = 1024
PCR_NUM_SIM = [1024, 2048]
PCR_PROB = [0.95, 0.05]
C_PUCT = 5.0
TEMPERATURE = 0.3
RESIGN_THRESHOLD = 0.05
RESIGN_CONSEC_TURNS = 5
LEAF_BATCH_SIZE = 8
ADD_DIRICHLET_NOISE = False

# Per-GPU worker counts. Matched by alphanumeric substring against nvidia-smi's
# GPU name (so "rtx6000_ada" matches "NVIDIA RTX 6000 Ada Generation").
NUM_WORKERS_BY_GPU = {
    "h100": 4,
    "a100": 4,
    "rtx6000_ada": 4,
    "rtxpro6000b": 4,
}
DEFAULT_NUM_WORKERS = 4  # fallback when GPU name doesn't match any key above


def _detect_num_workers() -> int:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, check=False, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print(f"WARN: nvidia-smi unavailable; defaulting to {DEFAULT_NUM_WORKERS} workers",
              flush=True)
        return DEFAULT_NUM_WORKERS
    name = (r.stdout.splitlines() or [""])[0].strip()
    normalized = "".join(c for c in name.lower() if c.isalnum())
    for key, n in NUM_WORKERS_BY_GPU.items():
        if "".join(c for c in key if c.isalnum()) in normalized:
            print(f"Auto num_workers: GPU='{name}' -> {n}", flush=True)
            return n
    print(f"WARN: GPU '{name}' not in NUM_WORKERS_BY_GPU; defaulting to {DEFAULT_NUM_WORKERS}",
          flush=True)
    return DEFAULT_NUM_WORKERS


def _make_agent(name: str, checkpoint: str) -> str:
    @register_agent(name)
    class _Agent(CppMCTSAgent):
        def __init__(self) -> None:
            evaluator = LeafBatchedNNEvaluator(checkpoint, 9, "18M")
            super().__init__(
                evaluator=evaluator,
                num_simulations=NUM_SIMULATIONS,
                c_puct=C_PUCT,
                temperature=TEMPERATURE,
                add_noise=ADD_DIRICHLET_NOISE,
                resign_threshold=RESIGN_THRESHOLD,
                resign_consec_turns=RESIGN_CONSEC_TURNS,
                leaf_batch_size=LEAF_BATCH_SIZE,
                pcr_sims=PCR_NUM_SIM,
                pcr_probs=PCR_PROB,
            )
    from alpha_go import self_play
    self_play._AGENT_MODEL_CONFIGS[name] = "18M"
    return name


def _ckpt_tag(path: str) -> str:
    import re
    m = re.search(r"iter(\d+)", path)
    return f"iter{m.group(1)}" if m else path.split("/")[-1].replace(".pt", "")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--black-checkpoint", type=str, required=True)
    pre.add_argument("--white-checkpoint", type=str, required=True)
    pre.add_argument("--num_games", type=int, default=80)
    pre.add_argument("--num_workers", type=int, default=None,
                     help="Override; if unset, auto-detected from GPU type.")
    pre.add_argument("--save-name", type=str, required=True)
    pre.add_argument("--seed", type=int, default=0)
    pre_args, remaining = pre.parse_known_args()

    num_workers = pre_args.num_workers if pre_args.num_workers is not None else _detect_num_workers()

    btag = _ckpt_tag(pre_args.black_checkpoint)
    wtag = _ckpt_tag(pre_args.white_checkpoint)
    # Distinct ltb-/ltw- prefixes keep the agent registry uniquely keyed even
    # when both sides point at the same checkpoint (selfplay).
    black = _make_agent(f"ltb-{btag}", pre_args.black_checkpoint)
    white = _make_agent(f"ltw-{wtag}", pre_args.white_checkpoint)

    print(f"=== Cross-play {pre_args.num_games} games (B={btag} vs W={wtag}, both teachers) ===",
          flush=True)
    from alpha_go.self_play import main as self_play_main
    sys.argv = [
        "self_play",
        "--board_size", "9",
        "--num_workers", str(num_workers),
        "--save-name", pre_args.save_name,
        "--collect-metrics",
        "--black", black, "--white", white,
        "--num_games", str(pre_args.num_games),
        "--seed", str(pre_args.seed),
        "--black-is-teacher", "--white-is-teacher",
        *remaining,
    ]
    self_play_main()


if __name__ == "__main__":
    main()
