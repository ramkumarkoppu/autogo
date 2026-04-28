"""Eval-only games for the fastlearn experiment.

Plays our checkpoint vs a fixed KataGo opponent (default: katago-human-9d) on
9x9 with the same MCTS settings as run_games.py, but with Dirichlet noise
disabled — eval games should be deterministic-ish at the root so accuracy
metrics aren't muddied by added exploration noise. Single-color per process
so the launcher can shard chunks across (iter, color, chunk).
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys
from pathlib import Path
from alpha_go.agents.base import register_agent
from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator

_GAME_FILE_RE = re.compile(r"game(\d+)\.npz$")

# Mirrors run_games.py constants for this experiment, EXCEPT
# ADD_DIRICHLET_NOISE — eval games turn it off.
NUM_SIMULATIONS = 1024
PCR_NUM_SIM = [1024, 2048]
PCR_PROB = [0.95, 0.05]
C_PUCT = 0.5
TEMPERATURE = 0.3
RESIGN_THRESHOLD = 0.05
RESIGN_CONSEC_TURNS = 5
LEAF_BATCH_SIZE = 8
ADD_DIRICHLET_NOISE = False

MODE_OPPONENT = {
    "eval-katago-human": "katago-human-9d",
}

NUM_WORKERS_BY_GPU = {
    "h100": 4,
    "a100": 4,
    "rtx6000_ada": 4,
    "rtxpro6000b": 4,
}
DEFAULT_NUM_WORKERS = 4


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


def _resolve_save_dir(save_name: str) -> Path:
    p = Path(save_name)
    if p.is_absolute():
        return p
    return Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root")) / p


def _existing_unique_game_indices_in_range(
    save_dir: Path, offset: int, num_games: int,
) -> int:
    if not save_dir.exists():
        return 0
    target = set(range(offset, offset + num_games))
    seen: set[int] = set()
    for f in save_dir.glob("*.npz"):
        m = _GAME_FILE_RE.search(f.name)
        if m:
            idx = int(m.group(1))
            if idx in target:
                seen.add(idx)
    return len(seen)


def _game_index_offset_from_remaining(remaining: list[str]) -> int:
    for i, tok in enumerate(remaining):
        if tok == "--game_index_offset" and i + 1 < len(remaining):
            try:
                return int(remaining[i + 1])
            except ValueError:
                return 0
    return 0


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
    m = re.search(r"iter(\d+)", path)
    return f"iter{m.group(1)}" if m else path.split("/")[-1].replace(".pt", "")


def main() -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--mode", choices=list(MODE_OPPONENT.keys()), required=True)
    pre.add_argument("--checkpoint", type=str, required=True)
    pre.add_argument("--num_games", type=int, default=20)
    pre.add_argument("--num_workers", type=int, default=None)
    pre.add_argument("--save-name", type=str, required=True)
    pre.add_argument("--seed", type=int, default=0)
    pre.add_argument("--color", choices=["black", "white"], required=True)
    pre_args, remaining = pre.parse_known_args()

    offset = _game_index_offset_from_remaining(remaining)
    save_dir = _resolve_save_dir(pre_args.save_name)
    already = _existing_unique_game_indices_in_range(
        save_dir, offset, pre_args.num_games,
    )
    if already >= pre_args.num_games:
        print(f"=== [{pre_args.mode}] slot [{offset},{offset + pre_args.num_games}) "
              f"already has {already} games in {save_dir}; skipping ===",
              flush=True)
        return

    num_workers = (pre_args.num_workers if pre_args.num_workers is not None
                   else _detect_num_workers())

    tag = _ckpt_tag(pre_args.checkpoint)
    opponent = MODE_OPPONENT[pre_args.mode]
    agent = _make_agent(f"lt-{tag}", pre_args.checkpoint)
    from alpha_go.self_play import main as self_play_main

    base = [
        "self_play",
        "--board_size", "9",
        "--num_workers", str(num_workers),
        "--save-name", pre_args.save_name,
        "--collect-metrics",
        *remaining,
    ]

    if pre_args.color == "black":
        print(f"=== {pre_args.num_games} games as BLACK vs {opponent} (teacher=B) ===", flush=True)
        sys.argv = base + [
            "--black", agent, "--white", opponent,
            "--num_games", str(pre_args.num_games),
            "--seed", str(pre_args.seed),
            "--black-is-teacher",
        ]
    else:
        print(f"=== {pre_args.num_games} games as WHITE vs {opponent} (teacher=W) ===", flush=True)
        sys.argv = base + [
            "--black", opponent, "--white", agent,
            "--num_games", str(pre_args.num_games),
            "--seed", str(pre_args.seed),
            "--white-is-teacher",
        ]
    self_play_main()


if __name__ == "__main__":
    main()
