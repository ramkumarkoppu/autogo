"""Single-mode MCTS throughput benchmark on 19x19.

Plays self-play games (agent vs itself, same checkpoint) for a fixed
max_moves cap and reports moves/sec.

Modes
-----
- py-mcts          : Python MCTS with single-board NN inference.
- cpp-mcts-seq     : C++ MCTS with single-board NN inference (LeafBatchedNN).
- cpp-batched      : N parallel games, C++ MCTS sequential within a game,
                     shared LocalBatchedInferenceEngine across games.
- cpp-batched-leaf : N parallel games, C++ MCTS leaf-parallel within a game
                     (virtual loss + leaf_batch_size leaves per step), shared
                     LocalBatchedInferenceEngine across games.

Output: a JSON written to --out describing total moves and elapsed seconds.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

import alpha_go_cpp
from alpha_go.agents.base import Agent, PASS
from alpha_go.agents.base import get_pass_index
from alpha_go.agents.nn_mcts import (
    BatchedLocalNNEvaluator,
    CppMCTSAgent,
    LeafBatchedNNEvaluator,
    NNEvaluator,
)
from alpha_go.gameplay import play_game
from alpha_go.go import GoState
from alpha_go.inference import LocalBatchedInferenceEngine
from alpha_go.mcts import MCTSConfig, run_mcts, select_action_from_mcts
from alpha_go.model import (
    SizeInvariantGoResNet,
    create_mup_model,
    upgrade_state_dict_for_pass,
)


def _build_model(checkpoint_path: str, board_size: int, model_config: str,
                 device: torch.device) -> torch.nn.Module:
    """Load a checkpoint and instantiate the matching model class.

    Inlined here because the docker image's `nn_mcts.py` predates the
    public helper. Auto-detects SizeInvariantGoResNet vs MuPGoResNet by
    state-dict fingerprint, mirroring `LeafBatchedNNEvaluator`'s rule.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"]
    config_name = str(checkpoint.get("config", ""))
    if config_name.startswith("SizeInvariantGoResNet") or "pass_fc.weight" in state_dict:
        channels = int(state_dict["input_conv.weight"].shape[0])
        n_blocks = sum(1 for k in state_dict
                       if k.startswith("blocks.") and k.endswith(".conv1.weight"))
        value_hidden = int(state_dict["value_fc2.weight"].shape[1])
        norm_type = "bn" if "input_bn.running_mean" in state_dict else "gn"
        use_se = "blocks.0.se.fc1.weight" in state_dict
        se_reduction = 8
        if use_se:
            se_hidden = int(state_dict["blocks.0.se.fc1.weight"].shape[0])
            se_reduction = max(1, channels // max(1, se_hidden))
        model: torch.nn.Module = SizeInvariantGoResNet(
            channels=channels, n_blocks=n_blocks, value_hidden=value_hidden,
            norm_type=norm_type, use_se=use_se, se_reduction=se_reduction,
        ).to(device)
        model.load_state_dict(state_dict)
    else:
        model = create_mup_model(
            config=model_config, board_size=board_size, device=device,
        )
        state_dict = upgrade_state_dict_for_pass(state_dict, board_size)
        model.load_state_dict(state_dict)
    model.eval()
    return model


def _make_engine(checkpoint_path: str, board_size: int, model_config: str,
                 batch_size: int, batch_timeout_ms: float) -> LocalBatchedInferenceEngine:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(checkpoint_path, board_size, model_config, device)
    engine = LocalBatchedInferenceEngine(
        model=model, device=device, batch_size=batch_size,
        batch_timeout_ms=batch_timeout_ms, board_size=board_size,
    )
    engine.start()
    return engine

BOARD_SIZE = 19
MODEL_CONFIG = "18M"  # ignored when checkpoint is SizeInvariantGoResNet
DEFAULT_CKPT = "/nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt"
NUM_SIMULATIONS = 1024
C_PUCT = 0.5
TEMPERATURE = 0.3


# ----- Mode 1: Python MCTS agent ------------------------------------------------


class PyMCTSAgent(Agent):
    """Single-thread Python MCTS, single-board NN inference per leaf.

    Constructed directly here (rather than via NNMCTSAgent) so it can use
    SizeInvariantGoResNet — the registered NNMCTSAgent hardcodes MuPGoResNet.
    """

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 board_size: int, num_simulations: int,
                 c_puct: float, temperature: float) -> None:
        self.evaluator = NNEvaluator(model, device, board_size)
        self.config = MCTSConfig(
            c_puct=c_puct, dirichlet_alpha=0.0, dirichlet_weight=0.0, lambda_=0.0,
        )
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.board_size = board_size

    def select_move(self, board, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        state = GoState.from_cpp_board(board)
        root = run_mcts(
            root_state=state,
            num_simulations=self.num_simulations,
            config=self.config,
            get_policy_and_value_fn=self.evaluator.evaluate,
            rollout_policy_fn=None,
        )
        action = select_action_from_mcts(root, temperature=self.temperature)
        return PASS if action is None else action


# ----- Mode 4 helper: batch_evaluate on a shared engine -------------------------


class _BatchedLeafEvaluator:
    """Wraps LocalBatchedInferenceEngine to expose a `batch_evaluate` for
    leaf-parallel C++ MCTS. Each leaf in the batch is submitted to the engine
    individually as a Future; the engine groups them with concurrent leaves
    from sibling game threads (and returns whichever it gets to first).

    `BatchedLocalNNEvaluator` (in src/) only has a single-board `evaluate`,
    so we add `batch_evaluate` here without modifying src/.
    """

    def __init__(self, engine, board_size: int) -> None:
        self.engine = engine
        self.board_size = board_size
        self._pass_index = get_pass_index(board_size)
        self._n_actions = self._pass_index + 1
        self._inner = BatchedLocalNNEvaluator(engine=engine, board_size=board_size)

    def evaluate(self, cpp_board):
        return self._inner.evaluate(cpp_board)

    def batch_evaluate(self, cpp_boards):
        # Submit every leaf, then await — engine batches across all in-flight.
        boards_np = []
        for b in cpp_boards:
            arr = b.to_numpy().astype(np.float32)
            if b.to_play() == alpha_go_cpp.GoBoard.WHITE:
                arr = np.where(arr == 1, 2.0, np.where(arr == 2, 1.0, arr))
            boards_np.append(arr)
        futures = [self.engine.submit(arr) for arr in boards_np]
        out = []
        for cpp_board, fut in zip(cpp_boards, futures):
            policy_logits, value, _ = fut.result()
            legal = cpp_board.get_legal_moves_flat()
            mask = np.full(self._n_actions, float("-inf"))
            for idx in legal:
                mask[idx] = 0.0
            mask[self._pass_index] = 0.0
            masked = policy_logits[: self._n_actions] + mask
            exp = np.exp(masked - np.max(masked))
            probs = exp / exp.sum()
            policy_dict = {idx: float(probs[idx]) for idx in legal}
            policy_dict[alpha_go_cpp.PASS_ACTION] = float(probs[self._pass_index])
            out.append((policy_dict, value))
        return out

    def close(self) -> None:
        pass


# ----- Mode runners -------------------------------------------------------------


def _play_one(black: Agent, white: Agent, max_moves: int, seed: int) -> int:
    record = play_game(
        black_agent=black, white_agent=white,
        board_size=BOARD_SIZE, seed=seed, max_moves=max_moves, komi=7.5,
    )
    return int(record.num_moves)


def run_py_mcts(args) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(args.checkpoint, BOARD_SIZE, MODEL_CONFIG, device)
    agent = PyMCTSAgent(model, device, BOARD_SIZE,
                        args.num_simulations, C_PUCT, TEMPERATURE)
    t0 = time.time()
    moves = _play_one(agent, agent, args.max_moves, seed=0)
    return dict(total_moves=moves, elapsed_seconds=time.time() - t0,
                num_games=1)


def run_cpp_mcts_seq(args) -> dict:
    ev = LeafBatchedNNEvaluator(args.checkpoint, BOARD_SIZE, MODEL_CONFIG)
    agent = CppMCTSAgent(
        evaluator=ev, num_simulations=args.num_simulations,
        c_puct=C_PUCT, temperature=TEMPERATURE, add_noise=False,
        resign_threshold=0.0, leaf_batch_size=0,
    )
    t0 = time.time()
    moves = _play_one(agent, agent, args.max_moves, seed=0)
    return dict(total_moves=moves, elapsed_seconds=time.time() - t0,
                num_games=1)


def run_cpp_batched(args, leaf_parallel: bool) -> dict:
    engine = _make_engine(
        checkpoint_path=args.checkpoint, board_size=BOARD_SIZE,
        model_config=MODEL_CONFIG, batch_size=args.engine_batch_size,
        batch_timeout_ms=args.engine_batch_timeout_ms,
    )

    def make_agent() -> CppMCTSAgent:
        if leaf_parallel:
            ev = _BatchedLeafEvaluator(engine, BOARD_SIZE)
            lbs = args.leaf_batch_size
        else:
            ev = BatchedLocalNNEvaluator(engine=engine, board_size=BOARD_SIZE,
                                         checkpoint_path=args.checkpoint)
            lbs = 0
        return CppMCTSAgent(
            evaluator=ev, num_simulations=args.num_simulations,
            c_puct=C_PUCT, temperature=TEMPERATURE, add_noise=False,
            resign_threshold=0.0, leaf_batch_size=lbs,
        )

    def play(seed: int) -> int:
        return _play_one(make_agent(), make_agent(), args.max_moves, seed)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_games) as ex:
        futures = [ex.submit(play, i) for i in range(args.num_games)]
        total_moves = sum(f.result() for f in as_completed(futures))
    elapsed = time.time() - t0
    engine.stop()
    return dict(total_moves=int(total_moves), elapsed_seconds=elapsed,
                num_games=args.num_games)


# ----- CLI ----------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode",
                   choices=["py-mcts", "cpp-mcts-seq", "cpp-batched", "cpp-batched-leaf"],
                   required=True)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--num-games", type=int, default=8)
    p.add_argument("--max-moves", type=int, default=40)
    p.add_argument("--num-simulations", type=int, default=NUM_SIMULATIONS)
    p.add_argument("--leaf-batch-size", type=int, default=8)
    p.add_argument("--engine-batch-size", type=int, default=64)
    p.add_argument("--engine-batch-timeout-ms", type=float, default=2.0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    print(f"=== mode={args.mode} sims={args.num_simulations} max_moves={args.max_moves} "
          f"num_games={args.num_games} ===", flush=True)

    if args.mode == "py-mcts":
        result = run_py_mcts(args)
    elif args.mode == "cpp-mcts-seq":
        result = run_cpp_mcts_seq(args)
    elif args.mode == "cpp-batched":
        result = run_cpp_batched(args, leaf_parallel=False)
    else:  # cpp-batched-leaf
        result = run_cpp_batched(args, leaf_parallel=True)

    elapsed = max(result["elapsed_seconds"], 1e-9)
    moves_per_sec = result["total_moves"] / elapsed
    result.update(dict(
        mode=args.mode,
        checkpoint=args.checkpoint,
        max_moves=args.max_moves,
        num_simulations=args.num_simulations,
        leaf_batch_size=(args.leaf_batch_size if args.mode == "cpp-batched-leaf" else 0),
        engine_batch_size=args.engine_batch_size,
        engine_batch_timeout_ms=args.engine_batch_timeout_ms,
        moves_per_sec=moves_per_sec,
        # Headline metric: NN evaluations per second is what actually scales
        # with batching/leaf-parallel optimizations (moves/sec confounds it
        # with sims/move).
        simulations_per_sec=moves_per_sec * args.num_simulations,
    ))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
