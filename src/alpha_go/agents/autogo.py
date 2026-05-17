"""Registration for Eric Jang's published iter12 checkpoint (autogo).

Loads checkpoints/iter12_best.pt from the project root.
LeafBatchedNNEvaluator auto-detects the SizeInvariantGoResNet architecture
from the state-dict fingerprint, so no architecture args are needed.
"""
from __future__ import annotations

from pathlib import Path

from alpha_go.agents.base import register_agent
from alpha_go.agents.nn_mcts import CppMCTSAgent, LeafBatchedNNEvaluator

_CKPT = Path(__file__).resolve().parents[3] / "checkpoints" / "iter12_best.pt"


@register_agent("autogo")
class AutoGoMCTSAgent(CppMCTSAgent):
    """iter12_best.pt wrapped in C++ MCTS for stronger interactive play."""

    def __init__(self) -> None:
        evaluator = LeafBatchedNNEvaluator(checkpoint_path=str(_CKPT), board_size=9)
        super().__init__(
            evaluator=evaluator,
            num_simulations=128,
            c_puct=1.0,
            temperature=0.3,
            resign_threshold=0.02,
            resign_consec_turns=8,
        )


@register_agent("autogo-policy")
class AutoGoPolicyAgent(CppMCTSAgent):
    """iter12_best.pt with minimal MCTS (1 sim ≈ raw policy). Instant moves."""

    def __init__(self) -> None:
        evaluator = LeafBatchedNNEvaluator(checkpoint_path=str(_CKPT), board_size=9)
        super().__init__(evaluator=evaluator, num_simulations=1, temperature=1.0)
