"""Read per-mode throughput JSONs and produce the bar chart + report."""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
EXP_NAME = EXP_DIR.name
GAME_DATA_DIR = Path(os.environ.get("GAME_DATA_DIR", "/nfs/game_data_root"))
SRC_DIR = GAME_DATA_DIR / "experiments" / EXP_NAME / "results"
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

MODES = [
    ("py-mcts",          "1. Python MCTS\n(single-thread)"),
    ("cpp-mcts-seq",     "2. C++ MCTS\n+ py inference"),
    ("cpp-batched",      "3. game-parallel\n+ batched py inf"),
    ("cpp-batched-leaf", "4. game-parallel\n+ leaf-parallel\n+ batched py inf"),
]


def main() -> None:
    rows = []
    for slug, _label in MODES:
        path = SRC_DIR / f"{slug}.json"
        if not path.exists():
            print(f"missing {path} — skipping")
            rows.append(None)
            continue
        rows.append(json.loads(path.read_text()))

    # Headline metric is simulations / sec — older result JSONs only carry
    # `moves_per_sec` and `num_simulations`, so derive sims/sec from those
    # when the field isn't present.
    def _sims_per_sec(r):
        if "simulations_per_sec" in r:
            return float(r["simulations_per_sec"])
        return float(r["moves_per_sec"]) * float(r["num_simulations"])

    fig, ax = plt.subplots(figsize=(8, 5))
    xs = list(range(len(MODES)))
    labels = [lbl for _, lbl in MODES]
    vals = [(_sims_per_sec(r) if r else 0.0) for r in rows]
    ax.bar(xs, vals, color=["#bbb", "#88c", "#4c8", "#2a6"], edgecolor="black")
    for x, v, r in zip(xs, vals, rows):
        if r is None:
            continue
        ax.text(x, v * 1.02 + 0.05, f"{v:,.0f}", ha="center", va="bottom", fontsize=10)
        sub = (f"n_games={r['num_games']}, moves={r['total_moves']}, "
               f"sims={r['num_simulations']}, {r['elapsed_seconds']:.1f}s")
        ax.text(x, v * 0.5, sub, ha="center", va="center", fontsize=8,
                color="white" if v > max(vals) * 0.2 else "black")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MCTS simulations / sec (19x19, 1024 sims/move)")
    ax.set_title(f"{EXP_NAME}\nself-play MCTS throughput (higher = better)")
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.4, which="both")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "simulations_per_sec.png", dpi=140)
    plt.close(fig)

    # Report
    lines = [
        f"# {EXP_NAME}",
        "",
        "## Goal",
        "Quantify MCTS simulations/sec at each level of optimization on a 19x19 board.",
        "",
        "## Setup",
        "- Self-play (agent vs itself), 1024 sims/move, c_puct=0.5, temperature=0.3.",
        "- Resignation disabled. Komi 7.5. max_moves=40 per game.",
        "- Model: SizeInvariantGoResNet 18M, "
        "/nfs/checkpoints/2026-04-22_12-11-learngo-19x19-9x9-v0/iter12_best.pt",
        "- Modes 3-4: 8 parallel game threads sharing one "
        "`LocalBatchedInferenceEngine` (batch_size=64, timeout=2ms).",
        "- Mode 4 leaf_batch_size=8 (virtual-loss leaf parallel inside C++ MCTS).",
        "",
        "## Results",
        "",
        "| # | mode | num_games | total_moves | elapsed_s | moves/sec | sims/sec |",
        "|---|------|----------:|------------:|----------:|----------:|---------:|",
    ]
    for (slug, _), r in zip(MODES, rows):
        if r is None:
            lines.append(f"| | {slug} | — | — | — | — | — |")
        else:
            lines.append(
                f"| {slug.split('-')[0]} | {slug} | {r['num_games']} "
                f"| {r['total_moves']} | {r['elapsed_seconds']:.1f} "
                f"| {r['moves_per_sec']:.2f} "
                f"| **{_sims_per_sec(r):,.0f}** |"
            )
    decided = [r for r in rows if r is not None]
    if len(decided) >= 2:
        speedup = _sims_per_sec(decided[-1]) / _sims_per_sec(decided[0])
        lines += [
            "",
            f"End-to-end speedup (mode 4 / mode 1, sims/sec): **{speedup:.1f}x**",
        ]
    lines += [
        "",
        "Figures:",
        "- `figures/simulations_per_sec.png`",
        "",
        "## Key findings",
        "- TBD (fill in after run completes)",
        "",
    ]
    (EXP_DIR / "report.md").write_text("\n".join(lines))
    print(f"wrote {FIG_DIR}/simulations_per_sec.png and {EXP_DIR/'report.md'}")


if __name__ == "__main__":
    main()
