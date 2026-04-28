"""Plot parent's iter0..iter10 holdout_policy_acc curve alongside the
fastlearn experiment's own from-scratch training curve and the curated
Phase A / Phase B sweep bests."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
PARENT_HE = Path("/workspace/experiments/2026-04-27_16-31-train-fromscratch-champion/holdout_eval")
OURS_HE = EXP_DIR / "holdout_eval"


# `iter10_best.pt` in fastlearn is the Phase-A sweep winner staged as
# iter10 so it can flow through the standard `train.py --eval-only` path;
# it is NOT a real 10th from-scratch iteration. Plot it on its own marker
# instead of folding it into the from-scratch line.
PHASE_A_STAGED_ITERS = {10}


def _load_curve(holdout_dir: Path,
                exclude_iters: set[int] | None = None) -> tuple[list[int], list[float]]:
    iters, vals = [], []
    excl = exclude_iters or set()
    for f in sorted(holdout_dir.glob("it*.json"), key=lambda p: int(p.stem[2:])):
        d = json.loads(f.read_text())
        if d.get("policy_acc") is None or d["iteration"] in excl:
            continue
        iters.append(d["iteration"])
        vals.append(d["policy_acc"])
    return iters, vals


def _load_one(holdout_dir: Path, iteration: int) -> float | None:
    f = holdout_dir / f"it{iteration}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("policy_acc")


def main() -> None:
    p_iters, p_vals = _load_curve(PARENT_HE)
    o_iters, o_vals = _load_curve(OURS_HE, exclude_iters=PHASE_A_STAGED_ITERS)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(p_iters, p_vals, "o-", color="gray",
            label=f"parent: train-fromscratch-champion ({len(p_iters)} iters)")
    if 1 < len(p_vals):
        ax.annotate(f"{p_vals[1]:.3f}", (p_iters[1], p_vals[1]), fontsize=9,
                    xytext=(5, -15), textcoords="offset points")
    ax.annotate(f"{p_vals[-1]:.3f}", (p_iters[-1], p_vals[-1]), fontsize=9,
                xytext=(5, -15), textcoords="offset points")

    # Fastlearn's own from-scratch training curve.
    if o_iters:
        ax.plot(o_iters, o_vals, "D-", color="C0", linewidth=2,
                markersize=8,
                label=f"fastlearn (this run): {len(o_iters)} iters from-scratch")
        for i, v in zip(o_iters, o_vals):
            ax.annotate(f"{v:.3f}", (i, v), fontsize=9, color="C0",
                        xytext=(5, 5), textcoords="offset points")

    # Phase-A sweep winner — pulled out as its own marker so it is not
    # mistaken for the 10th from-scratch iteration. Eval'd via the same
    # holdout pipeline (file `it{iter}.json`); the staged iter is whatever
    # PHASE_A_STAGED_ITERS lists.
    for staged_iter in sorted(PHASE_A_STAGED_ITERS):
        v = _load_one(OURS_HE, staged_iter)
        if v is None:
            continue
        ax.plot([staged_iter], [v], "*", color="C1", markersize=18,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"Phase A best (staged as iter{staged_iter})")
        ax.annotate(f"{v:.3f}", (staged_iter, v), fontsize=10, color="C1",
                    fontweight="bold",
                    xytext=(8, -2), textcoords="offset points")

    # Curated sweep bests (Phase A best + Phase B baseline iter1) — kept as
    # reference points so the plot also shows what hand-tuned hyperparams
    # achieved relative to the unattended fromscratch loop.
    sweep_iters = [0, 1]
    sweep_vals = [0.3078, 0.3377]
    ax.plot(sweep_iters, sweep_vals, "s-", color="C2", linewidth=2.5,
            markersize=12, label="ours: Phase A best + Phase B baseline (sweep)")
    for i, v in zip(sweep_iters, sweep_vals):
        ax.annotate(f"{v:.3f}", (i, v), fontsize=10, color="C2",
                    fontweight="bold",
                    xytext=(5, -15), textcoords="offset points")

    last_x = max(p_iters[-1] if p_iters else 0,
                 o_iters[-1] if o_iters else 0,
                 max(PHASE_A_STAGED_ITERS, default=0),
                 sweep_iters[-1])
    ax.set_xlabel("iteration")
    ax.set_ylabel("holdout policy_acc (selfplay-it40 from v7)")
    ax.set_title("Holdout policy accuracy: parent vs fastlearn (from-scratch + sweep)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, last_x + 1))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "comparison_vs_parent.png", dpi=110)
    plt.close(fig)
    print(f"Wrote {FIG_DIR / 'comparison_vs_parent.png'}")


if __name__ == "__main__":
    main()
