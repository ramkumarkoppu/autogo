"""Plot Phase A and Phase B progress curves to figures/.

Layout: scatter on the left (run index → holdout acc), description list on
the right (so labels don't overlap). Discards render as hollow gray Xs;
keeps as filled circles; best run flagged with an orange star. Reference
lines (parent iter1 / iter10, Phase A best) anchor the right axis with
inline labels.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

EXP_DIR = Path(__file__).resolve().parent
FIG_DIR = EXP_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

# Colour palette (matplotlib tab10-ish, bumped for legibility on white).
KEEP_COLOR    = "#2ca02c"
DISCARD_COLOR = "#9e9e9e"
BEST_COLOR    = "#ff7f0e"
RUN_BEST_COL  = "#1f77b4"
REF_PARENT10  = "#d62728"
REF_PARENT1   = "#9467bd"
REF_PHASEA    = "#1f77b4"


def _load_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _setup_axes(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", alpha=0.25, linestyle="-", linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_alpha(0.4)
    ax.spines["bottom"].set_alpha(0.4)


def _ref_line(ax: plt.Axes, y: float, label: str, color: str, x_max: int,
              ls: str = "--", offset_pad: float = 0.12) -> None:
    """Horizontal reference line with an inline label past the right edge."""
    ax.axhline(y, color=color, linestyle=ls, linewidth=1.1, alpha=0.7, zorder=1)
    ax.text(x_max + offset_pad, y, f"{label}\n{y:.4f}", color=color,
            fontsize=8.5, ha="left", va="center", fontweight="medium")


def _draw_scatter(ax: plt.Axes, rows: list[dict], y_field: str,
                  ref_lines: list[tuple[float, str, str]],
                  title: str, subtitle: str) -> None:
    xs, ys, statuses = [], [], []
    for i, r in enumerate(rows):
        try:
            y = float(r[y_field])
        except (KeyError, ValueError):
            continue
        xs.append(i + 1)         # 1-based for human readability
        ys.append(y)
        statuses.append(r["status"])

    if not xs:
        return

    keep_xs   = [x for x, s in zip(xs, statuses) if s == "keep"]
    keep_ys   = [y for y, s in zip(ys, statuses) if s == "keep"]
    disc_xs   = [x for x, s in zip(xs, statuses) if s != "keep"]
    disc_ys   = [y for y, s in zip(ys, statuses) if s != "keep"]
    best_i    = max(range(len(ys)), key=lambda k: ys[k])
    best_x, best_y = xs[best_i], ys[best_i]

    # Running-best step trace + light fill underneath.
    rb = []
    cur = float("-inf")
    for y in ys:
        cur = max(cur, y)
        rb.append(cur)
    ax.fill_between(xs, [min(ys) - 0.005] * len(xs), rb, color=RUN_BEST_COL,
                    alpha=0.07, zorder=1)
    ax.step(xs, rb, where="post", color=RUN_BEST_COL, linewidth=1.5,
            alpha=0.55, label="running best", zorder=2)

    # Reference lines (parent baseline, etc.) anchored to the right edge.
    x_max = max(xs)
    for y_ref, label_ref, color_ref in ref_lines:
        _ref_line(ax, y_ref, label_ref, color_ref, x_max)

    # Discards: hollow gray X, lighter weight.
    ax.scatter(disc_xs, disc_ys, marker="x", color=DISCARD_COLOR,
               s=70, linewidths=1.6, alpha=0.85, label="discard", zorder=3)
    # Keeps: filled green circle, white edge for pop.
    ax.scatter(keep_xs, keep_ys, marker="o", color=KEEP_COLOR,
               s=120, edgecolor="white", linewidths=1.6,
               label="keep", zorder=4)
    # Best: orange star.
    ax.scatter([best_x], [best_y], marker="*", color=BEST_COLOR,
               s=420, edgecolor="black", linewidths=0.9,
               label="best", zorder=5)
    ax.annotate(f"{best_y:.4f}", (best_x, best_y), fontsize=10,
                fontweight="bold", color=BEST_COLOR, ha="center",
                xytext=(0, 14), textcoords="offset points")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs], fontsize=9)
    ax.set_xlim(0.5, x_max + 0.5)
    y_lo = min(min(ys), min(y for y, _, _ in ref_lines))
    y_hi = max(max(ys), max(y for y, _, _ in ref_lines))
    margin = (y_hi - y_lo) * 0.18 + 0.005
    ax.set_ylim(y_lo - margin, y_hi + margin)

    ax.set_xlabel("Run #", fontsize=11)
    ax.set_ylabel("Holdout policy_acc", fontsize=11)
    # Stack title above subtitle in axes-relative coords so they always sit
    # above the plot frame regardless of dpi or figure size.
    ax.text(0.0, 1.12, title, transform=ax.transAxes,
            fontsize=14, fontweight="bold", ha="left", va="bottom")
    if subtitle:
        ax.text(0.0, 1.03, subtitle, transform=ax.transAxes,
                fontsize=10, color="#555555", ha="left", va="bottom")

    leg = ax.legend(loc="lower right", frameon=True, fancybox=True,
                    framealpha=0.95, fontsize=9, borderpad=0.6,
                    handletextpad=0.5)
    leg.get_frame().set_edgecolor("#cccccc")


def _draw_run_table(ax: plt.Axes, rows: list[dict], y_field: str) -> None:
    """Right-side text panel: numbered list of runs with commit + description."""
    ax.axis("off")
    ax.text(0, 1.12, "Runs", transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom")

    line_h = 1.0 / max(len(rows), 1)
    for i, r in enumerate(rows):
        try:
            y = float(r[y_field])
        except (KeyError, ValueError):
            y = float("nan")
        keep = r["status"] == "keep"
        color = KEEP_COLOR if keep else DISCARD_COLOR
        weight = "semibold" if keep else "normal"
        commit = r["commit"][:7]
        # Truncate description to keep the panel tidy.
        desc = r["description"]
        if len(desc) > 78:
            desc = desc[:75] + "…"
        # Bullet · marker · run-num · commit · acc · desc
        bullet = "●" if keep else "○"
        line = f" {i+1:>2}  {bullet}  {commit}  {y:.4f}   {desc}"
        ax.text(0, 1 - (i + 1) * line_h + line_h * 0.05, line,
                transform=ax.transAxes,
                fontsize=8.2, family="monospace", color=color,
                fontweight=weight, va="top")


def plot_phaseA() -> None:
    rows = _load_tsv(EXP_DIR / "results-phaseA.tsv")
    if not rows:
        return
    n_keep = sum(1 for r in rows if r["status"] == "keep")
    best = max(rows, key=lambda r: float(r["holdout_policy_acc"]))
    subtitle = (f"{len(rows)} runs · {n_keep} kept · "
                f"best = {float(best['holdout_policy_acc']):.4f} "
                f"({best['commit'][:7]}, {best['description'][:38]}…)")

    fig = plt.figure(figsize=(15, 6.5), dpi=130)
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, width_ratios=[2.0, 1.05], wspace=0.25,
                  left=0.06, right=0.99, top=0.82, bottom=0.12)
    ax = fig.add_subplot(gs[0]); _setup_axes(ax)
    ax_t = fig.add_subplot(gs[1])

    _draw_scatter(
        ax, rows, "holdout_policy_acc",
        ref_lines=[(0.3034, "parent iter10", REF_PARENT10)],
        title="Phase A: hyperparameter tuning on dataset-it10",
        subtitle=subtitle,
    )
    _draw_run_table(ax_t, rows, "holdout_policy_acc")

    out = FIG_DIR / "phaseA_progress.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_phaseB() -> None:
    rows = _load_tsv(EXP_DIR / "results.tsv")
    if not rows:
        return
    n_keep = sum(1 for r in rows if r["status"] == "keep")
    best = max(rows, key=lambda r: float(r["holdout_policy_acc_it1"]))
    subtitle = (f"{len(rows)} runs · {n_keep} kept · "
                f"best = {float(best['holdout_policy_acc_it1']):.4f} "
                f"({best['commit'][:7]}, {best['description'][:38]}…)")

    fig = plt.figure(figsize=(15, 7.5), dpi=130)
    fig.patch.set_facecolor("white")
    gs = GridSpec(1, 2, width_ratios=[2.0, 1.05], wspace=0.25,
                  left=0.06, right=0.99, top=0.84, bottom=0.10)
    ax = fig.add_subplot(gs[0]); _setup_axes(ax)
    ax_t = fig.add_subplot(gs[1])

    _draw_scatter(
        ax, rows, "holdout_policy_acc_it1",
        ref_lines=[
            (0.3078, "Phase A best (it0)", REF_PHASEA),
            (0.3034, "parent iter10",       REF_PARENT10),
            (0.1014, "parent iter1",        REF_PARENT1),
        ],
        title="Phase B: collect-then-train one iteration from Phase A best",
        subtitle=subtitle,
    )
    _draw_run_table(ax_t, rows, "holdout_policy_acc_it1")

    out = FIG_DIR / "phaseB_progress.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    plot_phaseA()
    plot_phaseB()
